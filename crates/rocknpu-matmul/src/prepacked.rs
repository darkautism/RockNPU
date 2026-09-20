use super::*;
use rocket_runtime::{RocketBuffer, RocketDevice, Task};
use std::os::fd::RawFd;
use std::sync::OnceLock;
use std::time::Instant;

const WEIGHT_ALIGN: usize = 4096;

fn prefill_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ROCKNPU_PREFILL_PROFILE")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrepackedWeightStats {
    pub unique_tiles: usize,
    pub resident_bytes: usize,
    pub total_ns: u128,
    pub plan_ns: u128,
    pub layout_ns: u128,
    pub alloc_mmap_ns: u128,
    pub prep_ns: u128,
    pub zero_ns: u128,
    pub tile_pack_ns: u128,
    pub fini_ns: u128,
    pub pack_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WeightTile {
    n0: usize,
    k0: usize,
    n: usize,
    k: usize,
    offset: usize,
    bytes: usize,
}

pub struct Fp16PrepackedWeights<'a> {
    device_fd: RawFd,
    plan: Fp16MatmulPlan,
    bo: RocketBuffer<'a>,
    tiles: Vec<WeightTile>,
    stats: PrepackedWeightStats,
    compatible_m: bool,
}

impl Fp16PrepackedWeights<'_> {
    pub fn m(&self) -> usize {
        self.plan.m
    }
    pub fn k(&self) -> usize {
        self.plan.k
    }
    pub fn n(&self) -> usize {
        self.plan.n
    }
    pub fn stats(&self) -> PrepackedWeightStats {
        self.stats
    }
    pub fn dma_address(&self) -> u64 {
        self.bo.dma_address()
    }
    pub fn handle(&self) -> u32 {
        self.bo.handle()
    }
    /// True when this resident layout uses conservative N/K tiling that can be
    /// reused across any aligned M through `execute_prepacked_compatible_m`.
    pub fn is_m_compatible(&self) -> bool {
        self.compatible_m
    }
    pub fn kt(&self) -> usize {
        self.plan.kt
    }
    pub fn nt(&self) -> usize {
        self.plan.nt
    }

    fn tile(&self, tile: Fp16MatmulTile) -> Result<(u64, u32), MatmulError> {
        let entry = self
            .tiles
            .iter()
            .find(|entry| {
                entry.n0 == tile.n0 && entry.k0 == tile.k0 && entry.n == tile.n && entry.k == tile.k
            })
            .ok_or(MatmulError::Internal("prepacked weight tile missing"))?;
        let dma = self
            .bo
            .dma_address()
            .checked_add(entry.offset as u64)
            .ok_or(MatmulError::Internal("prepacked weight DMA overflow"))?;
        if dma > u32::MAX as u64 {
            return Err(MatmulError::AddressAbove32Bit(dma));
        }
        Ok((dma, self.bo.handle()))
    }
}

fn align_up(value: usize, align: usize) -> Result<usize, MatmulError> {
    let add = align
        .checked_sub(1)
        .ok_or(MatmulError::Internal("invalid weight alignment"))?;
    value
        .checked_add(add)
        .map(|v| v / align * align)
        .ok_or(MatmulError::Internal("prepacked weight size overflow"))
}

fn unique_weight_layout(plan: &Fp16MatmulPlan) -> Result<(Vec<WeightTile>, usize), MatmulError> {
    let mut out = Vec::new();
    let mut next = 0usize;
    for &tile in &plan.tiles {
        if out.iter().any(|entry: &WeightTile| {
            entry.n0 == tile.n0 && entry.k0 == tile.k0 && entry.n == tile.n && entry.k == tile.k
        }) {
            continue;
        }
        let bytes = tile
            .n
            .checked_mul(tile.k)
            .and_then(|v| v.checked_mul(2))
            .ok_or(MatmulError::Internal("prepacked weight tile size overflow"))?;
        next = align_up(next, WEIGHT_ALIGN)?;
        out.push(WeightTile {
            n0: tile.n0,
            k0: tile.k0,
            n: tile.n,
            k: tile.k,
            offset: next,
            bytes,
        });
        next = next.checked_add(bytes).ok_or(MatmulError::Internal(
            "prepacked weight total size overflow",
        ))?;
    }
    Ok((out, next.max(1)))
}

impl<'a> Fp16MatmulExecutor<'a> {
    /// High-accuracy fp16-input MatMul. Every NPU K partial is emitted as
    /// fp32 (C2=4), then K partials are accumulated on the host in f64 and
    /// narrowed once to f32. This intentionally avoids EW accumulation entirely.
    pub fn execute_prepacked_f32(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
    ) -> Result<Fp32MatmulOutput, MatmulError> {
        let total_start = Instant::now();
        let (m, k, n) = (weights.m(), weights.k(), weights.n());
        if weights.device_fd != self.device.fd()
            || weights.compatible_m
            || a.len() != m.saturating_mul(k)
        {
            return Err(MatmulError::InvalidInput(
                "exact prepacked FP32 shape/device mismatch",
            ));
        }
        let plan_start = Instant::now();
        let plan = weights.plan.clone();
        let mut timing = ExecutionTiming {
            plan_ns: plan_start.elapsed().as_nanos(),
            ..ExecutionTiming::default()
        };
        let k_tiles = plan.k_tiles();
        if k_tiles == 0 || plan.tiles.len() % k_tiles != 0 {
            return Err(MatmulError::Internal(
                "planner tile grouping is not rectangular",
            ));
        }
        let mut values = vec![0.0f32; m * n];
        let mut jobs = 0usize;
        let mut groups = 0usize;
        let mut host_kacc_groups = 0usize;

        for group in plan.tiles.chunks(k_tiles) {
            groups += 1;
            validate_group(group, k)?;
            let first = group[0];
            let mut acc = vec![0.0f64; first.m * first.n];
            for &tile in group {
                let phase = Instant::now();
                self.ensure_scratch(tile, 4, false)?;
                timing.scratch_ns += phase.elapsed().as_nanos();
                let (weight_dma, weight_handle) = weights.tile(tile)?;
                let ExecutorScratch {
                    regcmd: Some(regcmd),
                    input: Some(input),
                    output0: Some(output),
                    ..
                } = &mut self.scratch
                else {
                    return Err(MatmulError::Internal("fp32 scratch allocation invariant"));
                };
                let phase = Instant::now();
                pack_input(input, a, k, tile)?;
                prepare_output(output)?;
                timing.pack_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                let ops = encode_fp16_matmul_fp32_output(Fp16MatmulDesc::new(
                    tile.m,
                    tile.k,
                    tile.n,
                    input.dma_address(),
                    weight_dma,
                    output.dma_address(),
                ))?;
                timing.encode_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                write_regcmd(regcmd, &ops)?;
                timing.regcmd_write_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                submit_plain_weight(self.device, regcmd, input, weight_handle, output, ops.len())?;
                timing.submit_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                output.prep_relative(WAIT_NS)?;
                timing.wait_ns += phase.elapsed().as_nanos();
                let phase = Instant::now();
                accumulate_fp32_cube(&mut acc, output.as_slice(), tile.m, tile.n);
                timing.gather_ns += phase.elapsed().as_nanos();
                output.fini()?;
                jobs += 1;
            }
            if group.len() > 1 {
                host_kacc_groups += 1;
            }
            for tm in 0..first.m {
                for tn in 0..first.n {
                    values[(first.m0 + tm) * n + first.n0 + tn] = acc[tm * first.n + tn] as f32;
                }
            }
        }

        timing.total_ns = total_start.elapsed().as_nanos();
        Ok(Fp32MatmulOutput {
            values,
            stats: ExecutionStats {
                plan,
                jobs_submitted: jobs,
                output_tile_groups: groups,
                npu_kacc_groups: 0,
                host_kacc_groups,
                timing,
            },
        })
    }

    /// Pack B[N,K] once into a resident BO for the exact planner geometry of M/K/N.
    /// The native weight tiles are deduplicated across M tiles because their layout
    /// depends only on the N/K tile geometry.
    pub fn prepack_weights(
        &self,
        b: &[f16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Fp16PrepackedWeights<'a>, MatmulError> {
        let total_start = prefill_profile_enabled().then(Instant::now);
        let plan_start = prefill_profile_enabled().then(Instant::now);
        let plan = plan_fp16_matmul(m, k, n)?;
        let plan_ns = plan_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let mut prepared = self.prepack_weights_plan(b, k, n, plan, false)?;
        prepared.stats.plan_ns = plan_ns;
        prepared.stats.total_ns = total_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        Ok(prepared)
    }

    /// Pack B[N,K] once using an M-invariant N/K tile geometry.
    ///
    /// The planner chooses Kt against a worst-case 256-row feature tile, so the
    /// resulting native weight tiles can be reused for any aligned M. Small-M
    /// execution can submit more K tiles than the exact planner; this is the
    /// deliberate cost of avoiding weight repacking when batch/M changes.
    pub fn prepack_weights_compatible_m(
        &self,
        b: &[f16],
        k: usize,
        n: usize,
    ) -> Result<Fp16PrepackedWeights<'a>, MatmulError> {
        let total_start = prefill_profile_enabled().then(Instant::now);
        let plan_start = prefill_profile_enabled().then(Instant::now);
        let plan = plan_fp16_matmul_compatible_m(4, k, n)?;
        let plan_ns = plan_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let mut prepared = self.prepack_weights_plan(b, k, n, plan, true)?;
        prepared.stats.plan_ns = plan_ns;
        prepared.stats.total_ns = total_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        Ok(prepared)
    }

    fn prepack_weights_plan(
        &self,
        b: &[f16],
        k: usize,
        n: usize,
        plan: Fp16MatmulPlan,
        compatible_m: bool,
    ) -> Result<Fp16PrepackedWeights<'a>, MatmulError> {
        if b.len()
            != n.checked_mul(k)
                .ok_or(MatmulError::InvalidInput("B size overflow"))?
        {
            return Err(MatmulError::InvalidInput(
                "B must contain exactly N*K elements",
            ));
        }
        let profiling = prefill_profile_enabled();
        let layout_start = profiling.then(Instant::now);
        let (tiles, total_bytes) = unique_weight_layout(&plan)?;
        let layout_ns = layout_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let unique_tiles = tiles.len();
        let alloc_start = profiling.then(Instant::now);
        let mut bo = self.device.alloc_buffer(total_bytes)?;
        let alloc_mmap_ns = alloc_start.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let last = bo
            .dma_address()
            .checked_add((total_bytes - 1) as u64)
            .ok_or(MatmulError::Internal("prepacked weight IOVA overflow"))?;
        if last > u32::MAX as u64 {
            return Err(MatmulError::AddressAbove32Bit(last));
        }
        let pack_start = Instant::now();
        let phase = profiling.then(Instant::now);
        bo.prep_relative(0)?;
        let prep_ns = phase.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let phase = profiling.then(Instant::now);
        bo.as_mut_slice().fill(0);
        let zero_ns = phase.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let phase = profiling.then(Instant::now);
        for entry in &tiles {
            let tile = Fp16MatmulTile {
                m0: 0,
                n0: entry.n0,
                k0: entry.k0,
                m: 4,
                n: entry.n,
                k: entry.k,
            };
            let dst = &mut bo.as_mut_slice()[entry.offset..entry.offset + entry.bytes];
            pack_weight_bytes(dst, b, k, tile);
        }
        let tile_pack_ns = phase.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let phase = profiling.then(Instant::now);
        bo.fini()?;
        let fini_ns = phase.map(|started| started.elapsed().as_nanos()).unwrap_or(0);
        let pack_ns = pack_start.elapsed().as_nanos();
        Ok(Fp16PrepackedWeights {
            device_fd: self.device.fd(),
            plan,
            bo,
            tiles,
            stats: PrepackedWeightStats {
                unique_tiles,
                resident_bytes: total_bytes,
                total_ns: 0,
                plan_ns: 0,
                layout_ns,
                alloc_mmap_ns,
                prep_ns,
                zero_ns,
                tile_pack_ns,
                fini_ns,
                pack_ns,
            },
            compatible_m,
        })
    }

    /// Execute with a resident weight handle produced by `prepack_weights`.
    /// A is still packed per call; B is never repacked or CPU-touched here.
    pub fn execute_prepacked(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
    ) -> Result<Fp16MatmulOutput, MatmulError> {
        let total_start = Instant::now();
        let m = weights.plan.m;
        let k = weights.plan.k;
        let n = weights.plan.n;
        if a.len()
            != m.checked_mul(k)
                .ok_or(MatmulError::InvalidInput("A size overflow"))?
        {
            return Err(MatmulError::InvalidInput(
                "A must contain exactly M*K elements",
            ));
        }
        let plan_start = Instant::now();
        let plan = plan_fp16_matmul(m, k, n)?;
        let plan_ns = plan_start.elapsed().as_nanos();
        if plan != weights.plan || weights.compatible_m {
            return Err(MatmulError::InvalidInput("prepacked weight plan mismatch"));
        }
        self.execute_prepacked_plan(a, weights, plan, plan_ns, total_start)
    }

    /// Execute an aligned A[M,K] against M-compatible resident weights.
    /// M may differ on every call; K/N and resident N/K tile geometry remain fixed.
    pub fn execute_prepacked_compatible_m(
        &mut self,
        a: &[f16],
        m: usize,
        weights: &Fp16PrepackedWeights<'a>,
    ) -> Result<Fp16MatmulOutput, MatmulError> {
        let total_start = Instant::now();
        if !weights.compatible_m {
            return Err(MatmulError::InvalidInput(
                "resident weights were not prepared for M-compatible execution",
            ));
        }
        let k = weights.plan.k;
        let n = weights.plan.n;
        if a.len()
            != m.checked_mul(k)
                .ok_or(MatmulError::InvalidInput("A size overflow"))?
        {
            return Err(MatmulError::InvalidInput(
                "A must contain exactly M*K elements",
            ));
        }
        let plan_start = Instant::now();
        let plan = plan_fp16_matmul_compatible_m(m, k, n)?;
        let plan_ns = plan_start.elapsed().as_nanos();
        if plan.kt != weights.plan.kt || plan.nt != weights.plan.nt {
            return Err(MatmulError::InvalidInput(
                "M-compatible resident weight geometry mismatch",
            ));
        }
        self.execute_prepacked_plan(a, weights, plan, plan_ns, total_start)
    }

    fn execute_prepacked_plan(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
        plan: Fp16MatmulPlan,
        plan_ns: u128,
        total_start: Instant,
    ) -> Result<Fp16MatmulOutput, MatmulError> {
        if self.device.fd() != weights.device_fd {
            return Err(MatmulError::InvalidInput(
                "prepacked weights belong to a different Rocket fd",
            ));
        }
        let m = plan.m;
        let k = plan.k;
        let n = plan.n;
        let mut timing = ExecutionTiming {
            plan_ns,
            ..ExecutionTiming::default()
        };
        let k_tiles = plan.k_tiles();
        if k_tiles == 0 || plan.tiles.len() % k_tiles != 0 {
            return Err(MatmulError::Internal(
                "prepacked planner tile grouping is not rectangular",
            ));
        }

        let mut values = vec![f16::ZERO; m * n];
        let mut jobs = 0usize;
        let mut groups = 0usize;
        let mut npu_kacc_groups = 0usize;
        let mut host_kacc_groups = 0usize;
        for group in plan.tiles.chunks(k_tiles) {
            groups += 1;
            validate_group(group, k)?;
            match accumulation_mode(group) {
                KAccumulation::None => {
                    self.execute_prepacked_single(
                        a,
                        weights,
                        k,
                        n,
                        group[0],
                        &mut values,
                        &mut timing,
                    )?;
                    jobs += 1;
                }
                KAccumulation::NpuFp16PingPong => {
                    self.execute_prepacked_npu_kacc(
                        a,
                        weights,
                        k,
                        n,
                        group,
                        &mut values,
                        &mut timing,
                    )?;
                    jobs += group.len();
                    npu_kacc_groups += 1;
                }
                KAccumulation::HostFp32TinyM => {
                    self.execute_prepacked_host_kacc(
                        a,
                        weights,
                        k,
                        n,
                        group,
                        &mut values,
                        &mut timing,
                    )?;
                    jobs += group.len();
                    host_kacc_groups += 1;
                }
            }
        }
        timing.total_ns = total_start.elapsed().as_nanos();
        Ok(Fp16MatmulOutput {
            values,
            stats: ExecutionStats {
                plan,
                jobs_submitted: jobs,
                output_tile_groups: groups,
                npu_kacc_groups,
                host_kacc_groups,
                timing,
            },
        })
    }

    fn execute_prepacked_single(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
        k_total: usize,
        n_total: usize,
        tile: Fp16MatmulTile,
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let phase = Instant::now();
        self.ensure_scratch(tile, 2, false)?;
        timing.scratch_ns += phase.elapsed().as_nanos();
        let (weight_dma, weight_handle) = weights.tile(tile)?;
        let ExecutorScratch {
            regcmd: Some(regcmd),
            input: Some(input),
            output0: Some(output),
            ..
        } = &mut self.scratch
        else {
            return Err(MatmulError::Internal("prepacked single scratch invariant"));
        };
        let phase = Instant::now();
        pack_input(input, a, k_total, tile)?;
        prepare_output(output)?;
        timing.pack_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
            tile.m,
            tile.k,
            tile.n,
            input.dma_address(),
            weight_dma,
            output.dma_address(),
        ))?;
        timing.encode_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        write_regcmd(regcmd, &ops)?;
        timing.regcmd_write_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        submit_plain_weight(self.device, regcmd, input, weight_handle, output, ops.len())?;
        timing.submit_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        output.prep_relative(WAIT_NS)?;
        timing.wait_ns += phase.elapsed().as_nanos();
        let phase = Instant::now();
        gather_tile(output.as_slice(), tile, n_total, dst);
        timing.gather_ns += phase.elapsed().as_nanos();
        output.fini()?;
        Ok(())
    }

    fn execute_prepacked_npu_kacc(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
        k_total: usize,
        n_total: usize,
        group: &[Fp16MatmulTile],
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let first = group[0];
        if first.m < 12 {
            return Err(MatmulError::Internal(
                "tiny-M group routed to prepacked NPU KACC",
            ));
        }
        let phase = Instant::now();
        self.ensure_scratch(first, 2, true)?;
        timing.scratch_ns += phase.elapsed().as_nanos();
        {
            let ExecutorScratch {
                output0: Some(ping),
                output1: Some(pong),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal(
                    "prepacked KACC output scratch invariant",
                ));
            };
            let phase = Instant::now();
            prepare_output(ping)?;
            prepare_output(pong)?;
            timing.pack_ns += phase.elapsed().as_nanos();
        }
        for (ki, &tile) in group.iter().enumerate() {
            let phase = Instant::now();
            self.ensure_scratch(tile, 2, true)?;
            timing.scratch_ns += phase.elapsed().as_nanos();
            let (weight_dma, weight_handle) = weights.tile(tile)?;
            let ExecutorScratch {
                regcmd: Some(regcmd),
                input: Some(input),
                output0: Some(ping),
                output1: Some(pong),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal("prepacked KACC scratch invariant"));
            };
            let phase = Instant::now();
            pack_input(input, a, k_total, tile)?;
            timing.pack_ns += phase.elapsed().as_nanos();
            let (out, add) = if ki % 2 == 0 {
                (ping, pong)
            } else {
                (pong, ping)
            };
            out.prep_relative(0)?;
            out.fini()?;
            let desc = Fp16MatmulDesc::new(
                tile.m,
                tile.k,
                tile.n,
                input.dma_address(),
                weight_dma,
                out.dma_address(),
            );
            let phase = Instant::now();
            let ops = if ki == 0 {
                encode_fp16_matmul(desc)?
            } else {
                encode_fp16_matmul_accumulate(desc, add.dma_address())?
            };
            timing.encode_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            write_regcmd(regcmd, &ops)?;
            timing.regcmd_write_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            if ki == 0 {
                submit_plain_weight(self.device, regcmd, input, weight_handle, out, ops.len())?;
            } else {
                submit_accumulate_weight(
                    self.device,
                    regcmd,
                    input,
                    weight_handle,
                    add,
                    out,
                    ops.len(),
                )?;
            }
            timing.submit_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            out.prep_relative(WAIT_NS)?;
            timing.wait_ns += phase.elapsed().as_nanos();
            out.fini()?;
        }
        let final_out = if (group.len() - 1) % 2 == 0 {
            self.scratch.output0.as_ref()
        } else {
            self.scratch.output1.as_ref()
        }
        .ok_or(MatmulError::Internal("prepacked KACC final output missing"))?;
        final_out.prep_relative(0)?;
        let phase = Instant::now();
        gather_tile(final_out.as_slice(), first, n_total, dst);
        timing.gather_ns += phase.elapsed().as_nanos();
        final_out.fini()?;
        Ok(())
    }

    fn execute_prepacked_host_kacc(
        &mut self,
        a: &[f16],
        weights: &Fp16PrepackedWeights<'a>,
        k_total: usize,
        n_total: usize,
        group: &[Fp16MatmulTile],
        dst: &mut [f16],
        timing: &mut ExecutionTiming,
    ) -> Result<(), MatmulError> {
        let first = group[0];
        let mut acc = vec![0.0f32; first.m * first.n];
        for &tile in group {
            let phase = Instant::now();
            self.ensure_scratch(tile, 2, false)?;
            timing.scratch_ns += phase.elapsed().as_nanos();
            let (weight_dma, weight_handle) = weights.tile(tile)?;
            let ExecutorScratch {
                regcmd: Some(regcmd),
                input: Some(input),
                output0: Some(output),
                ..
            } = &mut self.scratch
            else {
                return Err(MatmulError::Internal(
                    "prepacked host-KACC scratch invariant",
                ));
            };
            let phase = Instant::now();
            pack_input(input, a, k_total, tile)?;
            prepare_output(output)?;
            timing.pack_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
                tile.m,
                tile.k,
                tile.n,
                input.dma_address(),
                weight_dma,
                output.dma_address(),
            ))?;
            timing.encode_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            write_regcmd(regcmd, &ops)?;
            timing.regcmd_write_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            submit_plain_weight(self.device, regcmd, input, weight_handle, output, ops.len())?;
            timing.submit_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            output.prep_relative(WAIT_NS)?;
            timing.wait_ns += phase.elapsed().as_nanos();
            let phase = Instant::now();
            for tm in 0..tile.m {
                for tn in 0..tile.n {
                    let native = feature_data(tile.n, tile.m, 1, 8, tn + 1, tm + 1, 1);
                    acc[tm * tile.n + tn] += get_f16(output.as_slice(), native).to_f32();
                }
            }
            timing.gather_ns += phase.elapsed().as_nanos();
            output.fini()?;
        }
        let phase = Instant::now();
        for tm in 0..first.m {
            for tn in 0..first.n {
                dst[(first.m0 + tm) * n_total + first.n0 + tn] =
                    f16::from_f32(acc[tm * first.n + tn]);
            }
        }
        timing.gather_ns += phase.elapsed().as_nanos();
        Ok(())
    }
}

// Read the DMA output in its native [N/4][M][4] order. Row-major reads
// jump between feature planes for every scalar and repeatedly pay address
// calculation/cache misses. K partials still accumulate in exactly the same order.
fn accumulate_fp32_cube(acc: &mut [f64], src: &[u8], m: usize, n: usize) {
    for (nb, plane) in src[..m * n * 4].chunks_exact(m * 16).enumerate() {
        for (tm, lanes) in plane.chunks_exact(16).enumerate() {
            let dst = &mut acc[tm * n + nb * 4..tm * n + nb * 4 + 4];
            for lane in 0..4 {
                dst[lane] += get_f32(lanes, lane) as f64;
            }
        }
    }
}

fn submit_plain_weight(
    device: &RocketDevice,
    regcmd: &RocketBuffer<'_>,
    input: &RocketBuffer<'_>,
    weight_handle: u32,
    output: &RocketBuffer<'_>,
    count: usize,
) -> Result<(), MatmulError> {
    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())
            .map_err(|_| MatmulError::AddressAbove32Bit(regcmd.dma_address()))?,
        regcmd_count: count as u32,
    };
    device.submit(
        &[task],
        &[input.handle(), weight_handle, regcmd.handle()],
        &[output.handle()],
    )?;
    Ok(())
}

fn submit_accumulate_weight(
    device: &RocketDevice,
    regcmd: &RocketBuffer<'_>,
    input: &RocketBuffer<'_>,
    weight_handle: u32,
    add: &RocketBuffer<'_>,
    output: &RocketBuffer<'_>,
    count: usize,
) -> Result<(), MatmulError> {
    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())
            .map_err(|_| MatmulError::AddressAbove32Bit(regcmd.dma_address()))?,
        regcmd_count: count as u32,
    };
    device.submit(
        &[task],
        &[input.handle(), weight_handle, regcmd.handle(), add.handle()],
        &[output.handle()],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn fp32_cube_gather_matches_feature_addressing_and_accumulates() {
        for (m, n) in [(4, 16), (12, 96), (256, 688)] {
            let mut raw = vec![0u8; m * n * 4];
            for row in 0..m {
                for col in 0..n {
                    let value = (row * n + col) as f32 / 16.0 - 3.0;
                    let native = feature_data(n, m, 1, 4, col + 1, row + 1, 1);
                    raw[native * 4..native * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            let mut acc = vec![1.0f64; m * n];
            accumulate_fp32_cube(&mut acc, &raw, m, n);
            accumulate_fp32_cube(&mut acc, &raw, m, n);
            for (index, actual) in acc.into_iter().enumerate() {
                assert_eq!(actual, 1.0 + 2.0 * (index as f64 / 16.0 - 3.0));
            }
        }
    }

    use super::*;

    #[test]
    fn m_tiling_deduplicates_resident_weight_tiles() {
        let p = plan_fp16_matmul(512, 512, 128).unwrap();
        let (tiles, _) = unique_weight_layout(&p).unwrap();
        assert_eq!(p.tiles.len(), 2);
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn mnk_tiling_deduplicates_only_m_dimension() {
        let p = plan_fp16_matmul(300, 512, 272).unwrap();
        let (tiles, _) = unique_weight_layout(&p).unwrap();
        assert_eq!(p.tiles.len(), 8);
        assert_eq!(tiles.len(), p.n_tiles() * p.k_tiles());
        assert_eq!(tiles.len(), 4);
    }
}
