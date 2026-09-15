#![forbid(unsafe_op_in_unsafe_fn)]

use bytemuck::pod_read_unaligned;
use half::f16;
use llama_gguf::tensor::quant::{BlockQ4K, BlockQ6K, dequantize_q4_k, dequantize_q6_k};
use rocket_runtime::RocketDevice;
use rocknpu_matmul::{
    Fp16MatmulExecutor, Int4DecodeExecutor, Int4DecodePool, Int4DecodePoolPreparedWeights,
    Int4GroupedPreparedWeights, Int4PreparedWeights, Int8DecodePool, Int8DecodePoolPreparedWeights,
    Int8DecodeSplit,
};
use std::collections::{HashMap, hash_map::Entry};
use std::env;
use std::mem::size_of;
use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::Instant;

const STATUS_OK: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = -1;
const STATUS_EXECUTION_ERROR: i32 = -3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DecodeWeightKind {
    Q4K,
    Q6K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DecodeWeightKey {
    address: usize,
    bytes: usize,
    k: usize,
    n: usize,
    kind: DecodeWeightKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeChoice {
    split: Int8DecodeSplit,
    workers: usize,
}

struct CachedW8A8Weight {
    prepared: Int8DecodePoolPreparedWeights,
    scales: Vec<f32>,
    choice: DecodeChoice,
}

struct CachedW4A4Weight {
    prepared: Int4PreparedWeights,
    scales: Vec<f32>,
}

struct CachedGroupedW4A4Weight {
    prepared: Int4GroupedPreparedWeights,
    scales: Vec<f32>,
}

struct CachedPoolW4A4Weight {
    prepared: Int4DecodePoolPreparedWeights,
    scales: Vec<f32>,
    workers: usize,
}

pub struct RockNpuContext {
    device: RocketDevice,
    decode_pool: Int8DecodePool,
    decode_worker_cache: HashMap<(usize, usize), DecodeChoice>,
    decode_weights: HashMap<DecodeWeightKey, CachedW8A8Weight>,
    decode_w4a4_weights: HashMap<DecodeWeightKey, CachedW4A4Weight>,
    decode_grouped_w4a4_weights: HashMap<(DecodeWeightKey, usize, bool), CachedGroupedW4A4Weight>,
    decode_pool_w4a4_weights: HashMap<(DecodeWeightKey, bool), CachedPoolW4A4Weight>,
    w4a4_pool: Option<Int4DecodePool>,
    w4a4_worker_cache: HashMap<(usize, usize), usize>,
    w4a4_enabled: bool,
    w4a4_calls: usize,
    w4a4_saturated_calls: usize,
    w4a4_saturated_outputs: usize,
    decode_cache_hits: usize,
    decode_cache_misses: usize,
    decode_cache_hit_ns: u64,
    decode_cache_miss_ns: u64,
    decode_worker_calls: [usize; 3],
    decode_ksplit_calls: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RockNpuDecodeCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
    pub resident_bytes: usize,
    pub hit_ns: u64,
    pub miss_ns: u64,
    pub tuned_shapes: usize,
    pub worker1_calls: usize,
    pub worker2_calls: usize,
    pub worker3_calls: usize,
    pub ksplit_calls: usize,
}

fn env_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
}

fn w4a4_shape_enabled(k: usize, n: usize) -> bool {
    match env::var("ROCKNPU_W4A4_SCOPE").as_deref() {
        Ok("ffn") => k == 2048 && n == 5632,
        Ok("attn") => k == 2048 && (n == 2048 || n == 256),
        Ok("proj2048") => k == 2048 && n == 2048,
        Ok("kv") => k == 2048 && n == 256,
        Ok("all") => true,
        _ => false,
    }
}

fn w4a4_group_size(k: usize) -> Option<usize> {
    let group = env::var("ROCKNPU_W4A4_GROUP").ok()?.parse::<usize>().ok()?;
    (group != 0 && group.is_multiple_of(32) && k.is_multiple_of(group)).then_some(group)
}

fn quantize_symmetric(values: &[f32]) -> Option<(Vec<i8>, f32)> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        return Some((vec![0; values.len()], 1.0));
    }
    let scale = max_abs / 127.0;
    let quantized = values
        .iter()
        .map(|&value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Some((quantized, scale))
}

fn quantize_symmetric_i4(values: &[f32]) -> Option<(Vec<i8>, f32)> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        return Some((vec![0; values.len()], 1.0));
    }
    let scale = max_abs / 7.0;
    let quantized = values
        .iter()
        .map(|&value| (value / scale).round().clamp(-7.0, 7.0) as i8)
        .collect();
    Some((quantized, scale))
}

fn append_quantized_symmetric(values: &[f32], dst: &mut Vec<i8>) -> Option<f32> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    dst.extend(
        values
            .iter()
            .map(|&value| (value / scale).round().clamp(-127.0, 127.0) as i8),
    );
    Some(scale)
}

fn prepare_q4_k_w8a8(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric(&row, &mut weights)?);
    }
    (weights.len() == n.checked_mul(k)? && scales.len() == n).then_some((weights, scales))
}

fn fwht_norm_in_place(values: &mut [f32]) -> bool {
    if values.is_empty() || !values.len().is_power_of_two() {
        return false;
    }
    let mut width = 1usize;
    while width < values.len() {
        let span = width * 2;
        for base in (0..values.len()).step_by(span) {
            for offset in 0..width {
                let a = values[base + offset];
                let b = values[base + offset + width];
                values[base + offset] = a + b;
                values[base + offset + width] = a - b;
            }
        }
        width = span;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
    true
}

fn quantize_symmetric_i4_grouped(
    values: &[f32],
    group_size: usize,
    hadamard: bool,
) -> Option<(Vec<i8>, Vec<f32>)> {
    if group_size == 0 || !values.len().is_multiple_of(group_size) {
        return None;
    }
    let mut transformed = Vec::new();
    let source = if hadamard {
        transformed.extend_from_slice(values);
        if !fwht_norm_in_place(&mut transformed) {
            return None;
        }
        transformed.as_slice()
    } else {
        values
    };
    let mut quantized = Vec::with_capacity(source.len());
    let mut scales = Vec::with_capacity(source.len() / group_size);
    for group in source.chunks_exact(group_size) {
        scales.push(append_quantized_symmetric_i4(group, &mut quantized)?);
    }
    Some((quantized, scales))
}

fn append_quantized_symmetric_i4(values: &[f32], dst: &mut Vec<i8>) -> Option<f32> {
    let mut max_abs = 0.0f32;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        max_abs = max_abs.max(value.abs());
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
    dst.extend(
        values
            .iter()
            .map(|&value| (value / scale).round().clamp(-7.0, 7.0) as i8),
    );
    Some(scale)
}

fn prepare_q4_k_w4a4(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric_i4(&row, &mut weights)?);
    }
    (weights.len() == n.checked_mul(k)? && scales.len() == n).then_some((weights, scales))
}

fn prepare_q4_k_grouped_w4a4(
    weight_bytes: &[u8],
    k: usize,
    n: usize,
    group_size: usize,
    hadamard: bool,
) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    if group_size == 0 || !group_size.is_multiple_of(32) || !k.is_multiple_of(group_size) {
        return None;
    }
    let groups = k / group_size;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ4K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = vec![0.0f32; groups.checked_mul(n)?];
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for (output_channel, encoded_row) in weight_bytes.chunks_exact(row_bytes).enumerate() {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ4K>()) {
            let block: BlockQ4K = pod_read_unaligned(bytes);
            dequantize_q4_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        if hadamard && !fwht_norm_in_place(&mut row) {
            return None;
        }
        for (group_index, group) in row.chunks_exact(group_size).enumerate() {
            scales[group_index * n + output_channel] =
                append_quantized_symmetric_i4(group, &mut weights)?;
        }
    }
    (weights.len() == n.checked_mul(k)?).then_some((weights, scales))
}

fn prepare_q6_k_w8a8(weight_bytes: &[u8], k: usize, n: usize) -> Option<(Vec<i8>, Vec<f32>)> {
    const VALUES: usize = 256;
    let row_bytes = (k / VALUES).checked_mul(size_of::<BlockQ6K>())?;
    if weight_bytes.len() != n.checked_mul(row_bytes)? {
        return None;
    }
    let mut weights = Vec::with_capacity(n.checked_mul(k)?);
    let mut scales = Vec::with_capacity(n);
    let mut row = Vec::with_capacity(k);
    let mut decoded = [0.0f32; VALUES];
    for encoded_row in weight_bytes.chunks_exact(row_bytes) {
        row.clear();
        for bytes in encoded_row.chunks_exact(size_of::<BlockQ6K>()) {
            let block: BlockQ6K = pod_read_unaligned(bytes);
            dequantize_q6_k(&block, &mut decoded);
            row.extend_from_slice(&decoded);
        }
        scales.push(append_quantized_symmetric(&row, &mut weights)?);
    }
    (weights.len() == n.checked_mul(k)? && scales.len() == n).then_some((weights, scales))
}

fn decode_worker_candidates(
    pool: &Int8DecodePool,
    k: usize,
    n: usize,
) -> Result<Vec<DecodeChoice>, ()> {
    let mut candidates = vec![DecodeChoice {
        split: Int8DecodeSplit::N,
        workers: 1,
    }];
    let mut last_effective_n = 1usize;
    for requested in 2..=pool.workers() {
        let effective = pool.effective_workers_for_n(n, requested).map_err(|_| ())?;
        if effective > last_effective_n {
            candidates.push(DecodeChoice {
                split: Int8DecodeSplit::N,
                workers: requested,
            });
            last_effective_n = effective;
        }
    }
    if k > 4096 {
        let mut last_effective_k = 1usize;
        for requested in 2..=pool.workers() {
            let effective = pool.effective_workers_for_k(k, requested).map_err(|_| ())?;
            if effective > last_effective_k {
                candidates.push(DecodeChoice {
                    split: Int8DecodeSplit::K,
                    workers: requested,
                });
                last_effective_k = effective;
            }
        }
    }
    Ok(candidates)
}

fn select_decode_worker_index(medians: &[u128]) -> Option<usize> {
    const REQUIRED_IMPROVEMENT_PERCENT: u128 = 5;
    let (&first, rest) = medians.split_first()?;
    let mut selected_index = 0usize;
    let mut selected_median = first;
    for (offset, &candidate_median) in rest.iter().enumerate() {
        let candidate_index = offset + 1;
        let candidate_scaled = candidate_median.saturating_mul(100);
        let required_scaled = selected_median.saturating_mul(100 - REQUIRED_IMPROVEMENT_PERCENT);
        if candidate_scaled <= required_scaled {
            selected_index = candidate_index;
            selected_median = candidate_median;
        }
    }
    Some(selected_index)
}

fn tune_decode_workers(
    pool: &mut Int8DecodePool,
    weights: Arc<[i8]>,
    activation: Arc<[i8]>,
    k: usize,
    n: usize,
) -> Result<(DecodeChoice, Int8DecodePoolPreparedWeights), ()> {
    let candidates = decode_worker_candidates(pool, k, n)?;
    let mut prepared = Vec::with_capacity(candidates.len());
    for &choice in &candidates {
        match pool.prepare_weights_with_split(
            Arc::clone(&weights),
            k,
            n,
            choice.workers,
            choice.split,
        ) {
            Ok(handle) => prepared.push((choice, handle)),
            Err(_) => {
                for (_, handle) in prepared {
                    let _ = pool.release_prepared(&handle);
                }
                return Err(());
            }
        }
    }

    let tuning = (|| -> Result<usize, ()> {
        for (_, handle) in &prepared {
            let _ = pool
                .execute_prepared(Arc::clone(&activation), handle)
                .map_err(|_| ())?;
        }
        let mut samples: Vec<Vec<u128>> = prepared.iter().map(|_| Vec::with_capacity(3)).collect();
        for round in 0..3 {
            if round % 2 == 0 {
                for (index, (_, handle)) in prepared.iter().enumerate() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            } else {
                for (index, (_, handle)) in prepared.iter().enumerate().rev() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            }
        }
        let mut medians = Vec::with_capacity(samples.len());
        for values in &mut samples {
            values.sort_unstable();
            medians.push(values[values.len() / 2]);
        }
        select_decode_worker_index(&medians).ok_or(())
    })();

    let best_index = match tuning {
        Ok(index) => index,
        Err(()) => {
            for (_, handle) in prepared {
                let _ = pool.release_prepared(&handle);
            }
            return Err(());
        }
    };

    let mut winner = None;
    for (index, (choice, handle)) in prepared.into_iter().enumerate() {
        if index == best_index {
            winner = Some((choice, handle));
        } else if pool.release_prepared(&handle).is_err() {
            return Err(());
        }
    }
    winner.ok_or(())
}

fn tune_w4a4_workers(
    pool: &mut Int4DecodePool,
    weights: Arc<[i8]>,
    activation: Arc<[i8]>,
    k: usize,
    n: usize,
) -> Result<(usize, Int4DecodePoolPreparedWeights), ()> {
    let mut candidates = vec![1usize];
    let mut last_effective = 1usize;
    for requested in 2..=pool.workers() {
        let effective = pool.effective_workers_for_n(n, requested).map_err(|_| ())?;
        if effective > last_effective {
            candidates.push(requested);
            last_effective = effective;
        }
    }

    let mut prepared = Vec::with_capacity(candidates.len());
    for &workers in &candidates {
        match pool.prepare_weights(Arc::clone(&weights), k, n, workers) {
            Ok(handle) => prepared.push((workers, handle)),
            Err(_) => {
                for (_, handle) in prepared {
                    let _ = pool.release_prepared(&handle);
                }
                return Err(());
            }
        }
    }

    let tuning = (|| -> Result<usize, ()> {
        for (_, handle) in &prepared {
            let _ = pool
                .execute_prepared(Arc::clone(&activation), handle)
                .map_err(|_| ())?;
        }
        let mut samples: Vec<Vec<u128>> = prepared.iter().map(|_| Vec::with_capacity(3)).collect();
        for round in 0usize..3 {
            if round.is_multiple_of(2) {
                for (index, (_, handle)) in prepared.iter().enumerate() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            } else {
                for (index, (_, handle)) in prepared.iter().enumerate().rev() {
                    let start = Instant::now();
                    let _ = pool
                        .execute_prepared(Arc::clone(&activation), handle)
                        .map_err(|_| ())?;
                    samples[index].push(start.elapsed().as_nanos());
                }
            }
        }
        let mut medians = Vec::with_capacity(samples.len());
        for values in &mut samples {
            values.sort_unstable();
            medians.push(values[values.len() / 2]);
        }
        let selected = select_decode_worker_index(&medians).ok_or(())?;
        if env_enabled("ROCKNPU_W4A4_TRACE") {
            let medians_us = medians
                .iter()
                .map(|&ns| ns as f64 / 1.0e3)
                .collect::<Vec<_>>();
            eprintln!(
                "ROCKNPU W4A4 tune K={} N={} candidates={:?} medians_us={:?} selected_workers={}",
                k, n, candidates, medians_us, candidates[selected]
            );
        }
        Ok(selected)
    })();

    let best_index = match tuning {
        Ok(index) => index,
        Err(()) => {
            for (_, handle) in prepared {
                let _ = pool.release_prepared(&handle);
            }
            return Err(());
        }
    };
    let mut winner = None;
    for (index, (workers, handle)) in prepared.into_iter().enumerate() {
        if index == best_index {
            winner = Some((workers, handle));
        } else if pool.release_prepared(&handle).is_err() {
            return Err(());
        }
    }
    winner.ok_or(())
}

fn execute_cached_pool_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    hadamard: bool,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    let Some(pool) = context.w4a4_pool.as_mut() else {
        return STATUS_EXECUTION_ERROR;
    };
    let Some((activations_i4, activation_scales)) =
        quantize_symmetric_i4_grouped(activations_k_f32, key.k, hadamard)
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(&activation_scale) = activation_scales.first() else {
        return STATUS_INVALID_ARGUMENT;
    };
    if activation_scales.len() != 1 || output_n_f32.len() != key.n {
        return STATUS_INVALID_ARGUMENT;
    }
    let activation: Arc<[i8]> = Arc::from(activations_i4);
    let cache_key = (key, hadamard);

    let (cached, cache_hit) = match context.decode_pool_w4a4_weights.entry(cache_key) {
        Entry::Occupied(entry) => {
            context.decode_cache_hits = context.decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if scales.len() != key.n {
                return STATUS_INVALID_ARGUMENT;
            }
            let weights: Arc<[i8]> = Arc::from(weights_i4);
            let shape = (key.k, key.n);
            let (workers, prepared) = if let Some(&workers) = context.w4a4_worker_cache.get(&shape)
            {
                let prepared =
                    match pool.prepare_weights(Arc::clone(&weights), key.k, key.n, workers) {
                        Ok(prepared) => prepared,
                        Err(_) => return STATUS_EXECUTION_ERROR,
                    };
                (workers, prepared)
            } else {
                let (workers, prepared) = match tune_w4a4_workers(
                    pool,
                    Arc::clone(&weights),
                    Arc::clone(&activation),
                    key.k,
                    key.n,
                ) {
                    Ok(result) => result,
                    Err(()) => return STATUS_EXECUTION_ERROR,
                };
                context.w4a4_worker_cache.insert(shape, workers);
                (workers, prepared)
            };
            context.decode_cache_misses = context.decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedPoolW4A4Weight {
                    prepared,
                    scales,
                    workers,
                }),
                false,
            )
        }
    };

    let result = match pool.execute_prepared(Arc::clone(&activation), &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    context.w4a4_calls = context.w4a4_calls.saturating_add(1);
    if let Some(calls) = context
        .decode_worker_calls
        .get_mut(cached.workers.saturating_sub(1))
    {
        *calls = calls.saturating_add(1);
    }
    if result.stats.saturated_outputs != 0 {
        context.w4a4_saturated_calls = context.w4a4_saturated_calls.saturating_add(1);
        context.w4a4_saturated_outputs = context
            .w4a4_saturated_outputs
            .saturating_add(result.stats.saturated_outputs);
    }
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = f32::from(acc) * activation_scale * weight_scale;
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        context.decode_cache_hit_ns = context.decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        context.decode_cache_miss_ns = context.decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_grouped_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    group_size: usize,
    hadamard: bool,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || group_size == 0
        || !group_size.is_multiple_of(32)
        || !key.k.is_multiple_of(group_size)
        || !key.n.is_multiple_of(64)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some((activations_i4, activation_scales)) =
        quantize_symmetric_i4_grouped(activations_k_f32, group_size, hadamard)
    else {
        return STATUS_INVALID_ARGUMENT;
    };
    let groups = key.k / group_size;
    if activation_scales.len() != groups {
        return STATUS_INVALID_ARGUMENT;
    }

    let RockNpuContext {
        device,
        decode_grouped_w4a4_weights,
        w4a4_calls,
        w4a4_saturated_calls,
        w4a4_saturated_outputs,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        ..
    } = context;
    let executor = Int4DecodeExecutor::new(device);
    let cache_key = (key, group_size, hadamard);
    let (cached, cache_hit) = match decode_grouped_w4a4_weights.entry(cache_key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            if scales.len() != groups.saturating_mul(key.n) {
                return STATUS_INVALID_ARGUMENT;
            }
            let prepared =
                match executor.prepare_grouped_weights(&weights_i4, key.k, key.n, group_size) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedGroupedW4A4Weight { prepared, scales }),
                false,
            )
        }
    };

    let result = match executor.execute_grouped_prepared(&activations_i4, &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if result.values.len() != groups.saturating_mul(key.n) {
        return STATUS_EXECUTION_ERROR;
    }
    *w4a4_calls = w4a4_calls.saturating_add(1);
    if result.stats.saturated_outputs != 0 {
        *w4a4_saturated_calls = w4a4_saturated_calls.saturating_add(1);
        *w4a4_saturated_outputs =
            w4a4_saturated_outputs.saturating_add(result.stats.saturated_outputs);
    }
    output_n_f32.fill(0.0);
    for group in 0..groups {
        let activation_scale = activation_scales[group];
        let partials = &result.values[group * key.n..(group + 1) * key.n];
        let weight_scales = &cached.scales[group * key.n..(group + 1) * key.n];
        for ((dst, &acc), &weight_scale) in output_n_f32.iter_mut().zip(partials).zip(weight_scales)
        {
            *dst += f32::from(acc) * activation_scale * weight_scale;
        }
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w4a4_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || !key.k.is_multiple_of(32)
        || key.k > 10_752
        || !key.n.is_multiple_of(64)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some((activations_i4, activation_scale)) = quantize_symmetric_i4(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };

    let RockNpuContext {
        device,
        decode_w4a4_weights,
        w4a4_calls,
        w4a4_saturated_calls,
        w4a4_saturated_outputs,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        ..
    } = context;
    let executor = Int4DecodeExecutor::new(device);

    let (cached, cache_hit) = match decode_w4a4_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i4, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            let prepared = match executor.prepare_weights(&weights_i4, key.k, key.n) {
                Ok(prepared) => prepared,
                Err(_) => return STATUS_EXECUTION_ERROR,
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (entry.insert(CachedW4A4Weight { prepared, scales }), false)
        }
    };

    let result = match executor.execute_prepared(&activations_i4, &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    *w4a4_calls = w4a4_calls.saturating_add(1);
    if result.stats.saturated_outputs != 0 {
        *w4a4_saturated_calls = w4a4_saturated_calls.saturating_add(1);
        *w4a4_saturated_outputs =
            w4a4_saturated_outputs.saturating_add(result.stats.saturated_outputs);
        if env_enabled("ROCKNPU_W4A4_TRACE") {
            eprintln!(
                "ROCKNPU W4A4 saturation K={} N={} saturated_outputs={}",
                key.k, key.n, result.stats.saturated_outputs
            );
        }
    }
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = f32::from(acc) * activation_scale * weight_scale;
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

fn execute_cached_w8a8_m1<F>(
    context: &mut RockNpuContext,
    key: DecodeWeightKey,
    activations_k_f32: &[f32],
    output_n_f32: &mut [f32],
    prepare: F,
) -> i32
where
    F: FnOnce() -> Option<(Vec<i8>, Vec<f32>)>,
{
    let call_start = Instant::now();
    if activations_k_f32.len() != key.k
        || output_n_f32.len() != key.n
        || !key.k.is_multiple_of(512)
        || !key.n.is_multiple_of(32)
        || key.n > 8192
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some((activations_i8, activation_scale)) = quantize_symmetric(activations_k_f32) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let activation: Arc<[i8]> = Arc::from(activations_i8);

    let RockNpuContext {
        device: _,
        decode_pool,
        decode_worker_cache,
        decode_weights,
        decode_cache_hits,
        decode_cache_misses,
        decode_cache_hit_ns,
        decode_cache_miss_ns,
        decode_worker_calls,
        decode_ksplit_calls,
        ..
    } = context;

    let (cached, cache_hit) = match decode_weights.entry(key) {
        Entry::Occupied(entry) => {
            *decode_cache_hits = decode_cache_hits.saturating_add(1);
            (entry.into_mut(), true)
        }
        Entry::Vacant(entry) => {
            let Some((weights_i8, scales)) = prepare() else {
                return STATUS_INVALID_ARGUMENT;
            };
            let weights: Arc<[i8]> = Arc::from(weights_i8);
            let shape = (key.k, key.n);
            let (choice, prepared) = if let Some(&choice) = decode_worker_cache.get(&shape) {
                let prepared = match decode_pool.prepare_weights_with_split(
                    Arc::clone(&weights),
                    key.k,
                    key.n,
                    choice.workers,
                    choice.split,
                ) {
                    Ok(prepared) => prepared,
                    Err(_) => return STATUS_EXECUTION_ERROR,
                };
                (choice, prepared)
            } else {
                let (choice, prepared) = match tune_decode_workers(
                    decode_pool,
                    Arc::clone(&weights),
                    Arc::clone(&activation),
                    key.k,
                    key.n,
                ) {
                    Ok(result) => result,
                    Err(()) => return STATUS_EXECUTION_ERROR,
                };
                decode_worker_cache.insert(shape, choice);
                (choice, prepared)
            };
            *decode_cache_misses = decode_cache_misses.saturating_add(1);
            (
                entry.insert(CachedW8A8Weight {
                    prepared,
                    scales,
                    choice,
                }),
                false,
            )
        }
    };

    let result = match decode_pool.execute_prepared(Arc::clone(&activation), &cached.prepared) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    if let Some(calls) = decode_worker_calls.get_mut(cached.choice.workers.saturating_sub(1)) {
        *calls = calls.saturating_add(1);
    }
    if cached.choice.split == Int8DecodeSplit::K {
        *decode_ksplit_calls = decode_ksplit_calls.saturating_add(1);
    }
    for ((dst, &acc), &weight_scale) in output_n_f32
        .iter_mut()
        .zip(&result.values)
        .zip(&cached.scales)
    {
        *dst = acc as f32 * activation_scale * weight_scale;
    }
    let elapsed_ns = u64::try_from(call_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if cache_hit {
        *decode_cache_hit_ns = decode_cache_hit_ns.saturating_add(elapsed_ns);
    } else {
        *decode_cache_miss_ns = decode_cache_miss_ns.saturating_add(elapsed_ns);
    }
    STATUS_OK
}

/// Return the number of RockNPU devices currently usable by the runtime.
///
/// The initial RK3588 backend exposes one default Rocket device at
/// `/dev/accel/accel0`. The C ABI intentionally reports availability rather
/// than exposing Rocket file descriptors or Linux-specific details.
#[unsafe(no_mangle)]
pub extern "C" fn rocknpu_device_count() -> usize {
    usize::from(RocketDevice::open().is_ok())
}

#[unsafe(no_mangle)]
pub extern "C" fn rocknpu_context_create() -> *mut RockNpuContext {
    let w4a4_enabled = env_enabled("ROCKNPU_W4A4");
    let w4a4_pool = if w4a4_enabled {
        Int4DecodePool::new(3).ok()
    } else {
        None
    };
    match (RocketDevice::open(), Int8DecodePool::new(3)) {
        (Ok(device), Ok(decode_pool)) => Box::into_raw(Box::new(RockNpuContext {
            device,
            decode_pool,
            decode_worker_cache: HashMap::new(),
            decode_weights: HashMap::new(),
            decode_w4a4_weights: HashMap::new(),
            decode_grouped_w4a4_weights: HashMap::new(),
            decode_pool_w4a4_weights: HashMap::new(),
            w4a4_pool,
            w4a4_worker_cache: HashMap::new(),
            w4a4_enabled,
            w4a4_calls: 0,
            w4a4_saturated_calls: 0,
            w4a4_saturated_outputs: 0,
            decode_cache_hits: 0,
            decode_cache_misses: 0,
            decode_cache_hit_ns: 0,
            decode_cache_miss_ns: 0,
            decode_worker_calls: [0; 3],
            decode_ksplit_calls: 0,
        })),
        _ => ptr::null_mut(),
    }
}

/// Read lazy W8A8 decode-cache statistics for diagnostics.
///
/// # Safety
/// `context` must be a live RockNPU context and `out` must point to writable
/// storage for one [`RockNpuDecodeCacheStats`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_context_decode_cache_stats(
    context: *const RockNpuContext,
    out: *mut RockNpuDecodeCacheStats,
) -> i32 {
    if context.is_null() || out.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: both pointers were checked above and are required by the C ABI to
    // remain valid for this synchronous call.
    let context = unsafe { &*context };
    let w8_resident_bytes = context.decode_weights.values().fold(0usize, |acc, cached| {
        acc.saturating_add(cached.prepared.stats().resident_bytes)
    });
    let w4_resident_bytes = context
        .decode_w4a4_weights
        .values()
        .fold(w8_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    let grouped_w4_resident_bytes = context
        .decode_grouped_w4a4_weights
        .values()
        .fold(w4_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    let resident_bytes = context
        .decode_pool_w4a4_weights
        .values()
        .fold(grouped_w4_resident_bytes, |acc, cached| {
            acc.saturating_add(cached.prepared.stats().resident_bytes)
        });
    // SAFETY: out is non-null and caller provides writable storage for one stats value.
    unsafe {
        *out = RockNpuDecodeCacheStats {
            hits: context.decode_cache_hits,
            misses: context.decode_cache_misses,
            entries: context
                .decode_weights
                .len()
                .saturating_add(context.decode_w4a4_weights.len())
                .saturating_add(context.decode_grouped_w4a4_weights.len())
                .saturating_add(context.decode_pool_w4a4_weights.len()),
            resident_bytes,
            hit_ns: context.decode_cache_hit_ns,
            miss_ns: context.decode_cache_miss_ns,
            tuned_shapes: context
                .decode_worker_cache
                .len()
                .saturating_add(context.w4a4_worker_cache.len()),
            worker1_calls: context.decode_worker_calls[0],
            worker2_calls: context.decode_worker_calls[1],
            worker3_calls: context.decode_worker_calls[2],
            ksplit_calls: context.decode_ksplit_calls,
        };
    }
    STATUS_OK
}

/// Destroy a context returned by [`rocknpu_context_create`].
///
/// # Safety
///
/// `context` must be null or a live pointer returned by `rocknpu_context_create`
/// that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_context_destroy(context: *mut RockNpuContext) {
    if context.is_null() {
        return;
    }
    // SAFETY: the C API requires a context returned by rocknpu_context_create,
    // and ownership is transferred back exactly once at destroy time.
    unsafe {
        let context = Box::from_raw(context);
        if env_enabled("ROCKNPU_W4A4_TRACE") && context.w4a4_calls != 0 {
            eprintln!(
                "ROCKNPU W4A4 summary calls={} cache_entries={} grouped_cache_entries={} pool_cache_entries={} tuned_shapes={} saturated_calls={} saturated_outputs={}",
                context.w4a4_calls,
                context.decode_w4a4_weights.len(),
                context.decode_grouped_w4a4_weights.len(),
                context.decode_pool_w4a4_weights.len(),
                context.w4a4_worker_cache.len(),
                context.w4a4_saturated_calls,
                context.w4a4_saturated_outputs
            );
        }
        drop(context);
    }
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T on RockNPU.
///
/// `weights_nk_f16_bits` contains IEEE-754 binary16 bit patterns in row-major
/// [N,K] order. `activations_mk_f32` and `output_mn_f32` are row-major.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. The three tensor pointers
/// must reference readable/writable storage for at least `N*K`, `M*K`, and
/// `M*N` elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_f16_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_f16_bits: *const u16,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    if context.is_null()
        || weights_nk_f16_bits.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(b_len) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };

    // SAFETY: non-null pointers and element counts are validated above. The C
    // caller owns these buffers and must provide at least the documented sizes.
    let (weights_bits, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_f16_bits, b_len),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    let weights = weights_bits
        .iter()
        .copied()
        .map(f16::from_bits)
        .collect::<Vec<_>>();
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T with GGML-compatible Q4_K weights.
///
/// Q4_K is decoded in this Rust bridge into the same FP16 weight contract used
/// by the existing RockNPU executor. The external GGML adapter only forwards
/// raw block bytes and never owns quantization math.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. `weights_nk_q4_k` must
/// reference exactly `weights_bytes` readable bytes encoding N contiguous rows
/// of K Q4_K values. Activation/output pointers must provide `M*K` readable and
/// `M*N` writable f32 elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q4_k_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_q4_k: *const u8,
    weights_bytes: usize,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    const Q4_K_VALUES_PER_BLOCK: usize = 256;

    if context.is_null()
        || weights_nk_q4_k.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
        || !k.is_multiple_of(Q4_K_VALUES_PER_BLOCK)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(weight_values) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let blocks_per_row = k / Q4_K_VALUES_PER_BLOCK;
    let Some(block_count) = n.checked_mul(blocks_per_row) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_weight_bytes) = block_count.checked_mul(size_of::<BlockQ4K>()) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_bytes != expected_weight_bytes {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: pointer/null/size contracts are validated above. The caller owns
    // the buffers for the duration of this synchronous call.
    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q4_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q4K,
        };
        if context.w4a4_enabled
            && w4a4_shape_enabled(k, n)
            && k <= 10_752
            && n.is_multiple_of(64)
            && n <= 8192
        {
            if let Some(group_size) = w4a4_group_size(k) {
                let hadamard = env_enabled("ROCKNPU_W4A4_HADAMARD");
                if group_size == k && context.w4a4_pool.is_some() {
                    return execute_cached_pool_w4a4_m1(
                        context,
                        key,
                        hadamard,
                        activations,
                        output,
                        || prepare_q4_k_grouped_w4a4(weight_bytes, k, n, group_size, hadamard),
                    );
                }
                return execute_cached_grouped_w4a4_m1(
                    context,
                    key,
                    group_size,
                    hadamard,
                    activations,
                    output,
                    || prepare_q4_k_grouped_w4a4(weight_bytes, k, n, group_size, hadamard),
                );
            }
            return execute_cached_w4a4_m1(context, key, activations, output, || {
                prepare_q4_k_w4a4(weight_bytes, k, n)
            });
        }
        return execute_cached_w8a8_m1(context, key, activations, output, || {
            prepare_q4_k_w8a8(weight_bytes, k, n)
        });
    }

    let mut weights = Vec::with_capacity(weight_values);
    let mut decoded = [0.0f32; Q4_K_VALUES_PER_BLOCK];
    for bytes in weight_bytes.chunks_exact(size_of::<BlockQ4K>()) {
        let block: BlockQ4K = pod_read_unaligned(bytes);
        dequantize_q4_k(&block, &mut decoded);
        weights.extend(decoded.iter().copied().map(f16::from_f32));
    }
    if weights.len() != weight_values {
        return STATUS_INVALID_ARGUMENT;
    }
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

/// Execute C[M,N] = A[M,K] x B[N,K]^T with GGML-compatible Q6_K weights.
///
/// Like the Q4_K bridge, Q6_K is decoded in Rust into the existing FP16
/// executor contract. The external GGML adapter only forwards raw block bytes.
///
/// # Safety
///
/// `context` must point to a live RockNPU context. `weights_nk_q6_k` must
/// reference exactly `weights_bytes` readable bytes encoding N contiguous rows
/// of K Q6_K values. Activation/output pointers must provide `M*K` readable and
/// `M*N` writable f32 elements respectively for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocknpu_matmul_q6_k_f32_f32(
    context: *mut RockNpuContext,
    weights_nk_q6_k: *const u8,
    weights_bytes: usize,
    activations_mk_f32: *const f32,
    output_mn_f32: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    const Q6_K_VALUES_PER_BLOCK: usize = 256;

    if context.is_null()
        || weights_nk_q6_k.is_null()
        || activations_mk_f32.is_null()
        || output_mn_f32.is_null()
        || m == 0
        || k == 0
        || n == 0
        || !k.is_multiple_of(Q6_K_VALUES_PER_BLOCK)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let Some(a_len) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(weight_values) = n.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let blocks_per_row = k / Q6_K_VALUES_PER_BLOCK;
    let Some(block_count) = n.checked_mul(blocks_per_row) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(expected_weight_bytes) = block_count.checked_mul(size_of::<BlockQ6K>()) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weights_bytes != expected_weight_bytes {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: pointer/null/size contracts are validated above. The caller owns
    // the buffers for the duration of this synchronous call.
    let (weight_bytes, activations, output, context) = unsafe {
        (
            slice::from_raw_parts(weights_nk_q6_k, weights_bytes),
            slice::from_raw_parts(activations_mk_f32, a_len),
            slice::from_raw_parts_mut(output_mn_f32, out_len),
            &mut *context,
        )
    };

    if m == 1 {
        let key = DecodeWeightKey {
            address: weight_bytes.as_ptr() as usize,
            bytes: weight_bytes.len(),
            k,
            n,
            kind: DecodeWeightKind::Q6K,
        };
        return execute_cached_w8a8_m1(context, key, activations, output, || {
            prepare_q6_k_w8a8(weight_bytes, k, n)
        });
    }

    let mut weights = Vec::with_capacity(weight_values);
    let mut decoded = [0.0f32; Q6_K_VALUES_PER_BLOCK];
    for bytes in weight_bytes.chunks_exact(size_of::<BlockQ6K>()) {
        let block: BlockQ6K = pod_read_unaligned(bytes);
        dequantize_q6_k(&block, &mut decoded);
        weights.extend(decoded.iter().copied().map(f16::from_f32));
    }
    if weights.len() != weight_values {
        return STATUS_INVALID_ARGUMENT;
    }
    let activations = activations
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();

    let mut executor = match Fp16MatmulExecutor::new(&context.device) {
        Ok(executor) => executor,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    let result = match executor.execute_f32(&activations, &weights, m, k, n) {
        Ok(result) => result,
        Err(_) => return STATUS_EXECUTION_ERROR,
    };
    output.copy_from_slice(&result.values);
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn q6_k_block_size_matches_ggml_abi() {
        assert_eq!(size_of::<BlockQ6K>(), 210);
    }

    #[test]
    fn normalized_fwht_preserves_dot_product() {
        let mut a = vec![1.0f32, -2.0, 3.5, 0.25, -1.5, 2.25, 0.5, -0.75];
        let mut b = vec![-0.5f32, 1.25, 2.0, -3.0, 0.75, -1.0, 4.0, 0.5];
        let before: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        assert!(fwht_norm_in_place(&mut a));
        assert!(fwht_norm_in_place(&mut b));
        let after: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        assert!(
            (before - after).abs() < 1.0e-5,
            "before={before} after={after}"
        );
    }

    #[test]
    fn decode_worker_selector_ignores_noise_level_gain() {
        assert_eq!(select_decode_worker_index(&[450, 446, 447]), Some(0));
    }

    #[test]
    fn decode_worker_selector_accepts_material_two_worker_gain() {
        assert_eq!(select_decode_worker_index(&[1598, 985, 1200]), Some(1));
    }

    #[test]
    fn decode_worker_selector_accepts_material_three_worker_gain() {
        assert_eq!(select_decode_worker_index(&[3960, 2220, 1647]), Some(2));
    }

    #[test]
    fn decode_worker_selector_requires_nonempty_samples() {
        assert_eq!(select_decode_worker_index(&[]), None);
    }

    #[test]
    fn matmul_rejects_null_context_before_pointer_access() {
        // SAFETY: null pointers are intentionally supplied to verify that the
        // FFI guard rejects them before any dereference.
        let status = unsafe {
            rocknpu_matmul_f16_f32_f32(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                4,
                32,
                16,
            )
        };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }
}
