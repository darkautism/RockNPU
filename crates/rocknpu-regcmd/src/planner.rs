use crate::{EncodeError, Fp16MatmulDesc};

const BANK_BYTES: usize = 32 * 1024;
const BANKS: usize = 12;
const MAX_MN_TILE: usize = 256;
const MAX_K_TILE: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp16MatmulTile {
    pub m0: usize,
    pub n0: usize,
    pub k0: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fp16MatmulPlan {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub mt: usize,
    pub kt: usize,
    pub nt: usize,
    pub tiles: Vec<Fp16MatmulTile>,
}

impl Fp16MatmulPlan {
    pub fn m_tiles(&self) -> usize {
        self.m.div_ceil(self.mt)
    }
    pub fn k_tiles(&self) -> usize {
        self.k.div_ceil(self.kt)
    }
    pub fn n_tiles(&self) -> usize {
        self.n.div_ceil(self.nt)
    }
}

fn banks(rows: usize, k: usize) -> Result<usize, EncodeError> {
    let bytes = rows
        .checked_mul(k)
        .and_then(|v| v.checked_mul(2))
        .ok_or(EncodeError::SizeOverflow)?;
    Ok(bytes.div_ceil(BANK_BYTES))
}

fn fits(m: usize, n: usize, k: usize) -> Result<bool, EncodeError> {
    Ok(banks(m, k)? + banks(n, k)? <= BANKS)
}

fn validate_shape(m: usize, k: usize, n: usize) -> Result<(), EncodeError> {
    if m == 0 || k == 0 || n == 0 || m % 4 != 0 || k % 32 != 0 || n % 16 != 0 {
        return Err(EncodeError::InvalidShape {
            m,
            k,
            n,
            reason: "planner requires non-zero M%4==0, K%32==0, N%16==0",
        });
    }
    Ok(())
}

fn choose_kt(k: usize, fit_m: usize, nt: usize) -> Result<usize, EncodeError> {
    let mut kt = k.min(MAX_K_TILE);
    kt -= kt % 32;
    while kt > 32 && !fits(fit_m, nt, kt)? {
        kt -= 32;
    }
    if !fits(fit_m, nt, kt)? {
        return Err(EncodeError::CbufUnsupported {
            feature_banks: banks(fit_m, kt)? as u32,
            weight_banks: banks(nt, kt)? as u32,
            reason: "even the minimum aligned K tile does not fit",
        });
    }
    Ok(kt)
}

fn build_plan(
    m: usize,
    k: usize,
    n: usize,
    mt: usize,
    kt: usize,
    nt: usize,
) -> Result<Fp16MatmulPlan, EncodeError> {
    let mut tiles = Vec::new();
    for m0 in (0..m).step_by(mt) {
        let tm = (m - m0).min(mt);
        for n0 in (0..n).step_by(nt) {
            let tn = (n - n0).min(nt);
            for k0 in (0..k).step_by(kt) {
                let tk = (k - k0).min(kt);
                crate::encode_fp16_matmul(Fp16MatmulDesc::new(tm, tk, tn, 0x1000, 0x2000, 0x3000))?;
                tiles.push(Fp16MatmulTile {
                    m0,
                    n0,
                    k0,
                    m: tm,
                    n: tn,
                    k: tk,
                });
            }
        }
    }
    Ok(Fp16MatmulPlan {
        m,
        k,
        n,
        mt,
        kt,
        nt,
        tiles,
    })
}

/// Correctness-first RK3588 fp16 MatMul planner.
///
/// Mt/Nt are capped at the currently validated 256 policy ceiling; Kt is the
/// largest 32-aligned slice up to 16384 that lets both operand tiles fit in the
/// 12 x 32 KiB CBUF. K-split accumulation is intentionally left to the executor.
pub fn plan_fp16_matmul(m: usize, k: usize, n: usize) -> Result<Fp16MatmulPlan, EncodeError> {
    validate_shape(m, k, n)?;
    let mt = m.min(MAX_MN_TILE);
    let nt = n.min(MAX_MN_TILE);
    let kt = choose_kt(k, mt, nt)?;
    build_plan(m, k, n, mt, kt, nt)
}

/// Plan MatMul with N/K tile geometry that is invariant across M.
///
/// Weight packing depends on N/K tile boundaries, while the normal planner may
/// change Kt when M changes. This mode deliberately chooses Kt against the
/// worst-case validated M tile (`256`) and therefore trades some small-M
/// efficiency for a stable resident-weight layout. Any aligned M can then use
/// the same prepacked B[N,K] tiles without repacking.
pub fn plan_fp16_matmul_compatible_m(
    m: usize,
    k: usize,
    n: usize,
) -> Result<Fp16MatmulPlan, EncodeError> {
    validate_shape(m, k, n)?;
    let mt = m.min(MAX_MN_TILE);
    let nt = n.min(MAX_MN_TILE);
    let kt = choose_kt(k, MAX_MN_TILE, nt)?;
    build_plan(m, k, n, mt, kt, nt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_tile_when_shape_fits() {
        let p = plan_fp16_matmul(64, 256, 64).unwrap();
        assert_eq!((p.mt, p.kt, p.nt), (64, 256, 64));
        assert_eq!((p.m_tiles(), p.k_tiles(), p.n_tiles()), (1, 1, 1));
        assert_eq!(p.tiles.len(), 1);
    }

    #[test]
    fn m_tiles_when_full_shape_exceeds_cbuf() {
        let p = plan_fp16_matmul(512, 512, 128).unwrap();
        assert_eq!((p.mt, p.kt, p.nt), (256, 512, 128));
        assert_eq!((p.m_tiles(), p.k_tiles(), p.n_tiles()), (2, 1, 1));
        assert_eq!(p.tiles.len(), 2);
        assert_eq!(p.tiles[0].m0, 0);
        assert_eq!(p.tiles[1].m0, 256);
    }

    #[test]
    fn k_shrinks_to_fit_symmetric_256_tiles() {
        let p = plan_fp16_matmul(256, 1024, 256).unwrap();
        assert_eq!(p.mt, 256);
        assert_eq!(p.nt, 256);
        assert!(p.kt < 1024);
        assert_eq!(p.kt % 32, 0);
        assert!(p.k_tiles() > 1);
        for t in p.tiles {
            assert!(fits(t.m, t.n, t.k).unwrap());
        }
    }

    #[test]
    fn ragged_aligned_tails_remain_legal() {
        let p = plan_fp16_matmul(300, 512, 272).unwrap();
        assert_eq!(p.tiles.last().unwrap().m % 4, 0);
        assert_eq!(p.tiles.last().unwrap().n % 16, 0);
        assert_eq!(p.tiles.last().unwrap().k % 32, 0);
    }

    #[test]
    fn compatible_m_keeps_weight_geometry_stable() {
        let plans: Vec<_> = [4usize, 16, 52, 256, 512]
            .into_iter()
            .map(|m| plan_fp16_matmul_compatible_m(m, 800, 800).unwrap())
            .collect();
        let geometry = (plans[0].kt, plans[0].nt);
        assert!(plans.iter().all(|p| (p.kt, p.nt) == geometry));
        assert!(
            plans
                .iter()
                .all(|p| p.tiles.iter().all(|t| fits(t.m, t.n, t.k).unwrap()))
        );
    }

    #[test]
    fn compatible_m_is_conservative_for_small_m() {
        let exact = plan_fp16_matmul(52, 800, 800).unwrap();
        let compatible = plan_fp16_matmul_compatible_m(52, 800, 800).unwrap();
        assert_eq!(exact.nt, compatible.nt);
        assert!(compatible.kt <= exact.kt);
        assert_eq!(compatible.kt % 32, 0);
    }
}
