//! Minimal RK3588 PC-chain trailer patching for Rocket batched jobs.
//!
//! RockNPU's regcmd encoders end in four u64 ops:
//! `[OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]`.  The first slot can be
//! replaced with a PC base redirect and the amount op can describe the next
//! program.  This mirrors the public ISC ork-driver PC descriptor semantics
//! while staying in RockNPU's native u64 command representation.

const OP_NONE: u16 = 0x0000;
const OP_REG_PC: u16 = 0x0101;
const PC_BASE_ADDRESS: u16 = 0x0010;
const PC_REGISTER_AMOUNTS: u16 = 0x0014;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainError {
    AddressAbove32Bit(u64),
    ProgramTooShort,
    EmptySuccessor,
    UnexpectedTrailer,
}

impl core::fmt::Display for ChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AddressAbove32Bit(v) => {
                write!(f, "PC-chain IOVA 0x{v:x} exceeds the 32-bit PC address field")
            }
            Self::ProgramTooShort => write!(f, "regcmd is too short to contain the four-op trailer"),
            Self::EmptySuccessor => write!(f, "PC-chain successor must contain at least one regcmd word"),
            Self::UnexpectedTrailer => write!(f, "regcmd does not have RockNPU's expected PC trailer"),
        }
    }
}
impl std::error::Error for ChainError {}

#[inline]
fn op(kind: u16, value: u32, reg: u16) -> u64 {
    ((kind as u64) << 48) | ((value as u64) << 16) | reg as u64
}

/// PC_DATA_AMOUNT for a Rocket task containing `regcmd_words` u64 commands.
#[inline]
pub const fn encoded_amount(regcmd_words: usize) -> u32 {
    ((regcmd_words + 1) / 2 - 1) as u32
}

#[inline]
pub const fn padded_words(regcmd_words: usize) -> usize {
    (regcmd_words + 1) & !1
}

fn trailer(ops: &[u64]) -> Result<usize, ChainError> {
    let base = ops.len().checked_sub(4).ok_or(ChainError::ProgramTooShort)?;
    let empty = ops[base];
    let amount = ops[base + 1];
    if (empty >> 48) as u16 != OP_NONE
        || (amount >> 48) as u16 != OP_REG_PC
        || amount as u16 != PC_REGISTER_AMOUNTS
    {
        return Err(ChainError::UnexpectedTrailer);
    }
    Ok(base)
}

/// Point this program at a concrete successor and encode that successor's
/// regcmd length. The command count is unchanged.
pub fn link_to_next(
    ops: &mut [u64],
    next_iova: u64,
    next_regcmd_words: usize,
) -> Result<(), ChainError> {
    if next_regcmd_words == 0 {
        return Err(ChainError::EmptySuccessor);
    }
    let next = u32::try_from(next_iova).map_err(|_| ChainError::AddressAbove32Bit(next_iova))?;
    let base = trailer(ops)?;
    ops[base] = op(OP_REG_PC, next, PC_BASE_ADDRESS);
    ops[base + 1] = op(
        OP_REG_PC,
        encoded_amount(next_regcmd_words),
        PC_REGISTER_AMOUNTS,
    );
    Ok(())
}

/// Restore the terminal program's first trailer slot to OP_NONE.
pub fn seal_last(ops: &mut [u64]) -> Result<(), ChainError> {
    let base = ops.len().checked_sub(4).ok_or(ChainError::ProgramTooShort)?;
    let amount = ops[base + 1];
    if (amount >> 48) as u16 != OP_REG_PC || amount as u16 != PC_REGISTER_AMOUNTS {
        return Err(ChainError::UnexpectedTrailer);
    }
    ops[base] = op(OP_NONE, 0, 0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> [u64; 8] {
        [
            op(0x0201, 1, 0x100c),
            op(0x1001, 2, 0x400c),
            op(0x2001, 3, 0x500c),
            op(0x2001, 4, 0x5010),
            op(OP_NONE, 0, 0),
            op(OP_REG_PC, 0, PC_REGISTER_AMOUNTS),
            op(0x0041, 0, 0),
            op(0x0081, 0x1d, 0x0008),
        ]
    }

    #[test]
    fn patches_only_the_two_pc_trailer_words() {
        let mut ops = sample();
        let before = ops;
        link_to_next(&mut ops, 0x1234_5000, 73).unwrap();
        assert_eq!(&ops[..4], &before[..4]);
        assert_eq!(ops[4], op(OP_REG_PC, 0x1234_5000, PC_BASE_ADDRESS));
        assert_eq!(ops[5], op(OP_REG_PC, 36, PC_REGISTER_AMOUNTS));
        assert_eq!(&ops[6..], &before[6..]);
    }

    #[test]
    fn seal_clears_the_forward_address() {
        let mut ops = sample();
        link_to_next(&mut ops, 0x1234_5000, 110).unwrap();
        seal_last(&mut ops).unwrap();
        assert_eq!(ops[4], 0);
    }

    #[test]
    fn rejects_high_iova_and_drifted_trailer() {
        let mut ops = sample();
        assert_eq!(
            link_to_next(&mut ops, 0x1_0000_0000, 73),
            Err(ChainError::AddressAbove32Bit(0x1_0000_0000))
        );
        assert_eq!(
            link_to_next(&mut ops, 0x1000, 0),
            Err(ChainError::EmptySuccessor)
        );
        ops[4] = op(0x0201, 1, 0x100c);
        assert_eq!(
            link_to_next(&mut ops, 0x1000, 73),
            Err(ChainError::UnexpectedTrailer)
        );
    }
}
