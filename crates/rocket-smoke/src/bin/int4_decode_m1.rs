use rocket_runtime::{RocketDevice, Task};
use rocknpu_regcmd::{INT4_REGCMD_COUNT, Int4DecodeDesc, encode_int4_decode_m1};
use std::time::Instant;

const K: usize = 2048;
const N: usize = 2048;
const WAIT_NS: i64 = 2_000_000_000;

fn put_nibble(dst: &mut [u8], index: usize, value: i8) {
    let nibble = (value as u8) & 0x0f;
    if index.is_multiple_of(2) {
        dst[index / 2] = (dst[index / 2] & 0xf0) | nibble;
    } else {
        dst[index / 2] = (dst[index / 2] & 0x0f) | (nibble << 4);
    }
}

fn pack_activation(a: &[i8]) -> Vec<u8> {
    let mut packed = vec![0u8; K / 2];
    for (k, &value) in a.iter().enumerate() {
        put_nibble(&mut packed, k, value);
    }
    packed
}

fn pack_weights_kn(b_kn: &[i8]) -> Vec<u8> {
    let mut packed = vec![0u8; K * N / 2];
    let kt = K / 32;
    for nb in 0..N / 64 {
        for kb in 0..kt {
            for nl in 0..64 {
                for kk in 0..32 {
                    let packed_index = (((nb * kt + kb) * 64 + nl) * 32) + kk;
                    let k = kb * 32 + kk;
                    let n = nb * 64 + nl;
                    put_nibble(&mut packed, packed_index, b_kn[k * N + n]);
                }
            }
        }
    }
    packed
}

fn cpu_reference(a: &[i8], b_kn: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; N];
    for n in 0..N {
        let mut sum = 0i32;
        for k in 0..K {
            sum += i32::from(a[k]) * i32::from(b_kn[k * N + n]);
        }
        out[n] = sum;
    }
    out
}

fn read_i16(bytes: &[u8], index: usize) -> i16 {
    let p = index * 2;
    i16::from_le_bytes(bytes[p..p + 2].try_into().unwrap())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<i8> = (0..K).map(|i| ((i * 7 + 3) % 5) as i8 - 2).collect();
    let b: Vec<i8> = (0..K * N).map(|i| ((i * 5 + 1) % 5) as i8 - 2).collect();
    let expected = cpu_reference(&a, &b);
    let packed_a = pack_activation(&a);
    let packed_b = pack_weights_kn(&b);

    let device = RocketDevice::open()?;
    let mut regcmd = device.alloc_buffer(4096)?;
    let mut input = device.alloc_buffer(packed_a.len())?;
    let mut weights = device.alloc_buffer(packed_b.len())?;
    let mut output = device.alloc_buffer(N * 2)?;
    for bo in [&regcmd, &input, &weights, &output] {
        if bo.dma_address() >> 32 != 0 {
            return Err(format!(
                "BO IOVA 0x{:x} exceeds 32-bit regcmd window",
                bo.dma_address()
            )
            .into());
        }
    }

    input.prep_relative(0)?;
    input.as_mut_slice()[..packed_a.len()].copy_from_slice(&packed_a);
    input.fini()?;
    weights.prep_relative(0)?;
    weights.as_mut_slice()[..packed_b.len()].copy_from_slice(&packed_b);
    weights.fini()?;

    let ops = encode_int4_decode_m1(Int4DecodeDesc::new(
        K,
        N,
        input.dma_address(),
        weights.dma_address(),
        output.dma_address(),
    ))?;
    regcmd.prep_relative(0)?;
    regcmd.as_mut_slice().fill(0);
    for (chunk, word) in regcmd.as_mut_slice().chunks_exact_mut(8).zip(ops.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    regcmd.fini()?;

    let task = Task {
        regcmd: u32::try_from(regcmd.dma_address())?,
        regcmd_count: u32::try_from(INT4_REGCMD_COUNT)?,
    };
    let inputs = [input.handle(), weights.handle(), regcmd.handle()];
    let outputs = [output.handle()];

    // Judge the second run. The existing int8 path likewise treats the first
    // entry into the mode as warm-up; correctness is never inferred from it.
    for run in 0..2 {
        output.prep_relative(0)?;
        output.as_mut_slice().fill(0x7f);
        output.fini()?;
        let start = Instant::now();
        device.submit(&[task], &inputs, &outputs)?;
        output.prep_relative(WAIT_NS)?;
        let elapsed = start.elapsed();
        let mut mismatches = Vec::new();
        for (n, &want) in expected.iter().enumerate() {
            let got = i32::from(read_i16(output.as_slice(), n));
            if got != want && mismatches.len() < 10 {
                mismatches.push((n, want, got));
            }
        }
        output.fini()?;
        println!(
            "W4A4 M=1 run={} K={} N={} regcmd={} submit_wait_ms={:.3} mismatches={}",
            run + 1,
            K,
            N,
            ops.len(),
            elapsed.as_secs_f64() * 1.0e3,
            mismatches.len(),
        );
        if run == 1 && !mismatches.is_empty() {
            return Err(format!("native W4A4 Rocket mismatch examples: {mismatches:?}").into());
        }
    }

    println!("W4A4 DECODE PASS M=1 K={K} N={N} outputs={N} exact=int16");
    Ok(())
}
