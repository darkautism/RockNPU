use rocknpu_regcmd::{Fp16MatmulDesc, encode_fp16_matmul};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().ok_or("M")?.parse()?;
    let k: usize = a.next().ok_or("K")?.parse()?;
    let n: usize = a.next().ok_or("N")?.parse()?;
    let ops = encode_fp16_matmul(Fp16MatmulDesc::new(
        m,
        k,
        n,
        0x1111_1000,
        0x2222_2000,
        0x3333_3000,
    ))?;
    println!("{}", ops.len());
    for word in ops {
        println!("0x{word:016x}");
    }
    Ok(())
}
