use rocknpu_regcmd::{Fp16MatmulDesc, encode_fp16_matmul_fp32_output};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        std::process::exit(2)
    }
    let m = a[1].parse().unwrap();
    let k = a[2].parse().unwrap();
    let n = a[3].parse().unwrap();
    let ops = encode_fp16_matmul_fp32_output(Fp16MatmulDesc::new(
        m, k, n, 0x11111000, 0x22222000, 0x33333000,
    ))
    .unwrap();
    println!("{}", ops.len());
    for (i, w) in ops.iter().enumerate() {
        println!("0x{w:016x}{}", if i + 1 == ops.len() { "" } else { "," });
    }
}
