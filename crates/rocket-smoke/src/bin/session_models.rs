use half::f16;
use rocknpu::{Session, Tensor};
use std::{error::Error, fs};

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("bad f32 byte length: {path}").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|v| v.0)
        .unwrap()
}

fn run_mnist8() -> Result<(), Box<dyn Error>> {
    let inputs = read_f32("artifacts/mnist8-input100-f32.bin")?;
    let reference = read_f32("artifacts/mnist8-ref100-f32.bin")?;
    let labels = fs::read("artifacts/mnist8-label100-u8.bin")?;
    if inputs.len() != 100 * 784 || reference.len() != 100 * 10 || labels.len() != 100 {
        return Err("MNIST-8 artifact size mismatch".into());
    }

    let session = Session::load("artifacts/mnist-8.onnx")?;
    let prepare = session.prepare_stats();
    if prepare.conv_weight_tensors != 2
        || prepare.dense_weight_tensors != 1
        || prepare.resident_weight_bytes == 0
    {
        return Err(format!("unexpected MNIST-8 Session prepare stats: {prepare:?}").into());
    }

    let mut top1_match = 0usize;
    let mut npu_correct = 0usize;
    let mut max_abs = 0.0f32;
    let mut all_output_bits = Vec::with_capacity(100 * 10 * 2);
    let mut accepted_stats = None;
    for sample in 0..100 {
        let input = Tensor::from_f32(
            vec![1, 1, 28, 28],
            inputs[sample * 784..(sample + 1) * 784].to_vec(),
        )?;
        let output = session.run(input)?;
        if output.tensor().shape() != [1, 10] {
            return Err(format!("MNIST-8 output shape: {:?}", output.tensor().shape()).into());
        }
        let stats = output.stats();
        if stats.conv_nodes != 2
            || stats.npu_conv_nodes != 2
            || stats.maxpool_nodes != 2
            || stats.reshape_nodes != 2
            || stats.matmul_nodes != 1
            || stats.npu_dense_nodes != 1
            || stats.add_nodes != 3
            || stats.relu_nodes != 2
        {
            return Err(format!("unexpected MNIST-8 Session run stats: {stats:?}").into());
        }
        accepted_stats = Some(stats);
        let got = output.tensor().values();
        let want = &reference[sample * 10..(sample + 1) * 10];
        if argmax(got) == argmax(want) {
            top1_match += 1;
        }
        if argmax(got) == labels[sample] as usize {
            npu_correct += 1;
        }
        for (&a, &b) in got.iter().zip(want) {
            max_abs = max_abs.max((a - b).abs());
            all_output_bits.extend_from_slice(&f16::from_f32(a).to_bits().to_le_bytes());
        }
    }
    if top1_match != 100 || npu_correct != 98 {
        return Err(format!(
            "MNIST-8 Session mismatch: top1_ref={top1_match}/100 accuracy={npu_correct}/100"
        )
        .into());
    }
    fs::write("artifacts/mnist8-session100-f16.bin", all_output_bits)?;
    println!(
        "SESSION MNIST-8 PASS top1_ref={top1_match}/100 accuracy={npu_correct}/100 max_abs={max_abs:.8} prepare={prepare:?} run={:?}",
        accepted_stats.unwrap()
    );
    Ok(())
}

fn run_cifar10() -> Result<(), Box<dyn Error>> {
    let input = read_f32("artifacts/cifar10-edgeinfer-input-f32.bin")?;
    let reference = read_f32("artifacts/cifar10-edgeinfer-ref-f32.bin")?;
    if input.len() != 3 * 32 * 32 || reference.len() != 10 {
        return Err("CIFAR-10 artifact size mismatch".into());
    }

    let session = Session::load("artifacts/cifar10-edgeinfer.onnx")?;
    let prepare = session.prepare_stats();
    if prepare.conv_weight_tensors != 3
        || prepare.dense_weight_tensors != 2
        || prepare.resident_weight_bytes == 0
    {
        return Err(format!("unexpected CIFAR-10 Session prepare stats: {prepare:?}").into());
    }
    let output = session.run(Tensor::from_f32(vec![1, 3, 32, 32], input)?)?;
    if output.tensor().shape() != [1, 10] {
        return Err(format!("CIFAR-10 output shape: {:?}", output.tensor().shape()).into());
    }
    let stats = output.stats();
    if stats.conv_nodes != 3
        || stats.npu_conv_nodes != 3
        || stats.maxpool_nodes != 3
        || stats.reshape_nodes != 1
        || stats.gemm_nodes != 2
        || stats.npu_dense_nodes != 2
        || stats.relu_nodes != 4
    {
        return Err(format!("unexpected CIFAR-10 Session run stats: {stats:?}").into());
    }
    let got = output.tensor().values();
    let max_abs = got
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let pred = argmax(got);
    let ref_pred = argmax(&reference);
    if pred != 8 || ref_pred != 8 || max_abs >= 0.1 {
        return Err(format!(
            "CIFAR-10 Session mismatch: pred={pred} ref={ref_pred} max_abs={max_abs}"
        )
        .into());
    }
    let raw: Vec<u8> = got
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    fs::write("artifacts/cifar10-edgeinfer-session-f16.bin", raw)?;
    println!(
        "SESSION CIFAR-10 PASS pred={pred} ref={ref_pred} max_abs={max_abs:.8} prepare={prepare:?} run={stats:?}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_mnist8()?;
    run_cifar10()?;
    println!("SESSION MODEL GATES PASS");
    Ok(())
}
