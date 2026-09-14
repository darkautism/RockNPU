use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_conv::Fp16Conv2dExecutor;
use rocknpu_onnx::{CnnOnnxModel, TensorF16};
use rocknpu_ops::{ExecutionTarget, SingleNpuBackend};
use std::error::Error;
use std::fs;

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("bad f32 byte length {path}").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_bytes = fs::read("artifacts/mnist-8.onnx")?;
    let model = CnnOnnxModel::from_bytes(&model_bytes)?;
    if model.node_count() != 12 {
        return Err(format!("expected 12 nodes got {}", model.node_count()).into());
    }
    let inputs = read_f32("artifacts/mnist8-input100-f32.bin")?;
    let reference = read_f32("artifacts/mnist8-ref100-f32.bin")?;
    let labels = fs::read("artifacts/mnist8-label100-u8.bin")?;
    if inputs.len() != 100 * 784 || reference.len() != 100 * 10 || labels.len() != 100 {
        return Err("MNIST-8 artifact size mismatch".into());
    }

    let dev = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&dev)?;
    let mut conv_backend = Fp16Conv2dExecutor::new(&dev)?;
    let mut conv_scratch_after_first = None;
    let mut pred_match = 0usize;
    let mut ref_correct = 0usize;
    let mut npu_correct = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut trace_saved = false;
    let mut all_output_bits = Vec::with_capacity(100 * 10 * 2);
    let mut accepted_stats = None;
    for sample in 0..100 {
        let x = TensorF16::from_vec(
            vec![1, 1, 28, 28],
            inputs[sample * 784..(sample + 1) * 784]
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect(),
        )?;
        let (out, stats, trace) = if sample == 0 {
            let (o, s, t) = model.run_fp16_traced_with_conv(
                &x,
                ExecutionTarget::NpuSingle,
                Some(&mut backend),
                Some(&mut conv_backend),
            )?;
            (o, s, Some(t))
        } else {
            let (o, s) = model.run_fp16_with_conv(
                &x,
                ExecutionTarget::NpuSingle,
                Some(&mut backend),
                Some(&mut conv_backend),
            )?;
            (o, s, None)
        };
        if out.dims() != [1, 10] {
            return Err(format!("sample {sample} output dims {:?}", out.dims()).into());
        }
        if stats.conv_nodes != 2
            || stats.maxpool_nodes != 2
            || stats.reshape_nodes != 2
            || stats.matmul_nodes != 1
            || stats.add_nodes != 3
            || stats.relu_nodes != 2
            || stats.npu_dense_nodes != 1
            || stats.npu_conv_nodes != 2
            || stats.padded_npu_dense_nodes != 1
        {
            return Err(format!("unexpected stats {stats:?}").into());
        }
        accepted_stats = Some(stats);
        if sample == 0 {
            conv_scratch_after_first = Some(conv_backend.scratch_stats());
        } else if Some(conv_backend.scratch_stats()) != conv_scratch_after_first {
            return Err(format!(
                "Conv scratch changed after first full inference at sample {sample}: {:?} -> {:?}",
                conv_scratch_after_first,
                conv_backend.scratch_stats()
            )
            .into());
        }
        let got: Vec<f32> = out.values().iter().map(|x| x.to_f32()).collect();
        for v in out.values() {
            all_output_bits.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let r = &reference[sample * 10..(sample + 1) * 10];
        let gp = argmax(&got);
        let rp = argmax(r);
        let label = labels[sample] as usize;
        if gp == rp {
            pred_match += 1;
        }
        if rp == label {
            ref_correct += 1;
        }
        if gp == label {
            npu_correct += 1;
        }
        for (&a, &b) in got.iter().zip(r) {
            let e = (a - b).abs();
            max_abs = max_abs.max(e);
            sum_abs += e as f64;
        }
        if let Some(trace) = trace {
            let mut meta = String::from("index\top\tname\tdims\toffset\tcount\n");
            let mut raw = Vec::new();
            let mut offset = 0usize;
            for (i, t) in trace.iter().enumerate() {
                meta.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\n",
                    i,
                    t.op,
                    t.name,
                    t.dims
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x"),
                    offset,
                    t.values.len()
                ));
                for v in &t.values {
                    raw.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                offset += t.values.len();
            }
            fs::write("artifacts/mnist8-allnpu-trace.tsv", meta)?;
            fs::write("artifacts/mnist8-allnpu-trace-f16.bin", raw)?;
            trace_saved = true;
        }
    }
    if pred_match != 100 {
        return Err(format!(
            "MNIST-8 top1 differs from ONNX reference on {}/100",
            100 - pred_match
        )
        .into());
    }
    fs::write("artifacts/mnist8-allnpu100-f16.bin", all_output_bits)?;
    let mean_abs = sum_abs as f32 / 1000.0;
    let stats = accepted_stats.unwrap();
    println!(
        "MNIST-8 CNN ALL-NPU-COMPUTE PASS nodes={} conv_nodes={} conv_npu={} pool_cpu={} reshape_cpu={} add_cpu={} relu_cpu={} matmul_npu={} padded_npu={} top1_ref={}/100 ref_accuracy={}/100 npu_accuracy={}/100 max_abs={:.8} mean_abs={:.8} trace_saved={} conv_scratch={:?}",
        model.node_count(),
        stats.conv_nodes,
        stats.npu_conv_nodes,
        stats.maxpool_nodes,
        stats.reshape_nodes,
        stats.add_nodes,
        stats.relu_nodes,
        stats.npu_dense_nodes,
        stats.padded_npu_dense_nodes,
        pred_match,
        ref_correct,
        npu_correct,
        max_abs,
        mean_abs,
        trace_saved,
        conv_backend.scratch_stats()
    );
    Ok(())
}
