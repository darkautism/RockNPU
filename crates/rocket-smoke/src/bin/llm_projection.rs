use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_llm::{
    HybridLinear, LinearExecution, TransformerBlock, TransformerBlockConfig,
    TransformerBlockWeights,
};
use rocknpu_ops::SingleNpuBackend;
use std::error::Error;

fn deterministic(count: usize, seed: usize, divisor: f32) -> Vec<f16> {
    (0..count)
        .map(|index| {
            let value = ((index.wrapping_mul(seed) % 29) as f32 - 14.0) / divisor;
            f16::from_f32(value)
        })
        .collect()
}

fn block_weights(config: TransformerBlockConfig) -> TransformerBlockWeights {
    let kv = config.kv_size();
    TransformerBlockWeights {
        attention_norm: vec![f16::ONE; config.hidden_size],
        q_proj: deterministic(config.hidden_size * config.hidden_size, 3, 256.0),
        k_proj: deterministic(kv * config.hidden_size, 5, 256.0),
        v_proj: deterministic(kv * config.hidden_size, 7, 256.0),
        o_proj: deterministic(config.hidden_size * config.hidden_size, 11, 256.0),
        ffn_norm: vec![f16::ONE; config.hidden_size],
        gate_proj: deterministic(config.intermediate_size * config.hidden_size, 13, 256.0),
        up_proj: deterministic(config.intermediate_size * config.hidden_size, 17, 256.0),
        down_proj: deterministic(config.hidden_size * config.intermediate_size, 19, 256.0),
    }
}

fn max_abs(lhs: &[f16], rhs: &[f16]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn Error>> {
    // First prove one projection at a real small-model hidden size.
    let hidden = 896usize;
    let prefill_tokens = 8usize;
    let weights = deterministic(hidden * hidden, 17, 256.0);
    let prefill = deterministic(prefill_tokens * hidden, 13, 64.0);

    let device = RocketDevice::open()?;
    let mut backend = SingleNpuBackend::new(&device)?;
    let linear = HybridLinear::prepare(Some(&backend), hidden, hidden, weights)?;
    if !linear.has_resident_npu_weights() {
        return Err("transformer projection did not prepare resident NPU weights".into());
    }

    let (npu, target) = linear.run(Some(&mut backend), &prefill, prefill_tokens)?;
    if target != LinearExecution::Npu {
        return Err("aligned prefill projection did not execute on NPU".into());
    }
    let (cpu, cpu_target) = linear.run(None, &prefill, prefill_tokens)?;
    if cpu_target != LinearExecution::Cpu || cpu.len() != npu.len() {
        return Err("CPU oracle projection shape/target mismatch".into());
    }
    let projection_max_abs = max_abs(&npu, &cpu);
    if projection_max_abs > 0.02 {
        return Err(format!(
            "NPU transformer projection max_abs={projection_max_abs} exceeds 0.02"
        )
        .into());
    }

    // Current Rocket MatMul contract deliberately sends M=1 decode to CPU.
    let decode = &prefill[..hidden];
    let (_, decode_target) = linear.run(Some(&mut backend), decode, 1)?;
    if decode_target != LinearExecution::Cpu {
        return Err("M=1 decode should use explicit CPU fallback until GEMV is benchmarked".into());
    }

    // Then prove a complete Llama/Qwen-style block. The dimensions are kept
    // moderate so the CPU oracle is cheap, while every Linear still satisfies
    // the same K32/N16 resident-NPU contract used by real model projections.
    let config = TransformerBlockConfig {
        hidden_size: 128,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 32,
        intermediate_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
    };
    let weights = block_weights(config);
    let cpu_block = TransformerBlock::prepare(config, weights.clone(), None)?;
    let npu_block = TransformerBlock::prepare(config, weights, Some(&backend))?;
    if npu_block.resident_projection_count() != 7 {
        return Err(format!(
            "expected seven resident transformer projections, got {}",
            npu_block.resident_projection_count()
        )
        .into());
    }
    let block_input = deterministic(prefill_tokens * config.hidden_size, 23, 128.0);
    let (cpu_output, cpu_stats) = cpu_block.run_prefill(None, &block_input, prefill_tokens, 0)?;
    let (npu_output, npu_stats) =
        npu_block.run_prefill(Some(&mut backend), &block_input, prefill_tokens, 0)?;
    if cpu_stats.cpu_linears != 7 || cpu_stats.npu_linears != 0 {
        return Err(format!("unexpected CPU block placement: {cpu_stats:?}").into());
    }
    if npu_stats.npu_linears != 7 || npu_stats.cpu_linears != 0 {
        return Err(format!("unexpected NPU block placement: {npu_stats:?}").into());
    }
    let block_max_abs = max_abs(&npu_output, &cpu_output);
    if block_max_abs > 0.05 {
        return Err(
            format!("hybrid transformer block max_abs={block_max_abs} exceeds 0.05").into(),
        );
    }

    println!(
        "LLM PROJECTION PASS hidden={hidden} prefill_tokens={prefill_tokens} resident=true prefill=NPU decode_m1=CPU max_abs={projection_max_abs:.8}"
    );
    println!(
        "LLM BLOCK PASS hidden={} heads={}/{} head_dim={} intermediate={} tokens={} resident_projections=7 npu_linears={} cpu_glue=RMSNorm+RoPE+GQA+SwiGLU+residual max_abs={block_max_abs:.8}",
        config.hidden_size,
        config.query_heads,
        config.kv_heads,
        config.head_dim,
        config.intermediate_size,
        prefill_tokens,
        npu_stats.npu_linears,
    );
    Ok(())
}
