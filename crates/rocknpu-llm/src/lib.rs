use half::f16;
use rocknpu_ops::{
    ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, OpError, PreparedFp16Matmul,
    SingleNpuBackend, execute_cpu,
};
use rocknpu_tensor::{Matrix, TensorError};
use std::fmt;

#[derive(Debug)]
pub enum LlmError {
    InvalidShape(String),
    Tensor(TensorError),
    Op(OpError),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message) => write!(f, "invalid LLM tensor shape: {message}"),
            Self::Tensor(error) => error.fmt(f),
            Self::Op(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for LlmError {}

impl From<TensorError> for LlmError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

impl From<OpError> for LlmError {
    fn from(value: OpError) -> Self {
        Self::Op(value)
    }
}

fn checked_product(values: &[usize]) -> Result<usize, LlmError> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| LlmError::InvalidShape("tensor element count overflow".into()))
}

/// Row-wise RMSNorm with FP32 reduction and FP16 storage.
pub fn rms_norm(
    input: &[f16],
    rows: usize,
    hidden: usize,
    weight: &[f16],
    eps: f32,
) -> Result<Vec<f16>, LlmError> {
    if rows == 0 || hidden == 0 || input.len() != rows.saturating_mul(hidden) {
        return Err(LlmError::InvalidShape(
            "RMSNorm input must have exact [rows, hidden] length".into(),
        ));
    }
    if weight.len() != hidden {
        return Err(LlmError::InvalidShape(
            "RMSNorm weight must have hidden elements".into(),
        ));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(LlmError::InvalidShape(
            "RMSNorm epsilon must be finite and positive".into(),
        ));
    }

    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(hidden) {
        let mean_square = row
            .iter()
            .map(|value| {
                let value = value.to_f32();
                value * value
            })
            .sum::<f32>()
            / hidden as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&value, &gain)| f16::from_f32(value.to_f32() * scale * gain.to_f32())),
        );
    }
    Ok(output)
}

/// Llama/Qwen-style rotate-half RoPE over [seq, heads, head_dim].
pub fn apply_rope(
    values: &mut [f16],
    seq: usize,
    heads: usize,
    head_dim: usize,
    position_start: usize,
    theta: f32,
) -> Result<(), LlmError> {
    if seq == 0 || heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
        return Err(LlmError::InvalidShape(
            "RoPE requires nonzero seq/heads and an even head_dim".into(),
        ));
    }
    if values.len() != checked_product(&[seq, heads, head_dim])? {
        return Err(LlmError::InvalidShape(
            "RoPE values must have exact [seq, heads, head_dim] length".into(),
        ));
    }
    if !(theta.is_finite() && theta > 0.0) {
        return Err(LlmError::InvalidShape(
            "RoPE theta must be finite and positive".into(),
        ));
    }

    let half = head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for token in 0..seq {
        let position = (position_start + token) as f32;
        for i in 0..half {
            let inv_freq = theta.powf(-((2 * i) as f32) / head_dim as f32);
            let angle = position * inv_freq;
            cos[i] = angle.cos();
            sin[i] = angle.sin();
        }
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for i in 0..half {
                let left = values[base + i].to_f32();
                let right = values[base + half + i].to_f32();
                values[base + i] = f16::from_f32(left * cos[i] - right * sin[i]);
                values[base + half + i] = f16::from_f32(right * cos[i] + left * sin[i]);
            }
        }
    }
    Ok(())
}

/// SwiGLU activation: silu(gate) * up.
pub fn swiglu(gate: &[f16], up: &[f16]) -> Result<Vec<f16>, LlmError> {
    if gate.len() != up.len() {
        return Err(LlmError::InvalidShape(
            "SwiGLU gate/up tensors must have equal lengths".into(),
        ));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| {
            let gate = gate.to_f32();
            let silu = gate / (1.0 + (-gate).exp());
            f16::from_f32(silu * up.to_f32())
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionConfig {
    fn validate(self) -> Result<(), LlmError> {
        if self.query_heads == 0 || self.kv_heads == 0 || self.head_dim == 0 {
            return Err(LlmError::InvalidShape(
                "attention head counts and head_dim must be nonzero".into(),
            ));
        }
        if self.query_heads % self.kv_heads != 0 {
            return Err(LlmError::InvalidShape(
                "query_heads must be divisible by kv_heads for GQA".into(),
            ));
        }
        Ok(())
    }
}

/// CPU reference causal attention for MHA/GQA.
/// q is [q_len, query_heads, head_dim]; k/v are [kv_len, kv_heads, head_dim].
/// query_start_position is the absolute position of q[0] in the KV sequence.
pub fn causal_attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    q_len: usize,
    kv_len: usize,
    query_start_position: usize,
    config: AttentionConfig,
) -> Result<Vec<f16>, LlmError> {
    config.validate()?;
    if q_len == 0 || kv_len == 0 || query_start_position >= kv_len {
        return Err(LlmError::InvalidShape(
            "attention sequence lengths/query start are invalid".into(),
        ));
    }
    if query_start_position.saturating_add(q_len) > kv_len {
        return Err(LlmError::InvalidShape(
            "attention query range extends beyond KV sequence".into(),
        ));
    }
    let q_stride = config.query_heads * config.head_dim;
    let kv_stride = config.kv_heads * config.head_dim;
    if q.len() != q_len.saturating_mul(q_stride)
        || k.len() != kv_len.saturating_mul(kv_stride)
        || v.len() != k.len()
    {
        return Err(LlmError::InvalidShape(
            "attention Q/K/V lengths do not match their declared shapes".into(),
        ));
    }

    let queries_per_kv = config.query_heads / config.kv_heads;
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut output = vec![f16::ZERO; q.len()];
    let mut scores = vec![0.0f32; kv_len];

    for query in 0..q_len {
        let absolute_query = query_start_position + query;
        let visible = absolute_query + 1;
        for q_head in 0..config.query_heads {
            let kv_head = q_head / queries_per_kv;
            let q_base = (query * config.query_heads + q_head) * config.head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for key in 0..visible {
                let k_base = (key * config.kv_heads + kv_head) * config.head_dim;
                let mut dot = 0.0f32;
                for dim in 0..config.head_dim {
                    dot += q[q_base + dim].to_f32() * k[k_base + dim].to_f32();
                }
                let score = dot * scale;
                scores[key] = score;
                max_score = max_score.max(score);
            }
            let mut denominator = 0.0f32;
            for score in &mut scores[..visible] {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            let out_base = q_base;
            for dim in 0..config.head_dim {
                let mut sum = 0.0f32;
                for key in 0..visible {
                    let v_base = (key * config.kv_heads + kv_head) * config.head_dim;
                    sum += scores[key] / denominator * v[v_base + dim].to_f32();
                }
                output[out_base + dim] = f16::from_f32(sum);
            }
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
pub struct KvCache {
    kv_heads: usize,
    head_dim: usize,
    max_tokens: usize,
    tokens: usize,
    keys: Vec<f16>,
    values: Vec<f16>,
}

impl KvCache {
    pub fn new(kv_heads: usize, head_dim: usize, max_tokens: usize) -> Result<Self, LlmError> {
        if kv_heads == 0 || head_dim == 0 || max_tokens == 0 {
            return Err(LlmError::InvalidShape(
                "KV cache dimensions must be nonzero".into(),
            ));
        }
        let capacity = checked_product(&[kv_heads, head_dim, max_tokens])?;
        Ok(Self {
            kv_heads,
            head_dim,
            max_tokens,
            tokens: 0,
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        })
    }

    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub const fn capacity(&self) -> usize {
        self.max_tokens
    }

    pub fn keys(&self) -> &[f16] {
        &self.keys
    }

    pub fn values(&self) -> &[f16] {
        &self.values
    }

    pub fn append(&mut self, keys: &[f16], values: &[f16], tokens: usize) -> Result<(), LlmError> {
        if tokens == 0 || self.tokens.saturating_add(tokens) > self.max_tokens {
            return Err(LlmError::InvalidShape(
                "KV cache append exceeds capacity or has zero tokens".into(),
            ));
        }
        let expected = checked_product(&[tokens, self.kv_heads, self.head_dim])?;
        if keys.len() != expected || values.len() != expected {
            return Err(LlmError::InvalidShape(
                "KV cache append tensors have the wrong length".into(),
            ));
        }
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.tokens += tokens;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.tokens = 0;
        self.keys.clear();
        self.values.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearExecution {
    Cpu,
    Npu,
}

pub struct HybridLinear<'d> {
    weights: Matrix<f16>,
    prepared: Option<PreparedFp16Matmul<'d>>,
}

impl<'d> HybridLinear<'d> {
    /// Weights are row-major [out_features, in_features].
    pub fn prepare(
        backend: Option<&SingleNpuBackend<'d>>,
        out_features: usize,
        in_features: usize,
        weights: Vec<f16>,
    ) -> Result<Self, LlmError> {
        let weights = Matrix::from_vec(out_features, in_features, weights)?;
        let prepared = if in_features % 32 == 0 && out_features % 16 == 0 {
            backend
                .map(|backend| backend.prepare_fp16_compatible_m(&weights))
                .transpose()?
        } else {
            None
        };
        Ok(Self { weights, prepared })
    }

    pub fn in_features(&self) -> usize {
        self.weights.cols()
    }

    pub fn out_features(&self) -> usize {
        self.weights.rows()
    }

    pub fn has_resident_npu_weights(&self) -> bool {
        self.prepared.is_some()
    }

    pub fn run(
        &self,
        backend: Option<&mut SingleNpuBackend<'d>>,
        input: &[f16],
        rows: usize,
    ) -> Result<(Vec<f16>, LinearExecution), LlmError> {
        let input = Matrix::from_vec(rows, self.in_features(), input.to_vec())?;
        if rows % 4 == 0
            && let (Some(backend), Some(prepared)) = (backend, self.prepared.as_ref())
        {
            let output = backend.execute_prepared_fp16_compatible_m(prepared, &input)?;
            return Ok((output.values().to_vec(), LinearExecution::Npu));
        }

        let spec = MatmulSpec::new(
            rows,
            self.in_features(),
            self.out_features(),
            MatmulPrecision::Fp16Fast,
            ExecutionTarget::Cpu,
        );
        match execute_cpu(spec, &input, &self.weights)? {
            MatmulOutput::F16(output) => Ok((output.values().to_vec(), LinearExecution::Cpu)),
            MatmulOutput::F32(_) => unreachable!("Fp16Fast CPU path returned FP32"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformerBlockConfig {
    pub hidden_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl TransformerBlockConfig {
    fn validate(self) -> Result<(), LlmError> {
        AttentionConfig {
            query_heads: self.query_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
        .validate()?;
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.query_heads.saturating_mul(self.head_dim) != self.hidden_size
        {
            return Err(LlmError::InvalidShape(
                "Transformer block requires hidden_size == query_heads * head_dim".into(),
            ));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return Err(LlmError::InvalidShape(
                "Transformer block RMS epsilon must be finite and positive".into(),
            ));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(LlmError::InvalidShape(
                "Transformer block RoPE theta must be finite and positive".into(),
            ));
        }
        Ok(())
    }

    pub const fn kv_size(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlockWeights {
    pub attention_norm: Vec<f16>,
    pub q_proj: Vec<f16>,
    pub k_proj: Vec<f16>,
    pub v_proj: Vec<f16>,
    pub o_proj: Vec<f16>,
    pub ffn_norm: Vec<f16>,
    pub gate_proj: Vec<f16>,
    pub up_proj: Vec<f16>,
    pub down_proj: Vec<f16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransformerBlockStats {
    pub npu_linears: usize,
    pub cpu_linears: usize,
}

impl TransformerBlockStats {
    fn record(&mut self, execution: LinearExecution) {
        match execution {
            LinearExecution::Cpu => self.cpu_linears += 1,
            LinearExecution::Npu => self.npu_linears += 1,
        }
    }
}

pub struct TransformerBlock<'d> {
    config: TransformerBlockConfig,
    attention_norm: Vec<f16>,
    q_proj: HybridLinear<'d>,
    k_proj: HybridLinear<'d>,
    v_proj: HybridLinear<'d>,
    o_proj: HybridLinear<'d>,
    ffn_norm: Vec<f16>,
    gate_proj: HybridLinear<'d>,
    up_proj: HybridLinear<'d>,
    down_proj: HybridLinear<'d>,
}

impl<'d> TransformerBlock<'d> {
    pub fn prepare(
        config: TransformerBlockConfig,
        weights: TransformerBlockWeights,
        backend: Option<&SingleNpuBackend<'d>>,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        if weights.attention_norm.len() != config.hidden_size
            || weights.ffn_norm.len() != config.hidden_size
        {
            return Err(LlmError::InvalidShape(
                "Transformer block RMSNorm weights must match hidden_size".into(),
            ));
        }
        let kv_size = config.kv_size();
        Ok(Self {
            config,
            attention_norm: weights.attention_norm,
            q_proj: HybridLinear::prepare(
                backend,
                config.hidden_size,
                config.hidden_size,
                weights.q_proj,
            )?,
            k_proj: HybridLinear::prepare(backend, kv_size, config.hidden_size, weights.k_proj)?,
            v_proj: HybridLinear::prepare(backend, kv_size, config.hidden_size, weights.v_proj)?,
            o_proj: HybridLinear::prepare(
                backend,
                config.hidden_size,
                config.hidden_size,
                weights.o_proj,
            )?,
            ffn_norm: weights.ffn_norm,
            gate_proj: HybridLinear::prepare(
                backend,
                config.intermediate_size,
                config.hidden_size,
                weights.gate_proj,
            )?,
            up_proj: HybridLinear::prepare(
                backend,
                config.intermediate_size,
                config.hidden_size,
                weights.up_proj,
            )?,
            down_proj: HybridLinear::prepare(
                backend,
                config.hidden_size,
                config.intermediate_size,
                weights.down_proj,
            )?,
        })
    }

    pub const fn config(&self) -> TransformerBlockConfig {
        self.config
    }

    pub fn resident_projection_count(&self) -> usize {
        [
            self.q_proj.has_resident_npu_weights(),
            self.k_proj.has_resident_npu_weights(),
            self.v_proj.has_resident_npu_weights(),
            self.o_proj.has_resident_npu_weights(),
            self.gate_proj.has_resident_npu_weights(),
            self.up_proj.has_resident_npu_weights(),
            self.down_proj.has_resident_npu_weights(),
        ]
        .into_iter()
        .filter(|resident| *resident)
        .count()
    }

    /// Full prefill for one transformer block. Large projections may run on NPU;
    /// normalization, RoPE, causal GQA, SwiGLU and residuals stay on CPU for now.
    pub fn run_prefill(
        &self,
        mut backend: Option<&mut SingleNpuBackend<'d>>,
        input: &[f16],
        tokens: usize,
        position_start: usize,
    ) -> Result<(Vec<f16>, TransformerBlockStats), LlmError> {
        if tokens == 0 || input.len() != tokens.saturating_mul(self.config.hidden_size) {
            return Err(LlmError::InvalidShape(
                "Transformer prefill input must have shape [tokens, hidden_size]".into(),
            ));
        }
        let mut stats = TransformerBlockStats::default();
        let normalized = rms_norm(
            input,
            tokens,
            self.config.hidden_size,
            &self.attention_norm,
            self.config.rms_norm_eps,
        )?;

        let (mut q, execution) = self
            .q_proj
            .run(backend.as_deref_mut(), &normalized, tokens)?;
        stats.record(execution);
        let (mut k, execution) = self
            .k_proj
            .run(backend.as_deref_mut(), &normalized, tokens)?;
        stats.record(execution);
        let (v, execution) = self
            .v_proj
            .run(backend.as_deref_mut(), &normalized, tokens)?;
        stats.record(execution);

        apply_rope(
            &mut q,
            tokens,
            self.config.query_heads,
            self.config.head_dim,
            position_start,
            self.config.rope_theta,
        )?;
        apply_rope(
            &mut k,
            tokens,
            self.config.kv_heads,
            self.config.head_dim,
            position_start,
            self.config.rope_theta,
        )?;
        let attention = causal_attention(
            &q,
            &k,
            &v,
            tokens,
            tokens,
            0,
            AttentionConfig {
                query_heads: self.config.query_heads,
                kv_heads: self.config.kv_heads,
                head_dim: self.config.head_dim,
            },
        )?;
        let (attention_output, execution) =
            self.o_proj
                .run(backend.as_deref_mut(), &attention, tokens)?;
        stats.record(execution);
        let after_attention = residual_add(input, &attention_output)?;

        let normalized = rms_norm(
            &after_attention,
            tokens,
            self.config.hidden_size,
            &self.ffn_norm,
            self.config.rms_norm_eps,
        )?;
        let (gate, execution) = self
            .gate_proj
            .run(backend.as_deref_mut(), &normalized, tokens)?;
        stats.record(execution);
        let (up, execution) = self
            .up_proj
            .run(backend.as_deref_mut(), &normalized, tokens)?;
        stats.record(execution);
        let activated = swiglu(&gate, &up)?;
        let (ffn_output, execution) =
            self.down_proj
                .run(backend.as_deref_mut(), &activated, tokens)?;
        stats.record(execution);
        Ok((residual_add(&after_attention, &ffn_output)?, stats))
    }
}

fn residual_add(lhs: &[f16], rhs: &[f16]) -> Result<Vec<f16>, LlmError> {
    if lhs.len() != rhs.len() {
        return Err(LlmError::InvalidShape(
            "residual add tensors must have equal lengths".into(),
        ));
    }
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| f16::from_f32(lhs.to_f32() + rhs.to_f32()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16s(values: &[f32]) -> Vec<f16> {
        values.iter().copied().map(f16::from_f32).collect()
    }

    #[test]
    fn rms_norm_matches_simple_reference() {
        let input = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let weight = f16s(&[1.0, 1.0, 1.0, 1.0]);
        let output = rms_norm(&input, 1, 4, &weight, 1e-5).unwrap();
        let scale = 1.0 / (7.5f32 + 1e-5).sqrt();
        for (got, want) in output.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((got.to_f32() - want * scale).abs() < 0.002);
        }
    }

    #[test]
    fn rope_position_zero_is_identity_and_preserves_pair_norm() {
        let original = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let mut zero = original.clone();
        apply_rope(&mut zero, 1, 1, 4, 0, 10_000.0).unwrap();
        assert_eq!(zero, original);

        let mut rotated = original.clone();
        apply_rope(&mut rotated, 1, 1, 4, 3, 10_000.0).unwrap();
        for i in 0..2 {
            let before = original[i].to_f32().powi(2) + original[2 + i].to_f32().powi(2);
            let after = rotated[i].to_f32().powi(2) + rotated[2 + i].to_f32().powi(2);
            assert!((before - after).abs() < 0.02);
        }
    }

    #[test]
    fn causal_gqa_obeys_visibility() {
        let config = AttentionConfig {
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
        };
        let q = f16s(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let k = f16s(&[1.0, 0.0, 0.0, 1.0]);
        let v = f16s(&[2.0, 4.0, 10.0, 20.0]);
        let out = causal_attention(&q, &k, &v, 2, 2, 0, config).unwrap();
        assert_eq!(&out[..4], &f16s(&[2.0, 4.0, 2.0, 4.0]));
        assert!(out[4].to_f32() > 2.0 && out[4].to_f32() < 10.0);
        assert!(out[5].to_f32() > 4.0 && out[5].to_f32() < 20.0);
    }

    #[test]
    fn kv_cache_appends_and_clears() {
        let mut cache = KvCache::new(2, 4, 8).unwrap();
        cache
            .append(&vec![f16::ONE; 16], &vec![f16::ZERO; 16], 2)
            .unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.keys().len(), 16);
        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.keys().is_empty());
    }

    fn tiny_block_weights(config: TransformerBlockConfig) -> TransformerBlockWeights {
        fn weights(count: usize, seed: usize) -> Vec<f16> {
            (0..count)
                .map(|index| {
                    let value = ((index.wrapping_mul(seed) % 17) as f32 - 8.0) / 128.0;
                    f16::from_f32(value)
                })
                .collect()
        }
        let kv = config.kv_size();
        TransformerBlockWeights {
            attention_norm: vec![f16::ONE; config.hidden_size],
            q_proj: weights(config.hidden_size * config.hidden_size, 3),
            k_proj: weights(kv * config.hidden_size, 5),
            v_proj: weights(kv * config.hidden_size, 7),
            o_proj: weights(config.hidden_size * config.hidden_size, 11),
            ffn_norm: vec![f16::ONE; config.hidden_size],
            gate_proj: weights(config.intermediate_size * config.hidden_size, 13),
            up_proj: weights(config.intermediate_size * config.hidden_size, 17),
            down_proj: weights(config.hidden_size * config.intermediate_size, 19),
        }
    }

    #[test]
    fn transformer_block_cpu_prefill_runs_all_seven_linears() {
        let config = TransformerBlockConfig {
            hidden_size: 64,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 32,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let block = TransformerBlock::prepare(config, tiny_block_weights(config), None).unwrap();
        let input = vec![f16::from_f32(0.01); 4 * config.hidden_size];
        let (output, stats) = block.run_prefill(None, &input, 4, 0).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(stats.npu_linears, 0);
        assert_eq!(stats.cpu_linears, 7);
        assert!(output.iter().all(|value| value.to_f32().is_finite()));
    }

    #[test]
    fn hybrid_linear_cpu_handles_decode_m1() {
        let weights = vec![f16::ONE; 16 * 32];
        let linear = HybridLinear::prepare(None, 16, 32, weights).unwrap();
        let input = vec![f16::from_f32(0.25); 32];
        let (output, execution) = linear.run(None, &input, 1).unwrap();
        assert_eq!(execution, LinearExecution::Cpu);
        assert_eq!(output.len(), 16);
        assert!(
            output
                .iter()
                .all(|value| (value.to_f32() - 8.0).abs() < 0.01)
        );
    }
}
