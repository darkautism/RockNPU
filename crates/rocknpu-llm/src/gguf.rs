use crate::{
    HybridLinear, LinearExecution, LlmError, RopeStyle, TransformerBlock, TransformerBlockConfig,
    TransformerBlockStats, TransformerBlockWeights, rms_norm,
};
use bytemuck::{Pod, pod_read_unaligned};
use half::{bf16, f16};
use llama_gguf::model::layers::{Linear as GgufLinear, NormLayer};
use llama_gguf::model::{ModelConfig, ModelLoader, RopeScalingType, TransformerLayer};
use llama_gguf::tensor::quant::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K, dequantize_q2_k, dequantize_q3_k, dequantize_q4_0,
    dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k,
    dequantize_q6_k, dequantize_q8_0, dequantize_q8_1, dequantize_q8_k,
};
use llama_gguf::{DType, GgufFile, Tensor as GgufTensor, Tokenizer};
use rocknpu_ops::SingleNpuBackend;
use std::mem::size_of;
use std::path::Path;

fn frontend_error(error: impl std::fmt::Display) -> LlmError {
    LlmError::Gguf(error.to_string())
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufModelInfo {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rope_style: RopeStyle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FirstTokenResult {
    pub prompt_tokens: Vec<u32>,
    pub token_id: u32,
    pub text: String,
    pub npu_linears: usize,
    pub cpu_linears: usize,
    pub padded_tokens: usize,
}

pub struct GgufLlama {
    tokenizer: Tokenizer,
    config: ModelConfig,
    embedding: Vec<f16>,
    layers: Vec<TransformerLayer>,
    final_norm: Vec<f16>,
    final_norm_eps: f32,
    output: GgufLinear,
    info: GgufModelInfo,
}

impl GgufLlama {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LlmError> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path).map_err(frontend_error)?;
        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(frontend_error)?;
        drop(gguf);

        let model = ModelLoader::load(path)
            .map_err(frontend_error)?
            .build_model()
            .map_err(frontend_error)?;
        let (config, token_embedding, layers, final_norm, output, _, recurrent_mask, recurrent) =
            model.into_parts();
        if recurrent.is_some() || recurrent_mask.iter().any(|value| *value) {
            return Err(LlmError::Gguf(
                "recurrent/Mamba/DeltaNet layers are not supported by the first RockNPU GGUF path"
                    .into(),
            ));
        }
        validate_model_config(&config)?;

        let embedding = tensor_to_f16(&token_embedding)?;
        let expected_embedding = config
            .vocab_size
            .checked_mul(config.hidden_size)
            .ok_or_else(|| LlmError::Gguf("embedding size overflow".into()))?;
        if embedding.len() != expected_embedding {
            return Err(LlmError::Gguf(format!(
                "token embedding has {} values, expected {expected_embedding}",
                embedding.len()
            )));
        }
        let rms = final_norm
            .as_rms()
            .ok_or_else(|| LlmError::Gguf("first GGUF path requires final RMSNorm".into()))?;
        let final_norm_weights = tensor_to_f16(&rms.weight)?;
        if final_norm_weights.len() != config.hidden_size {
            return Err(LlmError::Gguf(
                "final RMSNorm weight does not match hidden_size".into(),
            ));
        }

        let rope_style = match config.rope_config.rope_type {
            llama_gguf::model::RopeType::Normal => RopeStyle::Normal,
            llama_gguf::model::RopeType::NeoX => RopeStyle::NeoX,
        };
        let info = GgufModelInfo {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_config.freq_base,
            rope_style,
        };

        Ok(Self {
            tokenizer,
            config,
            embedding,
            layers,
            final_norm: final_norm_weights,
            final_norm_eps: rms.eps,
            output,
            info,
        })
    }

    pub fn info(&self) -> &GgufModelInfo {
        &self.info
    }

    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, LlmError> {
        self.tokenizer.encode(text, add_bos).map_err(frontend_error)
    }

    pub fn decode_token(&self, token: u32) -> Result<String, LlmError> {
        self.tokenizer.decode_token(token).map_err(frontend_error)
    }

    pub fn first_token<'d>(
        &self,
        mut backend: Option<&mut SingleNpuBackend<'d>>,
        prompt: &str,
        add_bos: bool,
    ) -> Result<FirstTokenResult, LlmError> {
        let prompt_tokens = self.tokenize(prompt, add_bos)?;
        if prompt_tokens.is_empty() {
            return Err(LlmError::Gguf(
                "prompt tokenized to an empty sequence".into(),
            ));
        }
        if prompt_tokens.len() > self.config.max_seq_len {
            return Err(LlmError::Gguf(format!(
                "prompt has {} tokens but model context is {}",
                prompt_tokens.len(),
                self.config.max_seq_len
            )));
        }

        // End padding is semantics-preserving for real prompt rows under causal attention,
        // while making M a multiple of four for the current Rocket MatMul contract.
        let padded_tokens = prompt_tokens.len().div_ceil(4) * 4;
        let mut hidden = vec![f16::ZERO; padded_tokens * self.config.hidden_size];
        for (row, token) in prompt_tokens.iter().copied().enumerate() {
            let token = token as usize;
            if token >= self.config.vocab_size {
                return Err(LlmError::Gguf(format!(
                    "token {token} is outside vocabulary {}",
                    self.config.vocab_size
                )));
            }
            let src = token * self.config.hidden_size;
            let dst = row * self.config.hidden_size;
            hidden[dst..dst + self.config.hidden_size]
                .copy_from_slice(&self.embedding[src..src + self.config.hidden_size]);
        }

        let mut total_stats = TransformerBlockStats::default();
        for layer_index in 0..self.layers.len() {
            let (block_config, weights) = self.block(layer_index)?;
            let block = TransformerBlock::prepare(block_config, weights, backend.as_deref())?;
            let (next, stats) =
                block.run_prefill(backend.as_deref_mut(), &hidden, padded_tokens, 0)?;
            total_stats.npu_linears += stats.npu_linears;
            total_stats.cpu_linears += stats.cpu_linears;
            hidden = next;
        }

        let normalized = rms_norm(
            &hidden,
            padded_tokens,
            self.config.hidden_size,
            &self.final_norm,
            self.final_norm_eps,
        )?;
        let last = (prompt_tokens.len() - 1) * self.config.hidden_size;
        let last_hidden = &normalized[last..last + self.config.hidden_size];
        let output_weights = linear_weights(&self.output)?;
        let lm_head = HybridLinear::prepare(
            None,
            self.config.vocab_size,
            self.config.hidden_size,
            output_weights,
        )?;
        let (logits, execution) = lm_head.run(None, last_hidden, 1)?;
        debug_assert_eq!(execution, LinearExecution::Cpu);
        let token_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.to_f32().total_cmp(&right.to_f32()))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| LlmError::Gguf("LM head produced no logits".into()))?;
        let text = self.decode_token(token_id)?;
        Ok(FirstTokenResult {
            prompt_tokens,
            token_id,
            text,
            npu_linears: total_stats.npu_linears,
            cpu_linears: total_stats.cpu_linears + 1,
            padded_tokens,
        })
    }

    fn block(
        &self,
        layer_index: usize,
    ) -> Result<(TransformerBlockConfig, TransformerBlockWeights), LlmError> {
        let layer = self
            .layers
            .get(layer_index)
            .ok_or_else(|| LlmError::Gguf(format!("missing transformer layer {layer_index}")))?;
        if layer.use_parallel_residual
            || layer.post_attn_norm.is_some()
            || layer.post_ffn_norm.is_some()
        {
            return Err(LlmError::Gguf(format!(
                "layer {layer_index} uses an unsupported residual/post-norm variant"
            )));
        }
        let attention = layer.attention().ok_or_else(|| {
            LlmError::Gguf(format!("layer {layer_index} is not full dense attention"))
        })?;
        let ffn = layer.ffn().ok_or_else(|| {
            LlmError::Gguf(format!("layer {layer_index} is not a dense gated FFN"))
        })?;
        if attention.key_length != self.config.head_dim
            || attention.value_length != self.config.head_dim
            || attention.rope_dims != self.config.head_dim
            || attention.has_attention_gate
            || attention.q_norm.is_some()
            || attention.k_norm.is_some()
            || attention.attn_logit_softcap != 0.0
        {
            return Err(LlmError::Gguf(format!(
                "layer {layer_index} uses an attention variant outside the first Llama path"
            )));
        }
        for linear in [
            &attention.wq,
            &attention.wk,
            &attention.wv,
            &attention.wo,
            &ffn.w_gate,
            &ffn.w_up,
            &ffn.w_down,
        ] {
            if linear.bias.is_some() {
                return Err(LlmError::Gguf(format!(
                    "layer {layer_index} has projection bias; first Llama path is biasless"
                )));
            }
        }
        let attn_norm = rms_norm_layer(&layer.attn_norm, layer_index, "attention")?;
        let ffn_norm = rms_norm_layer(&layer.ffn_norm, layer_index, "ffn")?;
        if (attn_norm.eps - ffn_norm.eps).abs() > f32::EPSILON {
            return Err(LlmError::Gguf(format!(
                "layer {layer_index} uses different RMSNorm epsilons"
            )));
        }
        let config = TransformerBlockConfig {
            hidden_size: self.config.hidden_size,
            query_heads: attention.num_heads,
            kv_heads: attention.num_kv_heads,
            head_dim: attention.head_dim,
            intermediate_size: ffn.intermediate_size,
            rms_norm_eps: attn_norm.eps,
            rope_theta: self.config.rope_config.freq_base,
            rope_style: if attention.use_neox_rope {
                RopeStyle::NeoX
            } else {
                RopeStyle::Normal
            },
        };
        Ok((
            config,
            TransformerBlockWeights {
                attention_norm: tensor_to_f16(&attn_norm.weight)?,
                q_proj: linear_weights(&attention.wq)?,
                k_proj: linear_weights(&attention.wk)?,
                v_proj: linear_weights(&attention.wv)?,
                o_proj: linear_weights(&attention.wo)?,
                ffn_norm: tensor_to_f16(&ffn_norm.weight)?,
                gate_proj: linear_weights(&ffn.w_gate)?,
                up_proj: linear_weights(&ffn.w_up)?,
                down_proj: linear_weights(&ffn.w_down)?,
            },
        ))
    }
}

fn validate_model_config(config: &ModelConfig) -> Result<(), LlmError> {
    if config.num_layers == 0
        || config.hidden_size == 0
        || config.num_heads == 0
        || config.num_kv_heads == 0
        || config.intermediate_size == 0
    {
        return Err(LlmError::Gguf(
            "model has zero-sized core dimensions".into(),
        ));
    }
    if config.uses_layer_norm
        || !config.has_ffn_gate
        || config.num_experts != 0
        || config.has_combined_qkv
        || config.attention_bias
        || config.mlp_bias
    {
        return Err(LlmError::Gguf(
            "first GGUF path requires a dense biasless RMSNorm Llama-family model".into(),
        ));
    }
    if config.key_length != config.head_dim || config.value_length != config.head_dim {
        return Err(LlmError::Gguf(
            "first GGUF path requires key/value length == head_dim".into(),
        ));
    }
    if config.rope_config.scaling_type != RopeScalingType::None
        || (config.rope_config.freq_scale - 1.0).abs() > f32::EPSILON
    {
        return Err(LlmError::Gguf(
            "scaled RoPE is not supported by the first GGUF correctness path".into(),
        ));
    }
    Ok(())
}

fn rms_norm_layer<'a>(
    norm: &'a NormLayer,
    layer_index: usize,
    kind: &str,
) -> Result<&'a llama_gguf::model::layers::RMSNorm, LlmError> {
    norm.as_rms()
        .ok_or_else(|| LlmError::Gguf(format!("layer {layer_index} {kind} norm is not RMSNorm")))
}

fn linear_weights(linear: &GgufLinear) -> Result<Vec<f16>, LlmError> {
    if linear.bias.is_some() {
        return Err(LlmError::Gguf(
            "biased Linear is not supported by the first GGUF path".into(),
        ));
    }
    if linear.weight.shape() != [linear.in_features, linear.out_features] {
        return Err(LlmError::Gguf(format!(
            "Linear shape {:?} does not match [{}, {}]",
            linear.weight.shape(),
            linear.in_features,
            linear.out_features
        )));
    }
    // GGUF's first dimension is contiguous: raw storage is already [out][in],
    // exactly the row-major [out_features, in_features] layout HybridLinear expects.
    tensor_to_f16(&linear.weight)
}

fn tensor_to_f16(tensor: &GgufTensor) -> Result<Vec<f16>, LlmError> {
    let count = tensor.numel();
    match tensor.dtype() {
        DType::F32 => Ok(tensor
            .as_f32()
            .map_err(frontend_error)?
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect()),
        DType::F16 => {
            if tensor.data().len() != count * 2 {
                return Err(LlmError::Gguf("bad F16 tensor byte length".into()));
            }
            Ok(tensor
                .data()
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect())
        }
        DType::BF16 => {
            if tensor.data().len() != count * 2 {
                return Err(LlmError::Gguf("bad BF16 tensor byte length".into()));
            }
            Ok(tensor
                .data()
                .chunks_exact(2)
                .map(|bytes| {
                    let value = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
                    f16::from_f32(value.to_f32())
                })
                .collect())
        }
        DType::Q4_0 => dequant_blocks::<BlockQ4_0, 32>(tensor, dequantize_q4_0),
        DType::Q4_1 => dequant_blocks::<BlockQ4_1, 32>(tensor, dequantize_q4_1),
        DType::Q5_0 => dequant_blocks::<BlockQ5_0, 32>(tensor, dequantize_q5_0),
        DType::Q5_1 => dequant_blocks::<BlockQ5_1, 32>(tensor, dequantize_q5_1),
        DType::Q8_0 => dequant_blocks::<BlockQ8_0, 32>(tensor, dequantize_q8_0),
        DType::Q8_1 => dequant_blocks::<BlockQ8_1, 32>(tensor, dequantize_q8_1),
        DType::Q2K => dequant_blocks::<BlockQ2K, 256>(tensor, dequantize_q2_k),
        DType::Q3K => dequant_blocks::<BlockQ3K, 256>(tensor, dequantize_q3_k),
        DType::Q4K => dequant_blocks::<BlockQ4K, 256>(tensor, dequantize_q4_k),
        DType::Q5K => dequant_blocks::<BlockQ5K, 256>(tensor, dequantize_q5_k),
        DType::Q6K => dequant_blocks::<BlockQ6K, 256>(tensor, dequantize_q6_k),
        DType::Q8K => dequant_blocks::<BlockQ8K, 256>(tensor, dequantize_q8_k),
        dtype => Err(LlmError::Gguf(format!(
            "GGUF tensor dtype {} is not supported by the first RockNPU loader",
            dtype.name()
        ))),
    }
}

fn dequant_blocks<B: Pod, const BLOCK: usize>(
    tensor: &GgufTensor,
    dequantize: fn(&B, &mut [f32; BLOCK]),
) -> Result<Vec<f16>, LlmError> {
    let block_bytes = size_of::<B>();
    if !tensor.data().len().is_multiple_of(block_bytes) {
        return Err(LlmError::Gguf(format!(
            "quantized tensor byte length {} is not a multiple of block size {block_bytes}",
            tensor.data().len()
        )));
    }
    let mut output = Vec::with_capacity(tensor.numel());
    let mut values = [0.0f32; BLOCK];
    for bytes in tensor.data().chunks_exact(block_bytes) {
        let block: B = pod_read_unaligned(bytes);
        dequantize(&block, &mut values);
        let remaining = tensor.numel() - output.len();
        output.extend(
            values[..remaining.min(BLOCK)]
                .iter()
                .copied()
                .map(f16::from_f32),
        );
    }
    if output.len() != tensor.numel() {
        return Err(LlmError::Gguf(format!(
            "dequantized {} values for tensor with {} elements",
            output.len(),
            tensor.numel()
        )));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_tensor_conversion_preserves_bits() {
        let values = [f16::from_f32(1.25), f16::from_f32(-3.5)];
        let data = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let tensor = GgufTensor::new(data, vec![2], DType::F16).unwrap();
        assert_eq!(tensor_to_f16(&tensor).unwrap(), values);
    }

    #[test]
    fn f32_tensor_conversion_rounds_to_f16() {
        let tensor = GgufTensor::from_f32(&[1.0, -2.0, 0.125], vec![3]).unwrap();
        let got = tensor_to_f16(&tensor).unwrap();
        assert_eq!(
            got,
            vec![f16::ONE, f16::from_f32(-2.0), f16::from_f32(0.125)]
        );
    }
}
