use half::f16;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrError {
    InvalidTensor(String),
    InvalidGraph(String),
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTensor(message) => write!(f, "invalid tensor: {message}"),
            Self::InvalidGraph(message) => write!(f, "invalid graph: {message}"),
        }
    }
}

impl std::error::Error for IrError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    name: String,
    dtype: DType,
    shape: Vec<Option<usize>>,
}

impl TensorSpec {
    pub fn new(
        name: impl Into<String>,
        dtype: DType,
        shape: Vec<Option<usize>>,
    ) -> Result<Self, IrError> {
        let name = name.into();
        if name.is_empty() {
            return Err(IrError::InvalidGraph("tensor spec has no name".into()));
        }
        if shape.is_empty() {
            return Err(IrError::InvalidGraph(format!(
                "{name} scalar tensors are not supported yet"
            )));
        }
        if shape.iter().any(|dim| matches!(dim, Some(0))) {
            return Err(IrError::InvalidGraph(format!(
                "{name} has a zero-sized static dimension"
            )));
        }
        Ok(Self { name, dtype, shape })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[Option<usize>] {
        &self.shape
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct F16Tensor {
    pub dims: Vec<usize>,
    pub values: Vec<f16>,
}

impl F16Tensor {
    pub fn from_vec(dims: Vec<usize>, values: Vec<f16>) -> Result<Self, IrError> {
        if dims.is_empty() || dims.contains(&0) {
            return Err(IrError::InvalidTensor("tensor dims must be nonzero".into()));
        }
        let expected = dims
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| IrError::InvalidTensor("tensor element count overflow".into()))?;
        if expected != values.len() {
            return Err(IrError::InvalidTensor(format!(
                "tensor value count {} != {expected}",
                values.len()
            )));
        }
        Ok(Self { dims, values })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn values(&self) -> &[f16] {
        &self.values
    }
}

#[derive(Debug, Clone)]
pub enum ConstantTensor {
    F16(F16Tensor),
    I64 { dims: Vec<usize>, values: Vec<i64> },
}

impl ConstantTensor {
    pub fn f16(&self) -> Result<&F16Tensor, IrError> {
        match self {
            Self::F16(value) => Ok(value),
            _ => Err(IrError::InvalidTensor("expected F16 constant".into())),
        }
    }

    pub fn i64_values(&self) -> Result<&[i64], IrError> {
        match self {
            Self::I64 { values, .. } => Ok(values),
            _ => Err(IrError::InvalidTensor("expected I64 constant".into())),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F16(value) => value.dims(),
            Self::I64 { dims, .. } => dims,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoPad {
    NotSet,
    SameUpper,
}

#[derive(Debug, Clone)]
pub enum Node {
    Conv {
        input: String,
        weight: String,
        bias: Option<String>,
        output: String,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        auto_pad: AutoPad,
    },
    MaxPool {
        input: String,
        output: String,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        auto_pad: AutoPad,
    },
    Add {
        lhs: String,
        rhs: String,
        output: String,
    },
    Relu {
        input: String,
        output: String,
    },
    Reshape {
        input: String,
        shape: String,
        output: String,
        allowzero: bool,
    },
    MatMul {
        lhs: String,
        rhs: String,
        output: String,
    },
    Gemm {
        lhs: String,
        rhs: String,
        bias: String,
        output: String,
    },
}

impl Node {
    pub fn output(&self) -> &str {
        match self {
            Self::Conv { output, .. }
            | Self::MaxPool { output, .. }
            | Self::Add { output, .. }
            | Self::Relu { output, .. }
            | Self::Reshape { output, .. }
            | Self::MatMul { output, .. }
            | Self::Gemm { output, .. } => output,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    input: TensorSpec,
    output: TensorSpec,
    nodes: Vec<Node>,
    constants: HashMap<String, ConstantTensor>,
}

impl Graph {
    pub fn new(
        input: TensorSpec,
        output: TensorSpec,
        nodes: Vec<Node>,
        constants: HashMap<String, ConstantTensor>,
    ) -> Result<Self, IrError> {
        if input.name() == output.name() && !nodes.is_empty() {
            return Err(IrError::InvalidGraph(
                "graph input and output names must differ when nodes are present".into(),
            ));
        }
        Ok(Self {
            input,
            output,
            nodes,
            constants,
        })
    }

    pub fn input(&self) -> &TensorSpec {
        &self.input
    }

    pub fn output(&self) -> &TensorSpec {
        &self.output
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn constants(&self) -> &HashMap<String, ConstantTensor> {
        &self.constants
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_spec_accepts_dynamic_dimensions() {
        let spec =
            TensorSpec::new("input", DType::F32, vec![None, Some(3), Some(32), Some(32)]).unwrap();
        assert_eq!(spec.shape(), &[None, Some(3), Some(32), Some(32)]);
    }

    #[test]
    fn f16_tensor_checks_element_count() {
        assert!(F16Tensor::from_vec(vec![2, 2], vec![f16::ZERO; 4]).is_ok());
        assert!(F16Tensor::from_vec(vec![2, 2], vec![f16::ZERO; 3]).is_err());
    }
}
