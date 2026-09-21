//! Candle frontend adapter for RockNPU.
//!
//! Candle owns eager tensor/module semantics. RockNPU owns the shared userspace
//! execution primitives. No Candle type is used by RockNPU core crates.

use candle_core::{DType as CandleDType, Error as CandleError, Module, Result, Tensor};
use half::f16;
use rocket_runtime::RocketDevice;
use rocknpu_ops::{
    ExecutionTarget, MatmulOutput, MatmulPrecision, MatmulSpec, SingleNpuBackend, execute_cpu,
};
use rocknpu_tensor::Matrix;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

fn invalid(message: impl Into<String>) -> CandleError {
    CandleError::Msg(format!("candle-rocknpu: {}", message.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearTarget {
    Npu,
    Cpu,
}

enum Command {
    Run {
        rows: usize,
        input: Vec<f16>,
        reply: mpsc::Sender<std::result::Result<Vec<f32>, String>>,
    },
    Shutdown,
}

/// Prepared Candle-compatible linear module backed by shared RockNPU ops.
///
/// Candle linear weights use [out_features, in_features]. Static weights and
/// bias are rounded once to F16, matching RockNPU's current high-precision
/// prepared MatMul contract. F32 Candle inputs may have any rank >= 1; all
/// leading dimensions are flattened into M and restored on output.
pub struct RockNpuLinear {
    target: LinearTarget,
    in_features: usize,
    out_features: usize,
    tx: mpsc::Sender<Command>,
    worker: Option<JoinHandle<()>>,
}

impl RockNpuLinear {
    /// Construct an NPU-backed Candle linear module.
    pub fn new(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        Self::with_target(weight, bias, LinearTarget::Npu)
    }

    /// Construct a CPU-reference module using the same RockNPU FP16 contract.
    pub fn cpu(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        Self::with_target(weight, bias, LinearTarget::Cpu)
    }

    pub fn with_target(
        weight: &Tensor,
        bias: Option<&Tensor>,
        target: LinearTarget,
    ) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(invalid(format!(
                "linear weight must be rank 2 [out,in], got {:?}",
                weight.dims()
            )));
        }
        let out_features = weight.dims()[0];
        let in_features = weight.dims()[1];
        if in_features == 0 || out_features == 0 {
            return Err(invalid("linear features must be nonzero"));
        }

        let weight_rows = weight.to_dtype(CandleDType::F32)?.to_vec2::<f32>()?;
        let weights: Vec<f16> = weight_rows
            .iter()
            .flatten()
            .copied()
            .map(f16::from_f32)
            .collect();

        let bias = match bias {
            Some(bias) => {
                if bias.rank() != 1 || bias.dims()[0] != out_features {
                    return Err(invalid(format!(
                        "linear bias must have shape [{}], got {:?}",
                        out_features,
                        bias.dims()
                    )));
                }
                Some(
                    bias.to_dtype(CandleDType::F32)?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(f16::from_f32)
                        .collect::<Vec<_>>(),
                )
            }
            None => None,
        };

        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("candle-rocknpu-linear".into())
            .spawn(move || match target {
                LinearTarget::Npu => {
                    npu_worker(in_features, out_features, weights, bias, ready_tx, rx);
                }
                LinearTarget::Cpu => {
                    cpu_worker(in_features, out_features, weights, bias, ready_tx, rx);
                }
            })
            .map_err(|e| invalid(format!("spawn linear worker: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                target,
                in_features,
                out_features,
                tx,
                worker: Some(worker),
            }),
            Ok(Err(message)) => {
                let _ = worker.join();
                Err(invalid(message))
            }
            Err(_) => {
                let _ = worker.join();
                Err(invalid("linear worker exited during initialization"))
            }
        }
    }

    pub const fn target(&self) -> LinearTarget {
        self.target
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn forward_rocknpu(&self, input: &Tensor) -> Result<Tensor> {
        if input.dtype() != CandleDType::F32 {
            return Err(CandleError::UnexpectedDType {
                msg: "RockNpuLinear currently requires F32 input",
                expected: CandleDType::F32,
                got: input.dtype(),
            });
        }
        let dims = input.dims();
        if dims.is_empty() || *dims.last().unwrap() != self.in_features {
            return Err(invalid(format!(
                "linear input must end in {}, got {:?}",
                self.in_features, dims
            )));
        }

        let leading = &dims[..dims.len() - 1];
        let rows = leading
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| invalid("linear input element count overflow"))?;
        if rows == 0 {
            return Err(invalid("zero-sized Candle tensors are not supported"));
        }

        let device = input.device().clone();
        let values = input
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Run {
                rows,
                input: values,
                reply: reply_tx,
            })
            .map_err(|_| invalid("linear worker is not running"))?;
        let values = reply_rx
            .recv()
            .map_err(|_| invalid("linear worker exited during run"))?
            .map_err(invalid)?;

        let mut output_shape = leading.to_vec();
        output_shape.push(self.out_features);
        Tensor::from_vec(values, output_shape, &candle_core::Device::Cpu)?.to_device(&device)
    }
}

impl Module for RockNpuLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_rocknpu(input)
    }
}

impl Drop for RockNpuLinear {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn finish_output(
    values: &[f16],
    rows: usize,
    out_features: usize,
    bias: Option<&[f16]>,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(rows * out_features);
    for row in 0..rows {
        for col in 0..out_features {
            let mut value = values[row * out_features + col];
            if let Some(bias) = bias {
                value = f16::from_f32(value.to_f32() + bias[col].to_f32());
            }
            output.push(value.to_f32());
        }
    }
    output
}

fn cpu_worker(
    in_features: usize,
    out_features: usize,
    weights: Vec<f16>,
    bias: Option<Vec<f16>>,
    ready: mpsc::Sender<std::result::Result<(), String>>,
    rx: mpsc::Receiver<Command>,
) {
    let weights = match Matrix::from_vec(out_features, in_features, weights) {
        Ok(weights) => weights,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }

    while let Ok(command) = rx.recv() {
        match command {
            Command::Shutdown => break,
            Command::Run { rows, input, reply } => {
                let result = (|| {
                    let input = Matrix::from_vec(rows, in_features, input)
                        .map_err(|error| error.to_string())?;
                    let spec = MatmulSpec::new(
                        rows,
                        in_features,
                        out_features,
                        MatmulPrecision::Fp16Fast,
                        ExecutionTarget::Cpu,
                    );
                    let output =
                        execute_cpu(spec, &input, &weights).map_err(|error| error.to_string())?;
                    let MatmulOutput::F16(output) = output else {
                        return Err("FP16 linear unexpectedly returned F32 output".into());
                    };
                    Ok(finish_output(
                        output.values(),
                        rows,
                        out_features,
                        bias.as_deref(),
                    ))
                })();
                let _ = reply.send(result);
            }
        }
    }
}

fn npu_worker(
    in_features: usize,
    out_features: usize,
    weights: Vec<f16>,
    bias: Option<Vec<f16>>,
    ready: mpsc::Sender<std::result::Result<(), String>>,
    rx: mpsc::Receiver<Command>,
) {
    if in_features % 32 != 0 || out_features % 16 != 0 {
        let _ = ready.send(Err(
            "NPU linear currently requires in_features%32==0 and out_features%16==0".into(),
        ));
        return;
    }

    let device = match RocketDevice::open() {
        Ok(device) => device,
        Err(error) => {
            let _ = ready.send(Err(format!("open NPU: {error}")));
            return;
        }
    };
    let mut backend = match SingleNpuBackend::new(&device) {
        Ok(backend) => backend,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let weights = match Matrix::from_vec(out_features, in_features, weights) {
        Ok(weights) => weights,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let prepared = match backend.prepare_fp16_compatible_m(&weights) {
        Ok(prepared) => prepared,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }

    while let Ok(command) = rx.recv() {
        match command {
            Command::Shutdown => break,
            Command::Run { rows, input, reply } => {
                let result = (|| {
                    let padded_rows = rows
                        .checked_add(3)
                        .map(|rows| rows / 4 * 4)
                        .ok_or_else(|| "linear M padding overflow".to_string())?;
                    let mut padded = vec![f16::ZERO; padded_rows * in_features];
                    padded[..rows * in_features].copy_from_slice(&input);
                    let input = Matrix::from_vec(padded_rows, in_features, padded)
                        .map_err(|error| error.to_string())?;
                    let output = backend
                        .execute_prepared_fp16_compatible_m(&prepared, &input)
                        .map_err(|error| error.to_string())?;
                    Ok(finish_output(
                        output.values(),
                        rows,
                        out_features,
                        bias.as_deref(),
                    ))
                })();
                let _ = reply.send(result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.into_iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn rounded_candle_reference(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let input_q = input
            .to_dtype(CandleDType::F16)
            .unwrap()
            .to_dtype(CandleDType::F32)
            .unwrap();
        let weight_q = weight
            .to_dtype(CandleDType::F16)
            .unwrap()
            .to_dtype(CandleDType::F32)
            .unwrap();
        let mut output = input_q.matmul(&weight_q.t().unwrap()).unwrap();
        output = output
            .to_dtype(CandleDType::F16)
            .unwrap()
            .to_dtype(CandleDType::F32)
            .unwrap();
        if let Some(bias) = bias {
            let bias_q = bias
                .to_dtype(CandleDType::F16)
                .unwrap()
                .to_dtype(CandleDType::F32)
                .unwrap();
            output = output
                .broadcast_add(&bias_q)
                .unwrap()
                .to_dtype(CandleDType::F16)
                .unwrap()
                .to_dtype(CandleDType::F32)
                .unwrap();
        }
        output
    }

    #[test]
    fn cpu_linear_matches_candle_with_fp16_contract() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            (0..24)
                .map(|value| (value as f32 - 9.0) / 16.0)
                .collect::<Vec<_>>(),
            (6, 4),
            &device,
        )
        .unwrap();
        let bias =
            Tensor::from_vec(vec![0.25f32, -0.5, 0.75, -1.0, 0.125, 0.625], 6, &device).unwrap();
        let input = Tensor::from_vec(
            (0..24)
                .map(|value| (value as f32 - 7.0) / 10.0)
                .collect::<Vec<_>>(),
            (6, 4),
            &device,
        )
        .unwrap();

        let linear = RockNpuLinear::cpu(&weight, Some(&bias)).unwrap();
        let got = linear.forward(&input).unwrap();
        let expected = rounded_candle_reference(&input, &weight, Some(&bias));

        assert_eq!(got.dims(), &[6, 6]);
        assert!(max_abs(&got, &expected) <= 0.002);
    }

    #[test]
    fn cpu_linear_restores_leading_candle_dimensions() {
        let device = Device::Cpu;
        let weight = Tensor::ones((8, 4), CandleDType::F32, &device).unwrap();
        let input = Tensor::ones((2, 3, 4), CandleDType::F32, &device).unwrap();
        let linear = RockNpuLinear::cpu(&weight, None).unwrap();
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 3, 8]);
    }

    #[test]
    #[ignore = "requires an RK3588 NPU at /dev/accel/accel0"]
    fn npu_linear_matches_candle_cpu() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            (0..1024)
                .map(|value| ((value % 29) as f32 - 14.0) / 64.0)
                .collect::<Vec<_>>(),
            (32, 32),
            &device,
        )
        .unwrap();
        let bias = Tensor::from_vec(
            (0..32)
                .map(|value| (value as f32 - 12.0) / 128.0)
                .collect::<Vec<_>>(),
            32,
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(
            (0..96)
                .map(|value| ((value % 17) as f32 - 8.0) / 16.0)
                .collect::<Vec<_>>(),
            (3, 32),
            &device,
        )
        .unwrap();

        let linear = RockNpuLinear::new(&weight, Some(&bias)).unwrap();
        let got = linear.forward_rocknpu(&input).unwrap();
        let expected = rounded_candle_reference(&input, &weight, Some(&bias));
        let error = max_abs(&got, &expected);
        eprintln!("CANDLE ROCKNPU NPU linear max_abs={error:.6}");
        assert_eq!(got.dims(), &[3, 32]);
        assert!(error <= 0.02);
    }
}
