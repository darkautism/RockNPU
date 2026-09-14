use rocknpu::{Session, SessionOptions, SessionTarget, Tensor};
use std::env;
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "RockNPU developer-preview CLI\n\nUSAGE:\n    rocknpu run <model.onnx> --input <input.npy> --output <output.npy> [--target npu|cpu] [--device <path>]\n\nOPTIONS:\n    --target <npu|cpu>   Execution target (default: npu)\n    --device <path>      Rocket device path (default: /dev/accel/accel0; NPU only)\n    -h, --help           Show this help\n    -V, --version        Show version\n";

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunArgs {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    target: SessionTarget,
    device: Option<PathBuf>,
}

fn main() {
    if let Err(error) = real_main(env::args().skip(1)) {
        eprintln!("rocknpu: {error}");
        std::process::exit(2);
    }
}

fn real_main<I>(args: I) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let args: Vec<String> = args.into_iter().collect();
    if args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    if args.len() == 1 && (args[0] == "-V" || args[0] == "--version") {
        println!("rocknpu {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let run = parse_run_args(&args)?;
    execute_run(run)
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    if args.first().map(String::as_str) != Some("run") {
        return Err(format!("expected `run` subcommand\n\n{USAGE}"));
    }
    let model = args
        .get(1)
        .filter(|arg| !arg.starts_with('-'))
        .ok_or_else(|| format!("missing model path\n\n{USAGE}"))?;
    let mut input = None;
    let mut output = None;
    let mut target = SessionTarget::Npu;
    let mut device = None;
    let mut index = 2usize;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input = Some(PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--input requires a path".to_string())?,
                ));
            }
            "--output" => {
                index += 1;
                output = Some(PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--output requires a path".to_string())?,
                ));
            }
            "--target" => {
                index += 1;
                target = match args.get(index).map(String::as_str) {
                    Some("npu") => SessionTarget::Npu,
                    Some("cpu") => SessionTarget::Cpu,
                    Some(other) => {
                        return Err(format!("invalid --target {other:?}; use npu or cpu"));
                    }
                    None => return Err("--target requires npu or cpu".into()),
                };
            }
            "--device" => {
                index += 1;
                device = Some(PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| "--device requires a path".to_string())?,
                ));
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    if target == SessionTarget::Cpu && device.is_some() {
        return Err("--device is only valid with --target npu".into());
    }
    Ok(RunArgs {
        model: PathBuf::from(model),
        input: input.ok_or_else(|| "missing --input <input.npy>".to_string())?,
        output: output.ok_or_else(|| "missing --output <output.npy>".to_string())?,
        target,
        device,
    })
}

fn execute_run(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let mut options = match args.target {
        SessionTarget::Npu => SessionOptions::npu(),
        SessionTarget::Cpu => SessionOptions::cpu(),
    };
    if let Some(device) = &args.device {
        options = options.with_device_path(device.clone());
    }
    let session = Session::load_with_options(&args.model, options)?;
    let input = Tensor::load_npy(&args.input)?;
    let output = session.run(input)?;
    output.tensor().save_npy(&args.output)?;
    let prepare = session.prepare_stats();
    let stats = output.stats();
    println!(
        "rocknpu run PASS target={:?} input={} {:?} output={} {:?} resident_bytes={} prepare_ms={:.3} run_ms={:.3} npu_conv={} npu_dense={}",
        args.target,
        session.input().name(),
        session.input().shape(),
        output.name(),
        output.tensor().shape(),
        prepare.resident_weight_bytes,
        prepare.prepare_ns as f64 / 1_000_000.0,
        stats.total_ns as f64 / 1_000_000.0,
        stats.npu_conv_nodes,
        stats.npu_dense_nodes,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn parses_default_npu_run() {
        let args = strings(&[
            "run",
            "model.onnx",
            "--input",
            "input.npy",
            "--output",
            "output.npy",
        ]);
        let parsed = parse_run_args(&args).unwrap();
        assert_eq!(parsed.target, SessionTarget::Npu);
        assert_eq!(parsed.model, PathBuf::from("model.onnx"));
        assert_eq!(parsed.input, PathBuf::from("input.npy"));
        assert_eq!(parsed.output, PathBuf::from("output.npy"));
        assert_eq!(parsed.device, None);
    }

    #[test]
    fn parses_cpu_run_and_rejects_device() {
        let args = strings(&[
            "run",
            "model.onnx",
            "--input",
            "input.npy",
            "--output",
            "output.npy",
            "--target",
            "cpu",
        ]);
        assert_eq!(parse_run_args(&args).unwrap().target, SessionTarget::Cpu);

        let args = strings(&[
            "run",
            "model.onnx",
            "--input",
            "input.npy",
            "--output",
            "output.npy",
            "--target",
            "cpu",
            "--device",
            "/dev/accel/accel0",
        ]);
        assert!(parse_run_args(&args).is_err());
    }
}
