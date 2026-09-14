use rocket_runtime::RocketDevice;
use rocknpu_llm::GgufLlama;
use rocknpu_ops::SingleNpuBackend;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: llm_gguf <model.gguf> <prompt> [cpu|npu]".into());
    }

    let model_path = &args[0];
    let prompt = &args[1];
    let target = args.get(2).map(String::as_str).unwrap_or("npu");
    let model = GgufLlama::load(model_path)?;
    eprintln!("LLM GGUF model={:?}", model.info());

    let result = match target {
        "cpu" => model.first_token(None, prompt, true)?,
        "npu" => {
            let device = RocketDevice::open()?;
            let mut backend = SingleNpuBackend::new(&device)?;
            model.first_token(Some(&mut backend), prompt, true)?
        }
        other => return Err(format!("unknown target {other:?}; expected cpu or npu").into()),
    };

    println!(
        "LLM GGUF FIRST TOKEN PASS target={} prompt_tokens={} padded_tokens={} token_id={} text={:?} npu_linears={} cpu_linears={}",
        target,
        result.prompt_tokens.len(),
        result.padded_tokens,
        result.token_id,
        result.text,
        result.npu_linears,
        result.cpu_linears,
    );
    Ok(())
}
