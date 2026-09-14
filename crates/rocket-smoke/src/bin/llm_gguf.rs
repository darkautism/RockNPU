use rocket_runtime::RocketDevice;
use rocknpu_llm::GgufLlama;
use rocknpu_ops::SingleNpuBackend;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 4 {
        return Err("usage: llm_gguf <model.gguf> <prompt> [cpu|npu] [max_new_tokens]".into());
    }

    let model_path = &args[0];
    let prompt = &args[1];
    let target = args.get(2).map(String::as_str).unwrap_or("npu");
    let max_new_tokens = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let model = GgufLlama::load(model_path)?;
    eprintln!("LLM GGUF model={:?}", model.info());

    let result = match target {
        "cpu" => model.generate_greedy(None, prompt, true, max_new_tokens)?,
        "npu" => {
            let device = RocketDevice::open()?;
            let mut backend = SingleNpuBackend::new(&device)?;
            model.generate_greedy(Some(&mut backend), prompt, true, max_new_tokens)?
        }
        other => return Err(format!("unknown target {other:?}; expected cpu or npu").into()),
    };

    println!(
        "LLM GGUF GENERATION PASS target={} prompt_tokens={} padded_tokens={} generated_tokens={:?} text={:?} prefill_npu={} prefill_cpu={} decode_npu={} decode_cpu={} lm_head_cpu={}",
        target,
        result.prompt_tokens.len(),
        result.padded_tokens,
        result.generated_tokens,
        result.text,
        result.prefill_npu_linears,
        result.prefill_cpu_linears,
        result.decode_npu_linears,
        result.decode_cpu_linears,
        result.lm_head_cpu_linears,
    );
    Ok(())
}
