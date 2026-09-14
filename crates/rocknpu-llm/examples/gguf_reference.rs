use llama_gguf::{Backend, GgufFile, Model, ModelLoader, Tokenizer, default_backend};
use std::env;
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: gguf_reference <model.gguf> <prompt> [max_new_tokens]".into());
    }

    let model_path = &args[0];
    let prompt = &args[1];
    let max_new_tokens = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if max_new_tokens == 0 {
        return Err("max_new_tokens must be greater than zero".into());
    }

    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    drop(gguf);

    let model = ModelLoader::load(model_path)?.build_model()?;
    let backend: Arc<dyn Backend> = Arc::from(default_backend());
    let mut context = model.create_context(backend);
    let mut input = prompt_tokens.clone();
    let mut generated = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = Model::forward(&model, &input, &mut context)?;
        let values = logits.as_f32()?;
        let token_id = values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or("reference model returned no logits")?;
        if env::var_os("ROCKNPU_REFERENCE_TOP").is_some() {
            let mut ids: Vec<usize> = (0..values.len()).collect();
            ids.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
            let top = ids.iter().take(5).copied().collect::<Vec<_>>();
            let margin = values[top[0]] - values[top[1]];
            eprintln!(
                "step={} top={:?} margin={:+.9}",
                generated.len() + 1,
                top.iter().map(|&id| (id, values[id])).collect::<Vec<_>>(),
                margin,
            );
        }
        generated.push(token_id);
        if token_id == tokenizer.special_tokens.eos_token_id {
            break;
        }
        input.clear();
        input.push(token_id);
    }
    let text = tokenizer.decode(&generated)?;

    println!(
        "LLAMA-GGUF REFERENCE GENERATION prompt_tokens={} generated_tokens={:?} text={:?}",
        prompt_tokens.len(),
        generated,
        text,
    );
    Ok(())
}
