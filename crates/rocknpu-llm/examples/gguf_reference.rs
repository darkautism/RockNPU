use llama_gguf::{Backend, GgufFile, Model, ModelLoader, Tokenizer, default_backend};
use std::env;
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: gguf_reference <model.gguf> <prompt>".into());
    }

    let model_path = &args[0];
    let prompt = &args[1];
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let tokens = tokenizer.encode(prompt, true)?;
    drop(gguf);

    let model = ModelLoader::load(model_path)?.build_model()?;
    let backend: Arc<dyn Backend> = Arc::from(default_backend());
    let mut context = model.create_context(backend);
    let logits = Model::forward(&model, &tokens, &mut context)?;
    let values = logits.as_f32()?;
    let (token_id, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or("reference model returned no logits")?;
    let text = tokenizer.decode_token(token_id as u32)?;

    println!(
        "LLAMA-GGUF REFERENCE FIRST TOKEN prompt_tokens={} token_id={} text={:?}",
        tokens.len(),
        token_id,
        text,
    );
    Ok(())
}
