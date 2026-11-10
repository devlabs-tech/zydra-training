import argparse
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from modeling_zydra import ZydraForCausalLM


def generate_text(
    prompt,
    tokenizer,
    model,
    device,
    max_new_tokens,
    free_form,
    temperature,
    top_p,
):
    generation_prompt = prompt if free_form else f"Pertanyaan: {prompt}\nJawaban:"
    token_ids = tokenizer.encode(generation_prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        token_ids.insert(0, tokenizer.bos_token_id)

    context_limit = model.config.max_position_embeddings
    if len(token_ids) >= context_limit:
        raise ValueError(f"Prompt must be shorter than the model context limit ({context_limit} tokens).")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=min(max_new_tokens, context_limit - len(token_ids)),
            do_sample=free_form,
            temperature=temperature if free_form else None,
            top_p=top_p if free_form else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Test Zydra or send it one prompt.")
    parser.add_argument("--model", type=Path, default=Path("output/zydra-v2-hf"))
    parser.add_argument("--prompt", help="Run one prompt instead of the default smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--free-form",
        action="store_true",
        help="Treat --prompt as a raw text prefix and sample a continuation.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.free_form and args.prompt is None:
        parser.error("--free-form requires --prompt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = ZydraForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    prompts = [args.prompt] if args.prompt is not None else [
        "halo",
        "Siapa developer-mu?",
        "kau di buat menggunakan model apa??",
        "kau bisa apa saja??",
    ]
    for prompt in prompts:
        generated = generate_text(
            prompt,
            tokenizer,
            model,
            device,
            args.max_new_tokens,
            args.free_form,
            args.temperature,
            args.top_p,
        )
        print(f"Prompt: {prompt}")
        print(f"Zydra: {generated or '[tidak ada keluaran]'}\n")


if __name__ == "__main__":
    main()