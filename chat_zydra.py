import argparse
import os

import torch
from transformers import PreTrainedTokenizerFast

from modeling_zydra import ZydraForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Interactive terminal chat with Zydra.")
    parser.add_argument("--model", default="output/zydra-v2-hf")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, >0 = sampling")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = ZydraForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    context_limit = model.config.max_position_embeddings
    history = ""

    os.system("clear" if os.name == "posix" else "cls")
    print("Zydra CLI")
    print("Catatan: model masih eksperimen dan dapat memberi jawaban keliru.")

    while True:
        try:
            user_text = input("Kamu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"keluar", "/exit", "/quit"}:
            break
        if user_text == "/reset":
            history = ""
            print("Riwayat dihapus.")
            continue
        if not user_text:
            continue

        question = user_text if user_text.endswith("?") else user_text + "?"
        prompt = f"{history}{question}\n"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = token_ids[-(context_limit - 1):]
        if tokenizer.bos_token_id is not None:
            token_ids.insert(0, tokenizer.bos_token_id)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        generation_limit = min(args.max_new_tokens, context_limit - input_ids.shape[1])
        if generation_limit < 1:
            print("Zydra> Konteks penuh. Ketik /reset untuk memulai percakapan baru.")
            continue

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=generation_limit,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)

        raw = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        answer = raw.strip().split("\n")[0].strip()
        print(f"Zydra> {answer or '[tidak ada keluaran]'}")
        history = f"{history}{question}\n{answer}\n"


if __name__ == "__main__":
    main()