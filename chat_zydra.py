import argparse

import torch
from transformers import PreTrainedTokenizerFast

from modeling_zydra import ZydraForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Interactive terminal chat with Zydra.")
    parser.add_argument("--model", default="output/zydra-v2-hf")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = ZydraForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    context_limit = model.config.max_position_embeddings
    history = ""

    print("Zydra CLI siap. Ketik pertanyaan; ketik 'keluar' atau '/exit' untuk berhenti.")
    print("Catatan: model masih eksperimen dan dapat memberi jawaban keliru.")
    while True:
        try:
            user_text = input("Anda> ").strip()
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

        prompt = f"{history}Pertanyaan: {user_text}\nJawaban:"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = token_ids[-(context_limit - 1) :]
        if tokenizer.bos_token_id is not None:
            token_ids.insert(0, tokenizer.bos_token_id)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        generation_limit = min(args.max_new_tokens, context_limit - input_ids.shape[1])
        if generation_limit < 1:
            print("Zydra> Konteks penuh. Ketik /reset untuk memulai percakapan baru.")
            continue

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_limit,
                do_sample=False,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
        print(f"Zydra> {answer or '[tidak ada keluaran]'}")
        history = f"{history}Pertanyaan: {user_text}\nJawaban: {answer}\n"


if __name__ == "__main__":
    main()