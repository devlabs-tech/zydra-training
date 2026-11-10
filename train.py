import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from modeling_zydra import ZydraConfig, ZydraForCausalLM


class TokenBlocks(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        token_ids = []
        for text in texts:
            token_ids.extend(tokenizer(text, add_special_tokens=True)["input_ids"])

        self.blocks = [
            torch.tensor(token_ids[start : start + block_size + 1], dtype=torch.long)
            for start in range(0, len(token_ids) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        block = self.blocks[index]
        inputs = block[:-1]
        # Causal language models shift labels internally during loss calculation.
        return inputs, inputs


def load_texts(path, data_format):
    if data_format == "text":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"No training text found in {path}.")
        return [text]

    texts = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number} must contain a non-empty string field 'text'.")
            texts.append(text.strip())
    if not texts:
        raise ValueError(f"No training texts found in {path}.")
    return texts


def load_text_directory(directory):
    if not directory.is_dir():
        raise ValueError(f"Training data directory does not exist: {directory}")

    files = sorted(path for path in directory.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"No .txt training files found under {directory}.")

    texts = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Training text file is empty: {path}")
        texts.append(text)
        print(f"Loaded {path}")
    return texts


def build_toy_train(texts, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dataset_file:
        for text in texts:
            dataset_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Built training dataset {output_path} from {len(texts)} text files.")


def train_tokenizer(texts, vocab_size, output_dir):
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Zydra from random weights on a text or JSONL corpus.")
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--data", type=Path, help="Path to one text or JSONL training file.")
    data_source.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing .txt files; defaults to train/ and searches subdirectories.",
    )
    parser.add_argument("--data-format", choices=("jsonl", "text"), default="jsonl")
    parser.add_argument("--output", type=Path, default=Path("output/zydra-v2-hf"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=704)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.heads < 1 or args.kv_heads < 1 or args.hidden_size % args.heads != 0:
        parser.error("--hidden-size must be divisible by positive --heads")
    if args.heads % args.kv_heads != 0:
        parser.error("--heads must be divisible by --kv-heads")
    if args.block_size < 2 or args.epochs < 1:
        parser.error("--block-size must be >= 2 and --epochs must be >= 1")

    torch.manual_seed(args.seed)
    if args.data is not None:
        data_path = args.data
        texts = load_texts(data_path, args.data_format)
        print(f"Loaded {data_path}")
    else:
        data_directory = args.data_dir or Path("train")
        source_texts = load_text_directory(data_directory)
        dataset_path = Path("data/toy_train.jsonl")
        build_toy_train(source_texts, dataset_path)
        texts = load_texts(dataset_path, "jsonl")
        print(f"Loaded {len(texts)} documents from {dataset_path}.")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = train_tokenizer(texts, args.vocab_size, args.output)
    training_data = TokenBlocks(texts, tokenizer, args.block_size)
    if not training_data:
        raise ValueError("The corpus is too small; add more text or reduce --block-size.")

    config = ZydraConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.block_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = ZydraForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(loader) * args.epochs
    step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = model(input_ids=inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            step += 1
            if step % 50 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}")
        print(f"epoch {epoch + 1}/{args.epochs} mean_loss {total_loss / len(loader):.4f}")

    model.config.architectures = ["ZydraForCausalLM"]
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fresh-from-scratch Zydra checkpoint to {args.output}")


if __name__ == "__main__":
    main()