import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_CHECK_TEXT = (
    "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ "
    "(multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 "
    "3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ "
    "------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? "
    "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, "
    "'D you like some tea? We'Ve a'lL"
)


def main():
    parser = argparse.ArgumentParser(
        description="Export a Zydra Hugging Face checkpoint to GGUF with llama.cpp."
    )
    parser.add_argument("--model", type=Path, default=Path("output/zydra-v2-hf"))
    parser.add_argument("--llama-cpp", type=Path, default=Path("llama.cpp"))
    parser.add_argument("--outfile", type=Path, default=Path("model/zydra-v2-f16.gguf"))
    parser.add_argument(
        "--outtype",
        choices=("f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"),
        default="f16",
    )
    args = parser.parse_args()

    model_dir = args.model.resolve()
    llama_cpp_dir = args.llama_cpp.resolve()
    converter_path = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not model_dir.is_dir():
        parser.error(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        parser.error(f"Expected config.json and model.safetensors in {model_dir}")
    if not converter_path.is_file():
        parser.error(f"llama.cpp converter not found: {converter_path}")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.outfile.resolve()
    with tempfile.TemporaryDirectory(prefix="zydra-gguf-") as temp_dir:
        temp_root = Path(temp_dir)
        staged_model = temp_root / "hf-model"
        converter_root = temp_root / "converter"
        converter_root.mkdir()

        shutil.copytree(model_dir, staged_model)
        config_path = staged_model / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model_type"] = "llama"
        config["architectures"] = ["LlamaForCausalLM"]
        config_path.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

        tokenizer = AutoTokenizer.from_pretrained(staged_model)
        token_ids = tokenizer.encode(TOKENIZER_CHECK_TEXT)
        tokenizer_hash = hashlib.sha256(str(token_ids).encode()).hexdigest()

        shutil.copy2(converter_path, converter_root / "convert_hf_to_gguf.py")
        shutil.copytree(llama_cpp_dir / "conversion", converter_root / "conversion")
        shutil.copytree(llama_cpp_dir / "gguf-py", converter_root / "gguf-py")

        base_path = converter_root / "conversion" / "base.py"
        converter_source = base_path.read_text(encoding="utf-8")
        insertion_point = '        if res is None:\n            logger.warning("\\n")'
        if insertion_point not in converter_source:
            parser.error(
                "The llama.cpp converter layout has changed; cannot register "
                "Zydra's ByteLevel tokenizer safely."
            )
        tokenizer_mapping = (
            f'        if chkhsh == "{tokenizer_hash}":\n'
            '            res = "gpt-2"\n\n'
        )
        base_path.write_text(
            converter_source.replace(
                insertion_point,
                tokenizer_mapping + insertion_point,
                1,
            ),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(converter_root / "convert_hf_to_gguf.py"),
            str(staged_model),
            "--outfile",
            str(output_path),
            "--outtype",
            args.outtype,
            "--model-name",
            "Zydra",
        ]
        print(f"Converting {model_dir} to {output_path}")
        subprocess.run(command, check=True)

    print(f"GGUF model saved to {output_path}")


if __name__ == "__main__":
    main()
