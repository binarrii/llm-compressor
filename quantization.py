import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    _MODEL = args.model
    _OUT_DIR = f"{_MODEL[_MODEL.rfind('/') + 1 :]}-FP8"

    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["re:.*lm_head"]
    )

    oneshot(model=model, recipe=recipe, output_dir=_OUT_DIR)
