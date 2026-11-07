"""Generate text from a trained G-1N checkpoint."""
import argparse
from pathlib import Path

import torch

from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with G-1N.")
    parser.add_argument("--checkpoint", default="checkpoints/g1n.pt")
    parser.add_argument("--prompt", default="Hello, GOdev! ")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    if args.tokens < 0 or args.temperature <= 0 or args.top_k < 0:
        raise SystemExit("--tokens and --top-k must be non-negative; temperature must be positive.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    # Checkpoints are files produced by this project's own training script.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = G1N(
        config["vocab_size"],
        config["context_size"],
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt) or [32]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    output = model.generate(
        input_ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
