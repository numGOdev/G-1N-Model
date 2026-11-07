"""Train G-1N from random initialization on a UTF-8 text corpus."""
import argparse
import random
from pathlib import Path

import torch

from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the G-1N language model.")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--out", default="checkpoints/g1n.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.steps, args.batch_size, args.context, args.d_model,
           args.heads, args.layers, args.eval_every) <= 0:
        raise SystemExit("Training sizes and intervals must be positive.")
    if args.d_model % args.heads:
        raise SystemExit("--d-model must be divisible by --heads.")
    if not 0.0 <= args.dropout < 1.0 or args.lr <= 0:
        raise SystemExit("Dropout must be in [0, 1) and learning rate must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Training file not found: {data_path}")

    tokenizer = ByteTokenizer()
    token_ids = torch.tensor(
        tokenizer.encode(data_path.read_text(encoding="utf-8")),
        dtype=torch.long,
    )
    if len(token_ids) <= args.context + 1:
        raise SystemExit(
            f"Corpus is too short: need at least {args.context + 2} bytes; "
            f"found {len(token_ids)}."
        )

    # Keep a validation section when the corpus is large enough.
    split = int(len(token_ids) * 0.9)
    if split > args.context + 1 and len(token_ids) - split > args.context:
        train_ids, val_ids = token_ids[:split], token_ids[split:]
    else:
        print("Note: corpus is too small for a separate validation set; reusing training data.")
        train_ids = val_ids = token_ids

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Training bytes: {len(train_ids)}")
    model = G1N(
        tokenizer.vocab_size,
        args.context,
        args.d_model,
        args.heads,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def make_batch(source):
        max_start = len(source) - args.context - 1
        starts = torch.randint(max_start + 1, (args.batch_size,))
        x = torch.stack([source[int(s):int(s) + args.context] for s in starts])
        y = torch.stack([source[int(s) + 1:int(s) + args.context + 1] for s in starts])
        return x.to(device), y.to(device)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(train_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batch(val_ids)
                _, val_loss = model(vx, vy)
            print(
                f"step {step}/{args.steps} | "
                f"train loss {loss.item():.4f} | val loss {val_loss.item():.4f}"
            )
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "context_size": args.context,
                    "d_model": args.d_model,
                    "n_heads": args.heads,
                    "n_layers": args.layers,
                    "dropout": args.dropout,
                },
                "step": step,
                "model_name": "G-1N",
                "version": "0.2.0",
            }, output_path)

    print(f"Checkpoint saved to: {output_path}")
    print("Reminder: the tiny starter corpus is only for testing the training pipeline.")


if __name__ == "__main__":
    main()
