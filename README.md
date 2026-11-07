# G-1N — GOdev Neural Model

**G-1N** is an experimental language model built from scratch by GOdev with PyTorch. It is a small decoder-only Transformer that learns next-byte prediction from a text corpus.

- No Qwen, GPT, or other pretrained model weights.
- Random initialization: the model starts with newly initialized parameters.
- Tokenizer: UTF-8 bytes (256 possible token IDs).
- Current stage: educational prototype, not a trained assistant.

## Project structure

```text
G-1N-Model/
├── g1n/
│   ├── __init__.py
│   ├── model.py
│   └── tokenizer.py
├── data/
│   └── train.txt
├── web/
│   └── index.html
├── train.py
├── generate.py
├── app.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Setup

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

## Train from scratch

Add a sufficiently large UTF-8 text corpus to `data/train.txt`. Start with a small test run:

```bash
python train.py --steps 300 --batch-size 4 --context 64 --d-model 64 --heads 4 --layers 2
```

A larger experiment can use:

```bash
python train.py --steps 5000 --batch-size 8 --context 128 --d-model 128 --heads 4 --layers 4
```

The training script prints training and validation loss and saves the latest checkpoint to `checkpoints/g1n.pt`.

## Generate

After training:

```bash
python generate.py --prompt "Hello, GOdev!" --tokens 100
```

Optional settings:

```bash
python generate.py --prompt "Hello" --tokens 100 --temperature 0.8 --top-k 40
```

## Local browser demo

```bash
python app.py
```

Open `http://127.0.0.1:5000` on the same device.

## Model configuration

The default starter configuration is:

| Parameter | Default |
|---|---:|
| Vocabulary | 256 byte IDs |
| Context length | 128 |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Objective | Next-byte prediction |

These are small development settings, not a claim of high capability.

## Android and Acode

Acode can edit this project's files, but its HTML preview does not run PyTorch. Training requires a Python environment with PyTorch installed. On a phone with limited memory, use a cloud notebook or computer for training rather than attempting the larger settings locally.

## Data and limitations

The included `data/train.txt` is only a tiny smoke-test corpus. It cannot teach the model useful general language ability. A useful model needs much more high-quality, permitted training data, compute, validation, and testing.

Use text you wrote, public-domain material, or datasets whose licenses permit your intended use. Do not add private information, passwords, or material you do not have permission to use.

## License

MIT. See [LICENSE](LICENSE).
