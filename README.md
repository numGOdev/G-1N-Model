# G-1N — GOdev Transformer from Scratch

G-1N is an educational decoder-only Transformer implemented with PyTorch. It starts with random weights and trains on text supplied by you. It does **not** download, call, or reuse Qwen/GPT model weights.

## Important expectations
This is a tiny starter model, not a competitive general-purpose assistant. The included dataset is only a toy sample to verify the pipeline. It will generate poor text until trained on a much larger, legally usable dataset. Training a capable model requires substantial compute, data, evaluation, and iteration.

## Project layout
- `g1n/model.py` — causal self-attention, MLP, Transformer blocks, language-model head
- `g1n/tokenizer.py` — UTF-8 byte tokenizer (256-token vocabulary)
- `train.py` — train from random initialization and save weights
- `generate.py` — generate text from a saved checkpoint
- `app.py` + `web/index.html` — local browser demo
- `data/train.txt` — tiny example corpus
- `checkpoints/` — trained weights are saved here

## Train (recommended: Google Colab or a computer)
Install dependencies:
```bash
python -m pip install -r requirements.txt
```
Add your own permitted training text to `data/train.txt`, then run a small test:
```bash
python train.py --steps 300 --batch-size 8 --context 64 --d-model 64 --heads 4 --layers 2
```
For a somewhat larger experiment (needs more RAM/compute):
```bash
python train.py --steps 5000 --batch-size 16 --context 128 --d-model 128 --heads 4 --layers 4
```
The default checkpoint is `checkpoints/g1n.pt`.

Generate:
```bash
python generate.py --prompt "Hello, GOdev! " --tokens 120
```

Run the local web demo after training:
```bash
python app.py
```
Open `http://127.0.0.1:5000`.

## Android + Acode
Use Acode to edit the HTML and Python files. Acode's HTML preview cannot train or run a PyTorch model by itself. PyTorch generally needs a separate supported Python environment and may not install/run on many Android phones. With a phone with limited RAM, do not attempt the larger training settings locally. A practical workflow is to edit in Acode, train in a cloud notebook such as Colab, download `checkpoints/g1n.pt`, then run inference wherever PyTorch is supported.

## Model details
- Architecture: decoder-only Transformer
- Attention: causal multi-head self-attention
- Position encoding: learned positional embeddings
- Tokenization: byte-level UTF-8
- Objective: next-token cross-entropy
- Optimizer: AdamW
- Initialization: random weights

## Dataset and licensing
Use text you wrote, public-domain text, or datasets whose licenses permit your intended use. Keep personal data, credentials, and copyrighted books you do not have permission to use out of the corpus.

## GitHub
Do not commit large training datasets, checkpoints, API keys, or private data. The `.gitignore` excludes checkpoints and Python cache. You can publish source code and a small sample dataset.
