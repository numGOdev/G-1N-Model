"""Simple UTF-8 byte tokenizer: vocabulary is always 256 IDs."""
class ByteTokenizer:
    vocab_size = 256
    def encode(self, text):
        return list(text.encode("utf-8", errors="replace"))
    def decode(self, ids):
        return bytes([int(x) % 256 for x in ids]).decode("utf-8", errors="replace")
