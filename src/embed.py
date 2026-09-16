#!/usr/bin/env python3
"""
embed.py - meaning-level matching with no model server, no GPU, no new dependency.

WHY THIS EXISTS. Keyword search cannot connect words that mean the same thing.
Asked "did the user's residence change?", our session search could not find
"I just relocated to Melbourne", because not one word matches. On the MemConflict
benchmark that single blind spot sits under most of our lost points: the right
turn was in the index and ranked below the cut.

HOW. A static embedding model (model2vec's potion family): a table of one vector
per word piece, distilled from a sentence-transformer. Encoding a sentence is a
tokeniser pass and an average of table rows, so it needs numpy and nothing else,
loads in well under a second, and encodes tens of thousands of turns in seconds
on a laptop CPU. There is no neural network to run, which is the point: it can
live next to a memory system whose whole design rule is "no model on the query
path".

The tokeniser is reimplemented here (BERT normaliser, BERT pre-tokeniser,
WordPiece) so the `tokenizers` package is not required. It was checked
against the reference model2vec implementation: identical token ids and cosine
1.000000 on every text tried. `embed.py check` compares two texts.

Usage:
  embed.py fetch [--model potion-base-8M]
  embed.py check  "text one" "text two"      cosine similarity of two texts
"""

import json
import os
import re
import struct
import sys
import unicodedata
import urllib.request

import numpy as np

MODEL = os.environ.get("QUIPU_EMBED_MODEL", "potion-base-8M")
CACHE = os.path.expanduser(os.environ.get("QUIPU_EMBED_CACHE", "~/.cache/quipu/models"))
HF = "https://huggingface.co/minishlab/%s/resolve/main/%s"
FILES = ("model.safetensors", "tokenizer.json", "config.json")
MAX_TOKENS = 512


def model_dir(name=None):
    return os.path.join(CACHE, name or MODEL)


def fetch(name=None, verbose=True):
    """Download the model files once. Plain HTTPS, no hub client needed."""
    d = model_dir(name)
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        path = os.path.join(d, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        tmp = path + ".part"
        urllib.request.urlretrieve(HF % (name or MODEL, f), tmp)
        os.replace(tmp, path)
        if verbose:
            print("fetched %s (%.1f MB)" % (f, os.path.getsize(path) / 1e6))
    return d


def available(name=None):
    d = model_dir(name)
    return all(os.path.exists(os.path.join(d, f)) for f in FILES)


# --- tokeniser ----------------------------------------------------------------

def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """BertNormalizer + BertPreTokenizer + WordPiece, as tokenizer.json describes."""

    def __init__(self, tok_json):
        m = tok_json["model"]
        if m.get("type") != "WordPiece":
            raise ValueError("unsupported tokenizer model %r" % m.get("type"))
        self.vocab = m["vocab"]
        self.unk = m.get("unk_token", "[UNK]")
        self.unk_id = self.vocab.get(self.unk)
        self.prefix = m.get("continuing_subword_prefix", "##")
        self.max_chars = m.get("max_input_chars_per_word", 100)
        # Literal special tokens such as "[UNK]" are matched before normalising,
        # exactly as the reference tokeniser does. Without this "[UNK]" in a turn
        # tokenises as "[", "unk", "]" and the vector drifts (cosine 0.83).
        self.added = sorted((t["content"] for t in tok_json.get("added_tokens") or []
                             if t.get("content") in self.vocab), key=len, reverse=True)
        self._added_re = (re.compile("(" + "|".join(re.escape(t) for t in self.added) + ")")
                          if self.added else None)
        norm = tok_json.get("normalizer") or {}
        self.lower = norm.get("lowercase", True)
        strip = norm.get("strip_accents")
        self.strip_accents = self.lower if strip is None else strip
        self._memo = {}

    def normalize(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD:
                continue
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in "\t\n\r":
                continue
            if ch in "\t\n\r" or cat == "Zs":
                out.append(" ")
            elif _is_cjk(cp):
                out.append(" %s " % ch)
            else:
                out.append(ch)
        text = "".join(out)
        if self.lower:
            text = text.lower()
        if self.strip_accents:
            text = "".join(c for c in unicodedata.normalize("NFD", text)
                           if unicodedata.category(c) != "Mn")
        return text

    def pre_tokenize(self, text):
        words = []
        for chunk in text.split():
            buf = []
            for ch in chunk:
                if _is_punct(ch):
                    if buf:
                        words.append("".join(buf))
                        buf = []
                    words.append(ch)
                else:
                    buf.append(ch)
            if buf:
                words.append("".join(buf))
        return words

    def word_ids(self, word):
        hit = self._memo.get(word)
        if hit is not None:
            return hit
        if len(word) > self.max_chars:
            ids = [self.unk_id]
        else:
            ids, start = [], 0
            while start < len(word):
                end, cur = len(word), None
                while start < end:
                    piece = word[start:end]
                    if start > 0:
                        piece = self.prefix + piece
                    if piece in self.vocab:
                        cur = self.vocab[piece]
                        break
                    end -= 1
                if cur is None:
                    ids = [self.unk_id]
                    break
                ids.append(cur)
                start = end
        if len(self._memo) < 200000:
            self._memo[word] = ids
        return ids

    def encode(self, text):
        ids = []
        parts = self._added_re.split(text or "") if self._added_re else [text or ""]
        for part in parts:
            if not part:
                continue
            if part in self.vocab and part in self.added:
                ids.append(self.vocab[part])
                continue
            for w in self.pre_tokenize(self.normalize(part)):
                ids.extend(self.word_ids(w))
                if len(ids) >= MAX_TOKENS:
                    return ids[:MAX_TOKENS]
        return ids[:MAX_TOKENS]


# --- model --------------------------------------------------------------------

def _load_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
    dtypes = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        arr = np.frombuffer(mm[base + start: base + end], dtype=dtypes[info["dtype"]])
        tensors[name] = arr.reshape(info["shape"])
    return tensors


class StaticModel:
    def __init__(self, name=None):
        d = model_dir(name)
        tensors = _load_safetensors(os.path.join(d, "model.safetensors"))
        key = "embeddings" if "embeddings" in tensors else next(iter(tensors))
        self.table = np.asarray(tensors[key], dtype=np.float32)
        self.tok = WordPiece(json.load(open(os.path.join(d, "tokenizer.json"))))
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.normalize = cfg.get("normalize", True)
        self.dim = int(self.table.shape[1])

    def encode(self, texts):
        """One L2-normalised row per text. Empty text encodes to a zero vector."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            ids = [x for x in self.tok.encode(t) if x is not None and x != self.tok.unk_id]
            if not ids:
                continue
            v = self.table[ids].mean(axis=0)
            if self.normalize:
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n
            out[i] = v
        return out[0] if single else out


_MODEL = None


def model():
    """Process-wide singleton. Loading is cheap, but not free on a hot path."""
    global _MODEL
    if _MODEL is None:
        if not available():
            fetch(verbose=False)
        _MODEL = StaticModel()
    return _MODEL


def to_blob(vec):
    return np.asarray(vec, dtype=np.float16).tobytes()


def from_blob(blob, dim):
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(dim)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fetch":
        name = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--model" else None
        print(fetch(name))
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        m = model()
        a, b = m.encode([sys.argv[2], sys.argv[3]])
        print("cosine %.3f" % float(a @ b))
    else:
        print(__doc__)
