"""
Nexus Tokenizer - BPE-based tokenizer đơn giản cho song ngữ Việt-Anh
====================================================================
Hỗ trợ:
- Subword tokenization (BPE đơn giản)
- Special tokens: <pad>, <bos>, <eos>, <unk>, <system>, <user>, <assistant>
- Vocabulary size: 32,000
- Lưu/Load từ file JSON
"""
import json
import re
import os
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict


# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SYSTEM_TOKEN = "<system>"
USER_TOKEN = "<user>"
ASSISTANT_TOKEN = "<assistant>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
]

# ID của special tokens
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6


class SimpleBPETokenizer:
    """BPE Tokenizer đơn giản - huấn luyện được trên corpus nhỏ."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._is_trained = False

    def _get_word_freq(self, corpus: List[str]) -> Counter:
        """Đếm tần suất từ trong corpus."""
        word_freq = Counter()
        for text in corpus:
            words = text.split()
            for word in words:
                # Tách theo ký tự + thêm marker end-of-word
                chars = " ".join(list(word)) + " </w>"
                word_freq[chars] += 1
        return word_freq

    def _get_pairs(self, word_freq: Counter) -> Counter:
        """Đếm tần suất các cặp token."""
        pairs = Counter()
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge(self, pair: Tuple[str, str], word_freq: Counter) -> Counter:
        """Merge một cặp token."""
        new_word_freq = Counter()
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in word_freq.items():
            new_word = pattern.sub("".join(pair), word)
            new_word_freq[new_word] += freq
        return new_word_freq

    def train(self, corpus: List[str], verbose: bool = False) -> None:
        """Huấn luyện BPE trên corpus."""
        # Init vocab với special tokens + ký tự ASCII cơ bản
        self.vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        next_id = len(SPECIAL_TOKENS)

        # Thêm các ký tự cơ bản (a-z, 0-9, dấu câu)
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:-'\"()[]{} \n\t":
            if c not in self.vocab:
                self.vocab[c] = next_id
                next_id += 1

        # Thêm các ký tự tiếng Việt có dấu
        vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẲÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
        for c in vietnamese_chars:
            if c not in self.vocab:
                self.vocab[c] = next_id
                next_id += 1

        # Thêm các từ phổ biến (song ngữ)
        common_words = [
            # Tiếng Việt
            "tôi", "bạn", "của", "là", "và", "có", "một", "người", "trong", "cho",
            "với", "đó", "này", "không", "để", "được", "nào", "cũng", "đã", "sẽ",
            "về", "khi", "mà", "nhiều", "làm", "ra", "đến", "từ", "các", "hoặc",
            "ai", "gì", "đâu", "sao", "như", "vậy", "thế", "còn", "nhưng", "nếu",
            "nexus", "coder", "ai", "model", "hieu", "louis", "tác", "giả",
            # English
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "are", "was", "were",
            "be", "been", "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "may", "might", "must", "shall", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "code", "function", "class", "def", "return", "import", "from", "python",
            "nexus", "coder", "model", "agent", "ai", "author", "hieu", "louis",
        ]
        for word in common_words:
            token = word + "</w>"
            if token not in self.vocab and next_id < self.vocab_size:
                self.vocab[token] = next_id
                next_id += 1

        # BPE merges
        word_freq = self._get_word_freq(corpus)
        num_merges = self.vocab_size - next_id

        for i in range(num_merges):
            pairs = self._get_pairs(word_freq)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1].replace("</w>", "") + ("</w>" if "</w>" in best_pair[1] else "")

            if new_token in self.vocab:
                # Đã tồn tại, skip
                word_freq = self._merge(best_pair, word_freq)
                continue

            self.merges[best_pair] = i
            self.vocab[new_token] = next_id
            next_id += 1
            word_freq = self._merge(best_pair, word_freq)

            if verbose and i % 1000 == 0:
                print(f"  Merge {i}/{num_merges}: {best_pair} -> {new_token}")

        # Build reverse vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._is_trained = True

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize một từ sử dụng BPE merges."""
        if not self.merges:
            return [c for c in word] + ["</w>"]

        chars = list(word) + ["</w>"]
        while len(chars) > 1:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            valid_merges = [(pair, self.merges[pair]) for pair in pairs if pair in self.merges]
            if not valid_merges:
                break
            best_pair = min(valid_merges, key=lambda x: x[1])[0]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                    new_chars.append(chars[i] + chars[i + 1].replace("</w>", "") + ("</w>" if "</w>" in chars[i + 1] else ""))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars

        return chars

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Encode text thành list of token IDs."""
        if not self._is_trained:
            raise RuntimeError("Tokenizer chưa được huấn luyện. Gọi .train() hoặc .load() trước.")

        # Tách special tokens nếu có trong text
        for token in SPECIAL_TOKENS:
            text = text.replace(token, f" {token} ")

        words = text.split()
        ids = []

        if add_special:
            ids.append(BOS_ID)

        for word in words:
            if word in SPECIAL_TOKENS:
                ids.append(self.vocab[word])
                continue
            tokens = self._tokenize_word(word)
            for tok in tokens:
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                else:
                    # Fallback: encode từng ký tự
                    for c in tok:
                        if c in self.vocab:
                            ids.append(self.vocab[c])
                        else:
                            ids.append(UNK_ID)

        if add_special:
            ids.append(EOS_ID)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode list of token IDs thành text."""
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                tok = self.id_to_token[id_]
                if tok in SPECIAL_TOKENS:
                    tokens.append(f" {tok} ")
                else:
                    # Remove </w> marker
                    clean = tok.replace("</w>", " ")
                    tokens.append(clean)
            else:
                tokens.append(UNK_TOKEN)

        text = "".join(tokens)
        # Cleanup multiple spaces
        text = " ".join(text.split())
        return text.strip()

    def save(self, path: str) -> None:
        """Lưu tokenizer ra file JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load tokenizer từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.merges = {tuple(k.split("|")): v for k, v in data["merges"].items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._is_trained = True


class NexusTokenizer:
    """High-level wrapper cho Nexus Coder tokenizer."""

    def __init__(self, vocab_path: Optional[str] = None, vocab_size: int = 32000):
        self.bpe = SimpleBPETokenizer(vocab_size=vocab_size)
        if vocab_path and os.path.exists(vocab_path):
            self.bpe.load(vocab_path)

    def train(self, corpus: List[str], verbose: bool = False) -> None:
        self.bpe.train(corpus, verbose=verbose)

    def save(self, path: str) -> None:
        self.bpe.save(path)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        return self.bpe.encode(text, add_special=add_special)

    def decode(self, ids: List[int]) -> str:
        return self.bpe.decode(ids)

    def encode_chat(
        self,
        system: str,
        user: str,
        assistant: str = "",
    ) -> List[int]:
        """Encode một hội thoại theo format chat."""
        ids = [BOS_ID, SYSTEM_ID]
        ids.extend(self.bpe.encode(system))
        ids.append(USER_ID)
        ids.extend(self.bpe.encode(user))
        ids.append(ASSISTANT_ID)
        if assistant:
            ids.extend(self.bpe.encode(assistant))
            ids.append(EOS_ID)
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.bpe.vocab)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID
