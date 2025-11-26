"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
try:
    import rustbpe
    import tiktoken
    RUSTBPE_AVAILABLE = True
except ImportError:
    RUSTBPE_AVAILABLE = False

if RUSTBPE_AVAILABLE:
    class RustBPETokenizer:
        """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

        def __init__(self, enc, bos_token):
            self.enc = enc
            self.bos_token_id = self.encode_special(bos_token)

        @classmethod
        def train_from_iterator(cls, text_iterator, vocab_size):
            # 1) train using rustbpe
            tokenizer = rustbpe.Tokenizer()
            # the special tokens are inserted later in __init__, we don't train them here
            vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
            assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
            tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
            # 2) construct the associated tiktoken encoding for inference
            pattern = tokenizer.get_pattern()
            mergeable_ranks_list = tokenizer.get_mergeable_ranks()
            mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
            tokens_offset = len(mergeable_ranks)
            special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
            enc = tiktoken.Encoding(
                name="rustbpe",
                pat_str=pattern,
                mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
                special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
            )
            return cls(enc, "<|bos|>")

        @classmethod
        def from_directory(cls, tokenizer_dir):
            pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
            with open(pickle_path, "rb") as f:
                enc = pickle.load(f)
            return cls(enc, "<|bos|>")

        @classmethod
        def from_pretrained(cls, tiktoken_name):
            # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
            enc = tiktoken.get_encoding(tiktoken_name)
            # tiktoken calls the special document delimiter token "<|endoftext|>"
            # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
            # it most often is used to signal the start of a new sequence to the LLM during inference etc.
            # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
            return cls(enc, "<|endoftext|>")

        def get_vocab_size(self):
            return self.enc.n_vocab

        def get_special_tokens(self):
            return self.enc.special_tokens_set

        def id_to_token(self, id):
            return self.enc.decode([id])

        @lru_cache(maxsize=32)
        def encode_special(self, text):
            return self.enc.encode_single_token(text)

        def get_bos_token_id(self):
            return self.bos_token_id

        def encode(self, text, prepend=None, append=None, num_threads=8):
            # text can be either a string or a list of strings

            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            if append is not None:
                append_id = append if isinstance(append, int) else self.encode_special(append)

            if isinstance(text, str):
                ids = self.enc.encode_ordinary(text)
                if prepend is not None:
                    ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
                if append is not None:
                    ids.append(append_id)
            elif isinstance(text, list):
                ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
                if prepend is not None:
                    for ids_row in ids:
                        ids_row.insert(0, prepend_id) # TODO: same
                if append is not None:
                    for ids_row in ids:
                        ids_row.append(append_id)
            else:
                raise ValueError(f"Invalid input type: {type(text)}")

            return ids

        def __call__(self, *args, **kwargs):
            return self.encode(*args, **kwargs)

        def decode(self, ids):
            return self.enc.decode(ids)

        def save(self, tokenizer_dir):
            # save the encoding object to disk
            os.makedirs(tokenizer_dir, exist_ok=True)
            pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(self.enc, f)
            print(f"Saved tokenizer encoding to {pickle_path}")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from pilot.nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # Try RustBPE first, fallback to HuggingFace
    if RUSTBPE_AVAILABLE:
        try:
            return RustBPETokenizer.from_directory(tokenizer_dir)
        except:
            pass
    return HuggingFaceTokenizer.from_directory(tokenizer_dir)
