from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# process japanese text


def process_japanese_text(text_list):
    """
    处理日语文本，保留原始字符而不进行拼音转换
    
    参数:
        text_list: 文本列表
        
    返回:
        处理后的文本列表，每个文本是一个字符列表
    """
    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", """: '"', """: '"', "'": "'", "'": "'"}
    )  # 自定义转换，处理特殊字符
    
    for text in text_list:
        char_list = list(text.translate(custom_trans))
        final_text_list.append(char_list)
        
    return final_text_list


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        # 更精确的中文字符范围，不包括日语字符
        return (
            ('\u4E00' <= c <= '\u9FFF') or  # CJK统一汉字
            ('\u3400' <= c <= '\u4DBF') or  # CJK统一汉字扩展A
            ('\u20000' <= c <= '\u2A6DF') or  # CJK统一汉字扩展B
            ('\u2A700' <= c <= '\u2B73F') or  # CJK统一汉字扩展C
            ('\u2B740' <= c <= '\u2B81F') or  # CJK统一汉字扩展D
            ('\u2B820' <= c <= '\u2CEAF') or  # CJK统一汉字扩展E
            ('\u2CEB0' <= c <= '\u2EBEF') or  # CJK统一汉字扩展F
            ('\u30000' <= c <= '\u3134F') or  # CJK统一汉字扩展G
            ('\uF900' <= c <= '\uFAFF')  # CJK兼容汉字
        )
        
    def is_japanese(c):
        return (
            "\u3040" <= c <= "\u309f" or  # 平假名
            "\u30a0" <= c <= "\u30ff" or  # 片假名
            "\uff66" <= c <= "\uff9f"  # 半角片假名
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_japanese(c):
                        char_list.append(c)
                    elif is_chinese(c):
                        if not char_list or not is_japanese(char_list[-1]):
                            char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            if not char_list or not is_japanese(char_list[-1]):
                                char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False
