from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .emoji import emoji_runs, safe_cut

OPEN_BRACKETS = "「『（(【"
CLOSE_BRACKETS = "」』）)】"
SENTENCE_END = "。！？!?"
DIALOGUE_OPEN = "「『"

_RUBY_EXPLICIT = re.compile(r"[|｜]([^《|｜\n]+)《([^》]+)》")
_RUBY_KANJI = re.compile(r"([一-龥々〆ヶ]+)《([^》]+)》")
_AOZORA_NOTE = re.compile(r"［＃[^］]*］")
_WORD = re.compile(r"\w")

AUTO_TITLES = ("序", "全編")  # split_chapters が付ける題。本文に無いので読まない


@dataclass
class Segment:
    chapter_idx: int  # 0 始まり
    para_idx: int  # 章内の段落番号
    seg_idx: int  # 段落内の文番号
    kind: str  # "narration" | "dialogue" | "title" | "break"
    text: str

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.chapter_idx, self.para_idx, self.seg_idx)


@dataclass
class Chapter:
    idx: int
    title: str
    paragraphs: list[list[Segment]]


def read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp932")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text(text: str) -> str:
    """青空文庫形式のルビ・注記を読み上げ用に整える。ルビは読み仮名に置き換える。"""
    text = _AOZORA_NOTE.sub("", text)
    text = _RUBY_EXPLICIT.sub(r"\2", text)
    text = _RUBY_KANJI.sub(r"\2", text)
    return text


def split_chapters(text: str, chapter_regex: str) -> list[tuple[str, str]]:
    """(タイトル, 本文) のリスト。見出しが無ければ全体を 1 章とする。"""
    pattern = re.compile(chapter_regex, re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [("全編", text)]
    chapters: list[tuple[str, str]] = []
    head = text[: matches[0].start()]
    if head.strip():
        chapters.append(("序", head))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(0).strip()
        chapters.append((title, text[m.end() : end]))
    return chapters


def split_paragraphs(body: str) -> list[str]:
    return [line.strip() for line in body.split("\n") if line.strip()]


def split_sentences(paragraph: str, max_chars: int = 120) -> list[str]:
    return [s for s, _ in split_sentences_with_kind(paragraph, max_chars)]


def split_sentences_with_kind(paragraph: str, max_chars: int = 120) -> list[tuple[str, str]]:
    """。！？の後で文を区切る。括弧の内側では区切らず、長すぎる文だけ再分割する。

    kind は再分割前の文で決め、再分割した断片すべてに引き継ぐ。
    """
    sentences: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    n = len(paragraph)
    while i < n:
        ch = paragraph[i]
        buf.append(ch)
        if ch in OPEN_BRACKETS:
            depth += 1
        elif ch in CLOSE_BRACKETS:
            depth = max(0, depth - 1)
            # 「A」「B」のように台詞が連続する場合は区切る
            if depth == 0 and i + 1 < n and paragraph[i + 1] in DIALOGUE_OPEN:
                sentences.append("".join(buf))
                buf = []
        elif ch in SENTENCE_END and depth == 0:
            # 連続する句読点（！！、？！、…）と閉じ括弧を同じ文に含める
            while i + 1 < n and (
                paragraph[i + 1] in SENTENCE_END + "…" or paragraph[i + 1] in CLOSE_BRACKETS
            ):
                i += 1
                buf.append(paragraph[i])
            sentences.append("".join(buf))
            buf = []
        i += 1
    if buf:
        sentences.append("".join(buf))

    result: list[tuple[str, str]] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        kind = classify(s)
        result.extend((part, kind) for part in _split_long(s, max_chars))
    return result


_NO_SPEECH = set(CLOSE_BRACKETS + SENTENCE_END + "…、,，") | set(" \t　")


def _merge_unspoken(parts: list[str]) -> list[str]:
    """閉じ括弧・句読点・空白だけの断片は読み上げ本文を持たないので隣に併合する。"""
    merged: list[str] = []
    carry = ""
    for p in parts:
        if all(ch in _NO_SPEECH for ch in p):
            if merged:
                merged[-1] += p
            else:
                carry += p
        else:
            merged.append(carry + p)
            carry = ""
    if carry:
        merged.append(carry)
    return merged


def _split_long(sentence: str, max_chars: int) -> list[str]:
    return _merge_unspoken(_split_long_raw(sentence, max_chars))


def _split_long_raw(sentence: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("split.max_chars は 1 以上にしてください")
    if len(sentence) <= max_chars:
        return [sentence]
    # まず括弧内も含めた句点で、次に読点で、最後に文字数で割る
    for seps in (SENTENCE_END, "、,，"):
        parts = _cut_after(sentence, seps)
        if len(parts) > 1:
            out: list[str] = []
            for p in _pack(parts, max_chars):
                out.extend(_split_long_raw(p, max_chars) if len(p) > max_chars and p != sentence else [p])
            return out
    runs = emoji_runs(sentence)
    out = []
    i = 0
    while i < len(sentence):
        j = min(i + max_chars, len(sentence))
        if j < len(sentence):
            cut = safe_cut(sentence, j, runs)  # 絵文字の連なりの途中で切らない
            if i < cut <= i + max_chars:
                j = cut
        out.append(sentence[i:j])
        i = j
    return out


def _cut_after(text: str, seps: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in seps:
            parts.append("".join(buf))
            buf = []
    if buf:
        parts.append("".join(buf))
    return parts


def _pack(parts: list[str], max_chars: int) -> list[str]:
    """短い断片を max_chars 以内でまとめ直す。"""
    packed: list[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            packed.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        packed.append(cur)
    return packed


def classify(sentence: str) -> str:
    return "dialogue" if sentence[:1] in DIALOGUE_OPEN else "narration"


def is_separator(line: str) -> bool:
    """文字（かな・漢字・英数字）も括弧も絵文字も含まない行（＊ ◇　◇　◇ …… ―― など）。"""
    if _WORD.search(line) or any(ch in OPEN_BRACKETS + CLOSE_BRACKETS for ch in line):
        return False
    return not emoji_runs(line)


def build_chapters(
    text: str, chapter_regex: str, max_chars: int = 120, *, read_chapter_title: bool = True
) -> list[Chapter]:
    chapters: list[Chapter] = []
    for c_idx, (title, body) in enumerate(split_chapters(clean_text(text), chapter_regex)):
        paragraphs: list[list[Segment]] = []
        pending_break: str | None = None  # 本文の後ろに置く区切り（連続は最初の 1 つ）
        for para in split_paragraphs(body):
            if is_separator(para):
                if pending_break is None:
                    pending_break = para
                continue
            if pending_break is not None and paragraphs:  # 本文の先頭の区切りは捨てる
                paragraphs.append([Segment(c_idx, len(paragraphs), 0, "break", pending_break)])
            pending_break = None
            p_idx = len(paragraphs)
            segs = [
                Segment(c_idx, p_idx, s_idx, kind, s)
                for s_idx, (s, kind) in enumerate(split_sentences_with_kind(para, max_chars))
            ]
            if segs:
                paragraphs.append(segs)
        if not paragraphs:
            continue
        if read_chapter_title and title not in AUTO_TITLES:
            paragraphs.insert(0, [Segment(c_idx, 0, 0, "title", title)])
            for p_i, segs in enumerate(paragraphs):
                for s in segs:
                    s.para_idx = p_i
        chapters.append(Chapter(c_idx, title, paragraphs))
    return chapters
