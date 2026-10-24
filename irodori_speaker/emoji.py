"""Irodori-TTS が解釈する絵文字の一覧と、本文中での判定・切断位置の調整。"""

from __future__ import annotations

# 上流 irodori_tts/gradio_emoji_palette.py の EMOJI_PALETTE_ITEMS と同じ 45 種
IRODORI_EMOJI: tuple[str, ...] = (
    "👂", "😮‍💨", "⏸️", "🤭", "🥵", "📢", "😏", "🥺", "🌬️", "😮",
    "👅", "💋", "🫶", "😭", "😱", "😪", "😴", "⏩", "📞", "🐢",
    "🥤", "🤧", "😒", "😰", "😆", "💥", "😠", "😲", "🥱", "😖",
    "😟", "🫣", "🙄", "😊", "😎", "👌", "🙏", "🥴", "🎵", "🤐",
    "😌", "🤔", "💪", "👃", "📖",
)

_VS16 = "️"


def _forms() -> list[str]:
    """異体字セレクタの有無どちらも受け付け、長い並びを先に試す（最長一致）。"""
    forms: set[str] = set()
    for e in IRODORI_EMOJI:
        base = e.replace(_VS16, "")
        forms.add(e)
        forms.add(base)
        if len(base) == 1:
            forms.add(base + _VS16)
    return sorted(forms, key=len, reverse=True)


_FORMS = _forms()
_FIRST_CHARS = frozenset(f[0] for f in _FORMS)


def emoji_runs(text: str) -> list[tuple[int, int]]:
    """IRODORI_EMOJI が 1 つ以上連続した区間 (start, end) を前から順に返す。"""
    runs: list[tuple[int, int]] = []
    start = None
    i = 0
    while i < len(text):
        if text[i] not in _FIRST_CHARS:
            if start is not None:
                runs.append((start, i))
                start = None
            i += 1
            continue
        for form in _FORMS:
            if text.startswith(form, i):
                if start is None:
                    start = i
                i += len(form)
                break
        else:
            if start is not None:
                runs.append((start, i))
                start = None
            i += 1
    if start is not None:
        runs.append((start, len(text)))
    return runs


def safe_cut(text: str, pos: int, runs: list[tuple[int, int]] | None = None) -> int:
    """pos が絵文字の連なりの内側なら、連なりの前（先頭の連なりなら後ろ）に切断位置を寄せる。

    runs に emoji_runs(text) の結果を渡すと、繰り返し呼ぶときに全文の走査を省ける。
    """
    for start, end in emoji_runs(text) if runs is None else runs:
        if start < pos < end:
            return start if start > 0 else end
    return pos
