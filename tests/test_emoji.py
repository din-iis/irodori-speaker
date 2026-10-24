import pytest

from irodori_speaker.config import DEFAULT_EMOTION_EMOJI
from irodori_speaker.emoji import IRODORI_EMOJI, emoji_runs, safe_cut


def test_emoji_runs_は連続する同じ絵文字を1区間にする():
    assert emoji_runs("あ😠😠い") == [(1, 3)]


def test_emoji_runs_は異なる絵文字の連続も1区間にする():
    assert emoji_runs("😠😊あ") == [(0, 2)]


def test_emoji_runs_は_ZWJ_の絵文字を途中で切らない():
    text = "あ😮‍💨い"
    assert emoji_runs(text) == [(1, 1 + len("😮‍💨"))]


def test_emoji_runs_は_ZWJ_の絵文字を短い絵文字より優先する():
    assert emoji_runs("😮‍💨😮") == [(0, len("😮‍💨😮"))]
    assert len(emoji_runs("😮‍💨あ😮")) == 2


@pytest.mark.parametrize("emoji", ["⏸", "⏸️", "🌬", "🌬️"])
def test_emoji_runs_は異体字セレクタの有無どちらも認識する(emoji):
    assert emoji_runs("あ" + emoji + "い") == [(1, 1 + len(emoji))]


@pytest.mark.parametrize("ch", ["℃", "㎏", "♪", "★", "〈", "、", "。", "…", "「"])
def test_emoji_runs_は_Irodori_の絵文字以外を区間にしない(ch):
    assert emoji_runs(f"あ{ch}い") == []


def test_safe_cut_は連なりの内側なら前で切る():
    assert safe_cut("あああ😠😠い", 4) == 3


def test_safe_cut_は連なりが先頭なら後ろで切る():
    assert safe_cut("😠😠いい", 1) == 2


def test_safe_cut_はそれ以外ならそのまま():
    text = "あああ😠😠い"
    assert safe_cut(text, 2) == 2
    assert safe_cut(text, 3) == 3  # 連なりの直前
    assert safe_cut(text, 5) == 5  # 連なりの直後


def test_既定の感情絵文字はすべて_Irodori_の絵文字():
    for variants in DEFAULT_EMOTION_EMOJI.values():
        for v in variants:
            assert emoji_runs(v) == ([(0, len(v))] if v else [])


def test_IRODORI_EMOJI_は45種():
    assert len(IRODORI_EMOJI) == 45


@pytest.mark.parametrize("pos", [0, 2, 3, 4, 5, 6, 7])
def test_safe_cut_は_runs_を渡しても同じ結果(pos):
    text = "あああ😠😠い"
    assert safe_cut(text, pos, emoji_runs(text)) == safe_cut(text, pos)
