from pathlib import Path

import pytest

from irodori_speaker.config import DEFAULT_CHAPTER_REGEX
from irodori_speaker.text_splitter import (
    build_chapters,
    clean_text,
    is_separator,
    read_text,
    split_chapters,
    split_sentences,
)

SAMPLE = Path(__file__).parent / "sample_novel.txt"


def test_clean_text_ruby_and_notes():
    assert clean_text("|紫陽花《あじさい》が咲く") == "あじさいが咲く"
    assert clean_text("漢字《かんじ》です") == "かんじです"
    assert clean_text("本文［＃改ページ］続き") == "本文続き"


def test_split_chapters_default_regex():
    chapters = split_chapters(read_text(SAMPLE), DEFAULT_CHAPTER_REGEX)
    assert [t for t, _ in chapters] == ["第一章 雨の朝", "第二章 約束"]


def test_no_heading_is_single_chapter():
    assert len(split_chapters("ただの文章。", DEFAULT_CHAPTER_REGEX)) == 1


def test_sentences_do_not_split_inside_quotes():
    assert split_sentences("「やあ。元気？」と彼は言った。次の文。") == [
        "「やあ。元気？」と彼は言った。",
        "次の文。",
    ]


def test_consecutive_punctuation_stays_together():
    assert split_sentences("やった！！ついに見つけた？！") == ["やった！！", "ついに見つけた？！"]


def test_consecutive_dialogue_is_split():
    assert split_sentences("「A」「B」") == ["「A」", "「B」"]


def test_long_sentence_is_split_at_commas():
    text = "あ" * 50 + "、" + "い" * 50 + "、" + "う" * 50 + "。"
    parts = split_sentences(text, max_chars=120)
    assert all(len(p) <= 120 for p in parts)
    assert "".join(parts) == text


def test_build_chapters_kinds():
    chapters = build_chapters(read_text(SAMPLE), DEFAULT_CHAPTER_REGEX)
    assert len(chapters) == 2
    first = [s for para in chapters[0].paragraphs for s in para]
    assert first[0].kind == "title"
    assert first[1].kind == "narration"
    assert any(s.kind == "dialogue" and s.text.startswith("「おはよう") for s in first)
    assert any("あじさい" in s.text for s in first)
    assert chapters[1].idx == 1


def test_long_dialogue_keeps_kind_for_all_fragments():
    text = "「" + "あ" * 50 + "。" + "い" * 50 + "。」"
    chapters = build_chapters(text, DEFAULT_CHAPTER_REGEX, max_chars=80)
    segs = [s for c in chapters for p in c.paragraphs for s in p]
    assert len(segs) > 1
    assert {s.kind for s in segs} == {"dialogue"}


def test_split_long_has_no_punctuation_only_fragment():
    sentence = "「" + "あ" * 78 + "。」"
    parts = split_sentences(sentence, max_chars=80)
    assert "".join(parts) == sentence
    for p in parts:
        assert p.strip("」』）)】。！？!?…、,，  ") != ""


def test_explicit_ruby_does_not_cross_newline():
    text = "窓の外に縦線｜が見えた。\n翌朝、漢字《かんじ》を読んだ。"
    assert clean_text(text) == "窓の外に縦線｜が見えた。\n翌朝、かんじを読んだ。"


def test_default_config_splits_long_sentence_at_commas():
    # 長い文は生成の後半で参照音声の声から離れるため、既定では 40 文字以内に分けて合成する
    from irodori_speaker.config import load_config

    cfg = load_config(None)
    sentence = "あ" * 15 + "、" + "い" * 15 + "、" + "う" * 15 + "、" + "え" * 15 + "。"
    chapters = build_chapters(sentence, cfg.split.chapter_regex, cfg.split.max_chars)
    parts = [s.text for c in chapters for p in c.paragraphs for s in p]
    assert len(parts) > 1
    assert all(len(p) <= 40 for p in parts)
    assert "".join(parts) == sentence
    assert all(p.endswith(("、", "。")) for p in parts)  # 読点・句点の位置で分ける


def test_文字数での分割は絵文字の連なりを切らない():
    from irodori_speaker.text_splitter import split_sentences

    for emoji in ("😠😠", "😮‍💨", "⏸️", "🌬️"):
        text = "あ" * 39 + emoji + "い" * 20
        parts = split_sentences(text, max_chars=40)
        assert len(parts) >= 2
        assert "".join(parts) == text
        assert any(emoji in p for p in parts)
        assert not any(p.startswith(("‍", "️")) for p in parts)
        assert not any(p.endswith("‍") for p in parts)
        for p in parts:
            assert emoji[0] not in p or emoji in p  # 連なりの片割れだけの断片が無い


def test_句読点も絵文字も無い長文の分割が速い():
    import time

    from irodori_speaker.text_splitter import split_sentences

    text = "あ" * 4000
    t0 = time.perf_counter()
    parts = split_sentences(text, max_chars=40)
    elapsed = time.perf_counter() - t0
    assert "".join(parts) == text
    assert all(len(p) <= 40 for p in parts)
    assert elapsed < 0.5, f"{elapsed:.2f}秒"


def test_max_chars_が_0_以下なら分割は拒否する():
    import signal

    import pytest

    from irodori_speaker.text_splitter import split_sentences

    def timeout(*_):
        raise AssertionError("分割が止まらない")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2)
    try:
        for n in (0, -1):
            with pytest.raises(ValueError, match="max_chars"):
                split_sentences("あ" * 100, max_chars=n)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------- 区切り行・章題 ----------


@pytest.mark.parametrize("line", ["＊", "◇　◇　◇", "* * *", "……", "――"])
def test_is_separator_true(line):
    assert is_separator(line)


@pytest.mark.parametrize("line", ["「……」", "1", "😊", "あ", "（……）", "A"])
def test_is_separator_false(line):
    assert not is_separator(line)


def _segs(chapter):
    return [(s.para_idx, s.kind, s.text) for p in chapter.paragraphs for s in p]


def test_break_lines_become_one_break_segment(tmp_path):
    for line in ["＊", "◇　◇　◇", "* * *", "……", "――"]:
        chapters = build_chapters(f"第一章 題\nあい。\n{line}\nうえ。\n", DEFAULT_CHAPTER_REGEX)
        assert _segs(chapters[0]) == [
            (0, "title", "第一章 題"),
            (1, "narration", "あい。"),
            (2, "break", line),
            (3, "narration", "うえ。"),
        ]


def test_consecutive_breaks_merge_and_edges_dropped():
    text = "第一章 題\n＊\nあい。\n＊\n◇\nうえ。\n＊\n"
    chapters = build_chapters(text, DEFAULT_CHAPTER_REGEX)
    assert _segs(chapters[0]) == [
        (0, "title", "第一章 題"),
        (1, "narration", "あい。"),
        (2, "break", "＊"),
        (3, "narration", "うえ。"),
    ]


def test_chapter_of_only_breaks_is_dropped():
    text = "第一章 A\n＊\n＊\n第二章 B\nあい。\n"
    chapters = build_chapters(text, DEFAULT_CHAPTER_REGEX)
    assert [c.title for c in chapters] == ["第二章 B"]


def test_auto_title_is_not_read():
    chapters = build_chapters("あい。\nうえ。\n", DEFAULT_CHAPTER_REGEX)
    assert chapters[0].title == "全編"
    assert _segs(chapters[0])[0] == (0, "narration", "あい。")


def test_read_chapter_title_false():
    chapters = build_chapters("第一章 題\nあい。\n", DEFAULT_CHAPTER_REGEX, read_chapter_title=False)
    assert _segs(chapters[0]) == [(0, "narration", "あい。")]


@pytest.mark.parametrize("line", ["😀", "❤️", "🌸", "☺️"])
def test_non_palette_emoji_line_is_not_separator(line):
    assert not is_separator(line)


@pytest.mark.parametrize("line", ["★", "☆", "◆", "※", "◇"])
def test_symbol_lines_are_separators(line):
    assert is_separator(line)


def test_regex_matched_auto_like_title_is_read():
    chapters = build_chapters("序\nあい。\n", r"^序$")
    assert _segs(chapters[0])[0] == (0, "title", "序")


def test_auto_titles_are_not_read():
    head = build_chapters("前置き。\n第一章 題\nあい。\n", DEFAULT_CHAPTER_REGEX)
    assert head[0].title == "序"
    assert _segs(head[0]) == [(0, "narration", "前置き。")]
    whole = build_chapters("あい。\n", DEFAULT_CHAPTER_REGEX)
    assert whole[0].title == "全編"
    assert _segs(whole[0]) == [(0, "narration", "あい。")]
