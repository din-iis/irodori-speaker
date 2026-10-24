from pathlib import Path

from irodori_speaker.config import DEFAULT_CHAPTER_REGEX
from irodori_speaker.text_splitter import (
    build_chapters,
    clean_text,
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
    assert first[0].kind == "narration"
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
