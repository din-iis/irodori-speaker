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
