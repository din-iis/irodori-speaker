"""WebUI の「BGM を付ける」ボタン。純粋ヘルパーと、ハンドラ経由の振る舞い。"""

from dataclasses import replace
from pathlib import Path

import pytest

from irodori_speaker.config import load_config
from irodori_speaker.pipeline import BgmResult
from irodori_speaker.webui import bgm_error_message, bgm_summary

pytest.importorskip("requests")


def test_bgm_error_message_KeyMissing_は案内文をそのまま出す():
    from irodori_speaker.freesound import KeyMissing

    assert "secrets.env の作り方" in bgm_error_message(KeyMissing("secrets.env の作り方"))


def test_bgm_error_message_DailyLimit_は翌日の案内():
    from irodori_speaker.freesound import DailyLimit

    msg = bgm_error_message(DailyLimit())
    assert "1 日の上限" in msg and "翌日" in msg


def test_bgm_error_message_ImportError_は_transformers_の案内():
    msg = bgm_error_message(ImportError("No module named 'transformers'"))
    assert "transformers" in msg and "--extra bgm" in msg
    assert "No module named" in msg


@pytest.mark.parametrize("exc", [ValueError("先に synth してください"), RuntimeError("壊れた")])
def test_bgm_error_message_ValueError_と_RuntimeError_はそのまま(exc):
    assert bgm_error_message(exc) == str(exc)


def test_bgm_summary_通しがあれば場所を出す(tmp_path):
    full = tmp_path / "bgm" / "book_bgm_full.mp3"
    msg = bgm_summary(BgmResult([tmp_path / "a.mp3"], full, 0), tmp_path)
    assert str(full) in msg
    assert "通しは未作成" not in msg
    assert "0 件" in msg
    assert str(tmp_path / "bgm" / "CREDITS.txt") in msg


def test_bgm_summary_通しが無ければ一部の章だけと伝える(tmp_path):
    msg = bgm_summary(BgmResult([tmp_path / "a.mp3"], None, 3), tmp_path)
    assert "通しは未作成" in msg
    assert "3 件" in msg
    assert str(tmp_path / "bgm" / "missing.tsv") in msg


# -- ハンドラ経由 --


def _setup(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    from irodori_speaker import synth, webui

    events = []
    monkeypatch.setattr(synth, "release_runtime", lambda: events.append("release"))
    app = webui.build_app(replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out")))
    fns = {f.fn.__name__: f.fn for f in app.fns.values()}
    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    return webui, fns, events, novel


def _run(fns, novel, chapters="2-3"):
    return list(fns["add_bgm"](None, str(novel), chapters, "ollama", "http://x", "m", True, 2))


def test_add_bgm_は最初に_TTS_を解放し章の範囲と_LLM_設定を渡す(monkeypatch, tmp_path):
    webui, fns, events, novel = _setup(monkeypatch, tmp_path)
    seen = {}

    def fake(cfg, out_dir, **kw):
        events.append("bgm")
        seen.update(kw)
        kw["progress"](1, 4, "場面")
        return BgmResult([], None, 0)

    monkeypatch.setattr(webui, "build_bgm", fake)
    outs = _run(fns, novel)
    assert events == ["release", "bgm"]
    assert seen["chapters"] == webui.parse_chapter_range("2-3")
    assert seen["llm_overrides"] == webui.llm_overrides_from_ui("ollama", "http://x", "m", True, 2)
    assert "release" in outs[0][0] or "解放" in outs[0][0]
    assert any("🎵 BGM: 場面（1/4）" in o[0] for o in outs)


def test_add_bgm_は最後に通しと章の_MP3_を返す(monkeypatch, tmp_path):
    webui, fns, events, novel = _setup(monkeypatch, tmp_path)
    ch, full = tmp_path / "c.mp3", tmp_path / "full.mp3"
    monkeypatch.setattr(webui, "build_bgm", lambda *a, **k: BgmResult([ch], full, 1))
    status, audio, files = _run(fns, novel, "")[-1]
    assert audio == str(full)
    assert files == [str(ch)]
    assert "1 件" in status


def test_add_bgm_は_KeyMissing_で_gr_Error(monkeypatch, tmp_path):
    gr = pytest.importorskip("gradio")
    from irodori_speaker.freesound import KeyMissing

    webui, fns, events, novel = _setup(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise KeyMissing("キーの作り方")

    monkeypatch.setattr(webui, "build_bgm", boom)
    with pytest.raises(gr.Error, match="キーの作り方"):
        _run(fns, novel)
