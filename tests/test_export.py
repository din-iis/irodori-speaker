from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import audio_io, cli, export, pipeline
from irodori_speaker.assemble import Clip, write_chapter
from irodori_speaker.audio_io import ffmpeg_available
from irodori_speaker.config import Config, ExportConfig, load_config
from irodori_speaker.export import ExportChapter, ExportInput

NOVEL = Path(__file__).parent / "sample_novel.txt"
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg が無い")
SR = 24000


def _tone(seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")


def make_book(tmp_path: Path) -> tuple[Config, Path]:
    cfg = Config(output_dir=str(tmp_path / "out"))
    out_dir = pipeline.book_dir(cfg, NOVEL)
    pipeline.build_script(NOVEL, cfg, out_dir, use_emotion=False)
    titles, rows = pipeline.load_book(out_dir)
    for idx in sorted({int(r["chapter"]) for r in rows}):
        clips = []
        for r in pipeline._sorted_rows([r for r in rows if int(r["chapter"]) == idx]):
            p = tmp_path / "segs" / f"{idx}_{r['para']}_{r['seg']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, _tone(0.4), SR)
            clips.append(Clip(p, idx, int(r["para"]), int(r["seg"])))
        wav = pipeline._chapter_path(out_dir, idx, titles)
        write_chapter(wav, clips, cfg.silence)
        pipeline._sig_path(wav).write_text(f"narration-sig-{idx}", encoding="utf-8")
    return cfg, out_dir


def make_bgm_mp3s(out_dir: Path) -> None:
    """bgm の章 MP3（朗読より長いステレオ）と .sig / .sounds.json を作る。"""
    titles, _ = pipeline.load_book(out_dir)
    for idx in sorted(titles):
        mp3 = out_dir / "bgm" / "chapters" / f"{pipeline._chapter_path(out_dir, idx, titles).stem}.mp3"
        stereo = np.stack([_tone(8.0), _tone(8.0)], axis=1)
        audio_io.write_mp3(stereo, SR, mp3, "96k")
        pipeline._sig_path(mp3).write_text(f"bgm-sig-{idx}", encoding="utf-8")
        pipeline._sounds_path(mp3).write_text("[]", encoding="utf-8")


def inputs_for(out_dir: Path, *, bgm: bool = False) -> tuple[list[ExportInput], list[dict]]:
    titles, rows = pipeline.load_book(out_dir)
    result = []
    for idx in sorted(titles):
        wav = pipeline._chapter_path(out_dir, idx, titles)
        mp3 = out_dir / "bgm" / "chapters" / f"{wav.stem}.mp3" if bgm else None
        result.append(ExportInput(idx, titles[idx], wav, mp3))
    return result, rows


def ch(idx: int, title: str = "t", frames: int = 24000, sr: int = 24000) -> ExportChapter:
    return ExportChapter(idx, title, Path(f"/x/{idx}.wav"), sr, frames, f"sig{idx}")


# ---------- 純粋: 章題・時刻・ffmetadata ----------


def test_chapter_title_is_raw_or_default():
    assert export.chapter_title(0, "第一章 雨の朝") == "第一章 雨の朝"
    assert export.chapter_title(2, "") == "第3章"
    assert export.chapter_title(1, "  ") == "第2章"


def test_chapter_starts_accumulate_in_ms():
    chapters = [ch(0, frames=24000), ch(1, frames=36000), ch(2, frames=12000)]
    assert export.chapter_starts_ms(chapters) == [0, 1000, 2500, 3000]


def test_ffmetadata_has_header_and_chapter_blocks():
    text = export.ffmetadata("本", [ch(0, "第一章", 24000), ch(1, "", 12000)])
    lines = text.splitlines()
    assert lines[0] == ";FFMETADATA1"
    assert "title=本" in lines and "album=本" in lines
    assert text.count("[CHAPTER]") == 2
    assert "TIMEBASE=1/1000" in lines
    assert "START=0" in lines and "END=1000" in lines
    assert "START=1000" in lines and "END=1500" in lines
    assert "title=第一章" in lines and "title=第2章" in lines


def test_escape_ffmeta_escapes_specials():
    assert export.escape_ffmeta("a=b;c#d\\e") == "a\\=b\\;c\\#d\\\\e"
    assert export.escape_ffmeta("1行目\n2行目") == "1行目\\\n2行目"


# ---------- 純粋: パス・署名 ----------


@pytest.mark.parametrize(
    "source,fmt,full,chap",
    [
        ("narration", "m4b", "export/book.m4b", None),
        ("bgm", "m4b", "export/book_bgm.m4b", None),
        ("narration", "mp3", "export/book.mp3", "export/chapters"),
        ("bgm", "mp3", "export/book_bgm.mp3", "export/chapters_bgm"),
    ],
)
def test_export_paths(tmp_path, source, fmt, full, chap):
    out_dir = tmp_path / "book"
    got_full, got_chap = export.export_paths(out_dir, source=source, fmt=fmt)
    assert got_full == out_dir / full
    assert got_chap == (out_dir / chap if chap else None)


def test_chapter_mp3_path_uses_number_and_safe_title(tmp_path):
    assert export.chapter_mp3_path(tmp_path, ch(0, "第一章 雨/の朝")) == tmp_path / "01_第一章_雨_の朝.mp3"
    assert export.chapter_mp3_path(tmp_path, ch(1, "")).name == "02_第2章.mp3"


def test_export_signature_changes_with_each_input():
    base = dict(source="narration", fmt="m4b", bitrate="64k", channels=1)
    chapters = [ch(0), ch(1)]
    sig = export.export_signature(chapters, **base)
    assert sig == export.export_signature(chapters, **base)
    assert sig != export.export_signature([ch(0), replace(ch(1), sig="other")], **base)
    assert sig != export.export_signature([ch(0), ch(1, title="改題")], **base)
    assert sig != export.export_signature(chapters, **{**base, "bitrate": "96k"})
    assert sig != export.export_signature(chapters, **{**base, "fmt": "mp3"})
    assert sig != export.export_signature(chapters, **{**base, "source": "bgm"})
    assert sig != export.export_signature(chapters, **{**base, "channels": 2})
    assert sig != export.export_signature(chapters, **base, book="別の題名")


# ---------- collect_chapters ----------


def test_collect_chapters_reads_frames_and_sig(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    chapters = export.collect_chapters(inputs, rows, source="narration")
    assert [c.idx for c in chapters] == [0, 1]
    assert [c.title for c in chapters] == ["第一章 雨の朝", "第二章 約束"]
    assert chapters[0].sr == SR
    assert chapters[0].frames == sf.info(inputs[0].wav).frames
    assert chapters[0].sig == "narration-sig-0"


def test_collect_rejects_missing_wav(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    inputs[1].wav.unlink()
    with pytest.raises(ValueError, match="synth --chapters 2"):
        export.collect_chapters(inputs, rows, source="narration")


def test_collect_rejects_missing_sig(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    pipeline._sig_path(inputs[0].wav).unlink()
    with pytest.raises(ValueError, match="synth --chapters 1"):
        export.collect_chapters(inputs, rows, source="narration")


def test_collect_rejects_frames_mismatch(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    sf.write(inputs[0].wav, _tone(0.1), SR)  # .times.json の frames と合わなくなる
    with pytest.raises(ValueError, match="synth --chapters 1"):
        export.collect_chapters(inputs, rows, source="narration")


def test_collect_rejects_paragraph_mismatch(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    rows = [r for r in rows if not (int(r["chapter"]) == 0 and int(r["para"]) == 0)]
    with pytest.raises(ValueError, match="synth --chapters 1"):
        export.collect_chapters(inputs, rows, source="narration")


def test_collect_rejects_sample_rate_mismatch(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    titles, _ = pipeline.load_book(out_dir)
    # 第 2 章だけ 16kHz で作り直す
    clips = []
    for r in pipeline._sorted_rows([r for r in rows if int(r["chapter"]) == 1]):
        p = tmp_path / "segs16" / f"{r['para']}_{r['seg']}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, _tone(0.4, 16000), 16000)
        clips.append(Clip(p, 1, int(r["para"]), int(r["seg"])))
    write_chapter(inputs[1].wav, clips, Config().silence)
    with pytest.raises(ValueError, match="サンプリングレート"):
        export.collect_chapters(inputs, rows, source="narration")


@pytest.mark.parametrize("missing", ["mp3", "sig", "sounds"])
def test_collect_bgm_requires_mp3_sig_and_sounds(tmp_path, missing):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir, bgm=True)
    mp3 = inputs[0].bgm_mp3
    mp3.parent.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(b"x")
    pipeline._sig_path(mp3).write_text("s", encoding="utf-8")
    pipeline._sounds_path(mp3).write_text("[]", encoding="utf-8")
    {"mp3": mp3, "sig": pipeline._sig_path(mp3), "sounds": pipeline._sounds_path(mp3)}[missing].unlink()
    with pytest.raises(ValueError, match="bgm --chapters 1"):
        export.collect_chapters(inputs, rows, source="bgm")


@needs_ffmpeg
def test_collect_bgm_uses_mp3_frames_not_wav(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    make_bgm_mp3s(out_dir)
    inputs, rows = inputs_for(out_dir, bgm=True)
    chapters = export.collect_chapters(inputs, rows, source="bgm")
    assert chapters[0].src == inputs[0].bgm_mp3
    assert chapters[0].frames == sf.info(inputs[0].bgm_mp3).frames
    assert chapters[0].frames > sf.info(inputs[0].wav).frames
    assert chapters[0].sig == "bgm-sig-0"


# ---------- 設定 ----------


def test_export_config_defaults_and_yaml(tmp_path):
    assert ExportConfig() == ExportConfig(format="m4b", bitrate="64k", title=None)
    assert Config().export == ExportConfig()
    conf = tmp_path / "c.yaml"
    conf.write_text("export:\n  format: mp3\n  bitrate: 96k\n  title: 題名\n", encoding="utf-8")
    cfg = load_config(conf)
    assert (cfg.export.format, cfg.export.bitrate, cfg.export.title) == ("mp3", "96k", "題名")
    assert load_config("config.example.yaml").export == ExportConfig()


@pytest.mark.parametrize("body", ["format: wav", "bitrate: 64", "bitrate: 64kbps", "bitrate: k"])
def test_export_config_rejects_bad_values(tmp_path, body):
    conf = tmp_path / "c.yaml"
    conf.write_text(f"export:\n  {body}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="export"):
        load_config(conf)


# ---------- CLI ----------


def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    conf = tmp_path / "config.yaml"
    conf.write_text(f"output_dir: {tmp_path / 'out'}\n", encoding="utf-8")
    return ["export", str(NOVEL), "-c", str(conf), *extra]


def test_cli_export_passes_options_and_reports(tmp_path, monkeypatch, capsys):
    seen: dict = {}
    out = tmp_path / "x.mp3"
    out.write_bytes(b"\0" * 2_000_000)

    def fake(cfg, out_dir, **kw):
        seen.update(kw)
        return pipeline.ExportResult(out, [], False)

    monkeypatch.setattr(cli, "build_export", fake)
    assert cli.main(_cli_args(tmp_path, "--format", "mp3", "--bitrate", "96k", "--bgm")) == 0
    assert (seen["source"], seen["fmt"], seen["bitrate"]) == ("bgm", "mp3", "96k")
    text = capsys.readouterr().out
    assert str(out) in text and "MB" in text


def test_cli_export_defaults_to_narration(tmp_path, monkeypatch):
    seen: dict = {}

    def fake(cfg, out_dir, **kw):
        seen.update(kw)
        return pipeline.ExportResult(tmp_path / "none.m4b", [], True)

    monkeypatch.setattr(cli, "build_export", fake)
    assert cli.main(_cli_args(tmp_path)) == 0
    assert (seen["source"], seen["fmt"], seen["bitrate"]) == ("narration", None, None)


def test_cli_export_skipped_says_no_change(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_export", lambda *a, **k: pipeline.ExportResult(tmp_path / "n.m4b", [], True))
    cli.main(_cli_args(tmp_path))
    assert "変更なし" in capsys.readouterr().out


@pytest.mark.parametrize("exc", [ValueError("先に synth を実行してください"), RuntimeError("ffmpeg が失敗しました")])
def test_cli_export_errors_exit_1(tmp_path, monkeypatch, capsys, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(cli, "build_export", boom)
    assert cli.main(_cli_args(tmp_path)) == 1
    assert str(exc) in capsys.readouterr().out


def test_cli_export_help_is_japanese_and_has_no_chapters(capsys):
    with pytest.raises(SystemExit):
        cli.main(["export", "--help"])
    text = capsys.readouterr().out
    assert "--format" in text and "--bitrate" in text and "--bgm" in text
    assert "--chapters" not in text
    assert re.search(r"[ぁ-んァ-ン一-龥]", text)


# ---------- WebUI ----------


def _handlers(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    from irodori_speaker import synth, webui

    monkeypatch.setattr(synth, "release_runtime", lambda: None)
    app = webui.build_app(replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out")))
    return {f.fn.__name__: f.fn for f in app.fns.values()}


def test_webui_export_book_success(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    from irodori_speaker import webui

    seen: dict = {}
    out = tmp_path / "b.m4b"

    def fake(cfg, out_dir, **kw):
        seen.update(kw, out_dir=out_dir)
        return pipeline.ExportResult(out, [], False)

    monkeypatch.setattr(webui, "build_export", fake)
    path, status = fns["export_book"](None, str(NOVEL), "mp3", True)
    assert path == str(out) and "書き出しました" in status
    assert (seen["source"], seen["fmt"]) == ("bgm", "mp3")
    assert seen["out_dir"].name == NOVEL.stem


def test_webui_export_book_narration_when_unchecked(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    from irodori_speaker import webui

    seen: dict = {}
    monkeypatch.setattr(
        webui, "build_export", lambda cfg, d, **kw: seen.update(kw) or pipeline.ExportResult(tmp_path / "a", [], False)
    )
    fns["export_book"](None, str(NOVEL), "m4b", False)
    assert seen["source"] == "narration"


def test_webui_export_book_error_becomes_gr_error(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    import gradio as gr

    from irodori_speaker import webui

    def boom(*a, **k):
        raise ValueError("synth --chapters 1 を実行してください")

    monkeypatch.setattr(webui, "build_export", boom)
    with pytest.raises(gr.Error, match="synth --chapters 1"):
        fns["export_book"](None, str(NOVEL), "m4b", False)


# ---------- ffmpeg ----------


def _chapter_tags(path: Path) -> list[tuple[float, str]]:
    meta = audio_io._run(["-i", str(path), "-f", "ffmetadata", "-"]).decode("utf-8")
    result = []
    for block in meta.split("[CHAPTER]")[1:]:
        num, den = re.search(r"TIMEBASE=(\d+)/(\d+)", block).groups()
        start = int(re.search(r"START=(\d+)", block).group(1)) * int(num) * 1000 / int(den)
        title = re.search(r"title=(.*)", block).group(1)
        result.append((start, title))
    return result


def _sample_count(path: Path, channels: int) -> int:
    raw = audio_io._run(["-i", str(path), "-ar", "24000", "-ac", str(channels), "-f", "f32le", "-"])
    return len(raw) // 4 // channels


@needs_ffmpeg
def test_build_export_m4b_has_chapters_and_mono(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    result = pipeline.build_export(cfg, out_dir)
    assert result.path == out_dir / "export" / f"{out_dir.name}.m4b"
    assert result.path.exists() and not result.skipped and result.chapter_files == []
    inputs, rows = inputs_for(out_dir)
    chapters = export.collect_chapters(inputs, rows, source="narration")
    tags = _chapter_tags(result.path)
    assert [t for _, t in tags] == ["第一章 雨の朝", "第二章 約束"]
    expected = export.chapter_starts_ms(chapters)
    for (start, _), want in zip(tags, expected):
        assert abs(start - want) <= 50
    total = sum(c.frames for c in chapters)
    assert abs(_sample_count(result.path, 1) - total) < total * 0.05
    assert not list((out_dir / "export").glob("*.tmp*"))


@needs_ffmpeg
def test_build_export_mp3_writes_chapters_and_full(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    result = pipeline.build_export(cfg, out_dir, fmt="mp3", bitrate="48k")
    assert result.path == out_dir / "export" / f"{out_dir.name}.mp3"
    assert [p.name for p in result.chapter_files] == ["01_第一章_雨の朝.mp3", "02_第二章_約束.mp3"]
    assert all(p.exists() for p in [result.path, *result.chapter_files])


@needs_ffmpeg
def test_build_export_bgm_is_stereo_from_bgm_mp3(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    make_bgm_mp3s(out_dir)
    result = pipeline.build_export(cfg, out_dir, source="bgm")
    assert result.path.name == f"{out_dir.name}_bgm.m4b"
    total = sum(sf.info(m).frames for m in (out_dir / "bgm" / "chapters").glob("*.mp3"))
    assert abs(_sample_count(result.path, 2) - total) < total * 0.05
    starts = [s for s, _ in _chapter_tags(result.path)]
    first = sf.info(next((out_dir / "bgm" / "chapters").glob("01_*.mp3"))).frames
    assert abs(starts[1] - first / SR * 1000) <= 50


@needs_ffmpeg
def test_build_export_skips_when_signature_matches(tmp_path, monkeypatch):
    cfg, out_dir = make_book(tmp_path)
    pipeline.build_export(cfg, out_dir)

    def boom(*a, **k):
        raise AssertionError("再エンコードしてはいけない")

    monkeypatch.setattr(audio_io, "write_m4b", boom)
    again = pipeline.build_export(cfg, out_dir)
    assert again.skipped


@needs_ffmpeg
def test_build_export_reencodes_when_title_changes(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    pipeline.build_export(cfg, out_dir)
    cfg2 = replace(cfg, export=replace(cfg.export, title="別の題名"))
    assert not pipeline.build_export(cfg2, out_dir).skipped


@needs_ffmpeg
def test_build_export_failure_removes_output_and_sig(tmp_path, monkeypatch):
    cfg, out_dir = make_book(tmp_path)
    first = pipeline.build_export(cfg, out_dir)
    pipeline._sig_path(pipeline._chapter_path(out_dir, 0, pipeline.load_book(out_dir)[0])).write_text(
        "changed", encoding="utf-8"
    )

    def boom(*a, **k):
        raise RuntimeError("ffmpeg が失敗しました")

    monkeypatch.setattr(audio_io, "write_m4b", boom)
    with pytest.raises(RuntimeError):
        pipeline.build_export(cfg, out_dir)
    assert not first.path.exists()
    assert not pipeline._sig_path(first.path).exists()


def test_build_export_rejects_missing_synth(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    pipeline._chapter_path(out_dir, 0, pipeline.load_book(out_dir)[0]).unlink()
    with pytest.raises(ValueError, match="synth --chapters 1"):
        pipeline.build_export(cfg, out_dir)


@needs_ffmpeg
def test_encode_file_leaves_no_tmp(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(src, _tone(0.5), SR)
    dst = tmp_path / "o" / "a.mp3"
    audio_io.encode_file(src, dst, ["-b:a", "48k", "-f", "mp3"])
    assert dst.exists()
    assert [p.name for p in dst.parent.iterdir()] == ["a.mp3"]


@needs_ffmpeg
def test_encode_file_failure_leaves_no_tmp(tmp_path):
    src = tmp_path / "missing.wav"
    dst = tmp_path / "o" / "a.mp3"
    with pytest.raises(RuntimeError):
        audio_io.encode_file(src, dst, ["-f", "mp3"])
    assert not dst.exists()
    assert not list(dst.parent.glob("*tmp*"))


# ---------- 追加修正 1: 入力音声の更新を検出する ----------


def test_export_signature_changes_with_input_stamp():
    base = dict(source="narration", fmt="m4b", bitrate="64k", channels=1)
    a = [ch(0), ch(1)]
    assert export.export_signature(a, **base) != export.export_signature(
        [a[0], replace(a[1], stamp=(10, 20))], **base
    )


def test_collect_chapters_stamps_input_file(tmp_path):
    _cfg, out_dir = make_book(tmp_path)
    inputs, rows = inputs_for(out_dir)
    before = export.collect_chapters(inputs, rows, source="narration")
    st = inputs[0].wav.stat()
    assert before[0].stamp == (st.st_size, st.st_mtime_ns)
    # .sig は同じまま、WAV の mtime だけ変わる（seed: null の synth --force 相当）
    import os

    os.utime(inputs[0].wav, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    after = export.collect_chapters(inputs, rows, source="narration")
    base = dict(source="narration", fmt="m4b", bitrate="64k", channels=1)
    assert export.export_signature(before, **base) != export.export_signature(after, **base)


@needs_ffmpeg
def test_build_export_reencodes_when_wav_rewritten_with_same_sig(tmp_path):
    import os

    cfg, out_dir = make_book(tmp_path)
    pipeline.build_export(cfg, out_dir)
    wav = pipeline._chapter_path(out_dir, 0, pipeline.load_book(out_dir)[0])
    st = wav.stat()
    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert not pipeline.build_export(cfg, out_dir).skipped
    assert pipeline.build_export(cfg, out_dir).skipped


# ---------- 追加修正 2: mp3 の失敗時に章ファイルも消す ----------


@needs_ffmpeg
def test_build_export_mp3_failure_removes_chapter_files(tmp_path, monkeypatch):
    cfg, out_dir = make_book(tmp_path)
    real = audio_io.encode_file
    calls = []

    def flaky(src, dst, args):
        calls.append(dst)
        if len(calls) == 2:
            raise RuntimeError("ffmpeg が失敗しました")
        real(src, dst, args)

    monkeypatch.setattr(audio_io, "encode_file", flaky)
    with pytest.raises(RuntimeError):
        pipeline.build_export(cfg, out_dir, fmt="mp3")
    full, chapter_dir = export.export_paths(out_dir, source="narration", fmt="mp3")
    assert not calls[0].exists()
    assert not full.exists() and not pipeline._sig_path(full).exists()
    assert not list(chapter_dir.glob("*.mp3"))
