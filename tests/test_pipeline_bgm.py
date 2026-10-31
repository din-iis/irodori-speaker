from __future__ import annotations

import contextlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import bgm_mix, cli, pipeline
from irodori_speaker.assemble import Clip, write_chapter
from irodori_speaker.audio_io import ffmpeg_available
from irodori_speaker.bgm_scenes import ScenePlanner
from irodori_speaker.bgm_select import Gate, load_selection
from irodori_speaker.bgm_types import Sound
from irodori_speaker.config import BgmConfig, Config
from irodori_speaker.freesound import DailyLimit, KeyMissing
from irodori_speaker.pipeline import BgmDeps, BgmResult, build_bgm

NOVEL = Path(__file__).parent / "sample_novel.txt"
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg が無い")
SR = 24000


# ---------- 偽物 ----------


def _tone(seconds: float, sr: int) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")


def _sound(sid: int) -> Sound:
    return Sound(
        id=sid, name=f"sound {sid}", tags=["t"], description="d", duration=60.0,
        license="https://creativecommons.org/licenses/by/4.0/", username="alice",
        url=f"https://freesound.org/s/{sid}/", preview=f"https://cdn.example/{sid}.mp3",
    )


class FakeClient:
    def __init__(self, calls: list, error: Exception | None = None):
        self.calls = calls
        self.error = error

    def search(self, query, duration, n):
        self.calls.append(("search", query))
        if self.error is not None:
            raise self.error
        return [_sound({"rain": 1, "piano": 2}[query])]


class FakeScorer:
    model = "fake-clap"

    def score(self, path, duration, text):
        return 0.5


def scene_chat(system, user, schema):
    return {"scenes": [{
        "start_para": 0, "description": "rainy street", "ambient_query": "rain",
        "mood": "calm", "music_query": "piano",
    }]}


def gate_chat(cfg, system, user, schema, name):
    ids = [int(i) for i in re.findall(r"id=(\d+) \|", user)]
    return {"results": [{"id": i, "accept": True, "reason": "ok"} for i in ids]}


def make_deps(calls: list, *, client_error: Exception | None = None) -> BgmDeps:
    @contextlib.contextmanager
    def server(cfg):
        calls.append("server_enter")
        try:
            yield
        finally:
            calls.append("server_exit")

    def scorer(bgm):
        calls.append("scorer_init")
        return FakeScorer()

    def download(url, dst):
        calls.append("download")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, _tone(3.0, 44100), 44100, format="WAV")

    return BgmDeps(
        client_factory=lambda key: FakeClient(calls, client_error),
        planner_factory=lambda llm, path: ScenePlanner(llm, path, chat=scene_chat),
        gate_factory=lambda llm, path: Gate(llm, path, chat=gate_chat),
        scorer_factory=scorer,
        server_factory=server,
        unload=lambda llm: calls.append("unload"),
        download=download,
    )


def make_book(tmp_path: Path, *, key: bool = True, times: bool = True) -> tuple[Config, Path]:
    secrets = tmp_path / "secrets.env"
    if key:
        secrets.write_text("FREESOUND_API_KEY=testkey\n", encoding="utf-8")
    cfg = Config(output_dir=str(tmp_path / "out"))
    cfg = replace(cfg, bgm=replace(cfg.bgm, secrets_file=str(secrets)))
    out_dir = pipeline.book_dir(cfg, NOVEL)
    pipeline.build_script(NOVEL, cfg, out_dir, use_emotion=False)
    titles, rows = pipeline.load_book(out_dir)
    for idx in sorted({int(r["chapter"]) for r in rows}):
        clips = []
        for r in pipeline._sorted_rows([r for r in rows if int(r["chapter"]) == idx]):
            p = tmp_path / "segs" / f"{idx}_{r['para']}_{r['seg']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, _tone(0.4, SR), SR)
            clips.append(Clip(p, idx, int(r["para"]), int(r["seg"])))
        wav = pipeline._chapter_path(out_dir, idx, titles)
        write_chapter(wav, clips, cfg.silence)
        pipeline._sig_path(wav).write_text(f"narration-sig-{idx}", encoding="utf-8")
        if not times:
            wav.with_suffix(".times.json").unlink()
    return cfg, out_dir


# ---------- build_bgm ----------


@needs_ffmpeg
def test_build_bgm_end_to_end_with_fakes(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    calls: list = []
    result = build_bgm(cfg, out_dir, deps=make_deps(calls))
    bgm = out_dir / "bgm"
    assert isinstance(result, BgmResult)
    assert len(result.chapter_files) == 2
    assert all(p.suffix == ".mp3" and p.exists() for p in result.chapter_files)
    assert {p.parent for p in result.chapter_files} == {bgm / "chapters"}
    assert result.full_path == bgm / f"{out_dir.name}_bgm_full.mp3"
    assert result.full_path.exists()
    assert result.missing == 0
    for name in ("CREDITS.txt", "missing.tsv", "scenes.json", "selection.json", "gate.tsv"):
        assert (bgm / name).exists(), name
    assert "sound 1 by alice" in (bgm / "CREDITS.txt").read_text(encoding="utf-8")
    for p in result.chapter_files:
        assert p.with_name(p.name + ".sig").exists()
    assert not (bgm / "tmp").exists() or not list((bgm / "tmp").iterdir())  # ダウンロードした音は消す


@needs_ffmpeg
def test_llm_unloaded_before_scorer_created(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    calls: list = []
    build_bgm(cfg, out_dir, deps=make_deps(calls))
    assert calls.index("server_exit") < calls.index("scorer_init")
    assert calls.index("unload") < calls.index("scorer_init")
    assert calls.index("server_enter") < calls.index("unload")
    assert calls.count("scorer_init") == 1


def test_missing_times_stops_before_llm(tmp_path):
    cfg, out_dir = make_book(tmp_path, times=False)
    calls: list = []
    with pytest.raises(ValueError, match="synth --chapters 1"):
        build_bgm(cfg, out_dir, deps=make_deps(calls))
    assert calls == []


def test_missing_key_stops_before_llm(tmp_path):
    cfg, out_dir = make_book(tmp_path, key=False)
    calls: list = []
    with pytest.raises(KeyMissing):
        build_bgm(cfg, out_dir, deps=make_deps(calls))
    assert calls == []


def test_no_target_chapters_raises(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    with pytest.raises(ValueError, match="対象の章がありません"):
        build_bgm(cfg, out_dir, chapters={9}, deps=make_deps([]))


@needs_ffmpeg
def test_rerun_skips_mixing_when_signature_matches(tmp_path, monkeypatch):
    cfg, out_dir = make_book(tmp_path)
    calls: list = []
    first = build_bgm(cfg, out_dir, deps=make_deps(calls))
    before = calls.count("download")
    assert before > 0

    def boom(*a, **k):
        raise AssertionError("重ね直してはいけない")

    monkeypatch.setattr(bgm_mix, "render", boom)
    second = build_bgm(cfg, out_dir, deps=make_deps(calls))
    assert calls.count("download") == before
    assert second.chapter_files == first.chapter_files
    assert second.full_path is not None and second.full_path.exists()


def test_daily_limit_writes_no_mp3(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    calls: list = []
    with pytest.raises(DailyLimit):
        build_bgm(cfg, out_dir, deps=make_deps(calls, client_error=DailyLimit("limit")))
    bgm = out_dir / "bgm"
    assert not list(bgm.glob("**/*.mp3"))
    assert "scorer_init" not in calls
    assert "unload" in calls  # LLM は途中で止まっても解放する


@needs_ffmpeg
def test_partial_chapters_keep_other_chapters_in_selection(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    calls: list = []
    r1 = build_bgm(cfg, out_dir, chapters={1}, deps=make_deps(calls))
    assert len(r1.chapter_files) == 1 and r1.full_path is None
    assert {c.chapter for c in load_selection(out_dir / "bgm" / "selection.json")} == {0}
    r2 = build_bgm(cfg, out_dir, chapters={2}, deps=make_deps(calls))
    assert len(r2.chapter_files) == 1 and r2.full_path is None
    assert {c.chapter for c in load_selection(out_dir / "bgm" / "selection.json")} == {0, 1}
    credits = (out_dir / "bgm" / "CREDITS.txt").read_text(encoding="utf-8")
    assert "sound 1" in credits


# ---------- CLI ----------


def _cli_args(tmp_path: Path) -> list[str]:
    conf = tmp_path / "config.yaml"
    conf.write_text(f"output_dir: {tmp_path / 'out'}\n", encoding="utf-8")
    return ["bgm", str(NOVEL), "-c", str(conf)]


def test_cli_bgm_daily_limit_exit_code(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise DailyLimit("limit")

    monkeypatch.setattr(cli, "build_bgm", boom)
    assert cli.main(_cli_args(tmp_path)) == 1
    assert "1 日の上限に達しました" in capsys.readouterr().out


def test_cli_bgm_key_missing_exit_code_2(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise KeyMissing("キーがありません: /x/secrets.env")

    monkeypatch.setattr(cli, "build_bgm", boom)
    assert cli.main(_cli_args(tmp_path)) == 2
    assert "キーがありません" in capsys.readouterr().out


def test_cli_bgm_import_error_exit_code_2(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(cli, "build_bgm", boom)
    assert cli.main(_cli_args(tmp_path)) == 2
    assert "--extra bgm" in capsys.readouterr().out


def test_cli_bgm_value_error_exit_code_1(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("対象の章がありません")

    monkeypatch.setattr(cli, "build_bgm", boom)
    assert cli.main(_cli_args(tmp_path)) == 1
    assert "対象の章がありません" in capsys.readouterr().out


def test_cli_bgm_reports_results_and_passes_options(tmp_path, monkeypatch, capsys):
    seen: dict = {}

    def fake(cfg, out_dir, **kw):
        seen.update(kw)
        (out_dir / "bgm").mkdir(parents=True, exist_ok=True)
        return BgmResult([out_dir / "bgm" / "chapters" / "01_a.mp3"], out_dir / "bgm" / "x_bgm_full.mp3", 3)

    monkeypatch.setattr(cli, "build_bgm", fake)
    rc = cli.main(_cli_args(tmp_path) + ["--chapters", "1-2", "--replan", "--llm-model", "m", "--think"])
    assert rc == 0
    assert seen["chapters"] == {1, 2}
    assert seen["replan"] is True
    assert seen["llm_overrides"] == {"model": "m", "think": True}
    out = capsys.readouterr().out
    assert "01_a.mp3" in out and "x_bgm_full.mp3" in out
    assert "3" in out and "missing.tsv" in out and "CREDITS.txt" in out


def test_cli_bgm_help_is_japanese(capsys):
    with pytest.raises(SystemExit):
        cli.main(["bgm", "--help"])
    out = capsys.readouterr().out
    assert "--replan" in out and "場面" in out


# ---------- レビュー指摘の修正 ----------


@needs_ffmpeg
def test_full_not_built_when_script_has_only_some_chapters(tmp_path):
    cfg, out_dir = make_book(tmp_path)
    rows = [r for r in pipeline.load_book(out_dir)[1] if int(r["chapter"]) == 0]
    pipeline.write_script(out_dir / "script.tsv", rows)  # script --chapters 1 の後の状態
    stale = out_dir / "bgm" / f"{out_dir.name}_bgm_full.mp3"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    result = build_bgm(cfg, out_dir, deps=make_deps([]))
    assert len(result.chapter_files) == 1
    assert result.full_path is None
    assert not stale.exists()


@pytest.mark.parametrize("problem", ["key", "times"])
def test_precheck_failure_removes_stale_full(tmp_path, problem):
    cfg, out_dir = make_book(tmp_path, key=problem != "key", times=problem != "times")
    stale = out_dir / "bgm" / f"{out_dir.name}_bgm_full.mp3"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    with pytest.raises((KeyMissing, ValueError)):
        build_bgm(cfg, out_dir, deps=make_deps([]))
    assert not stale.exists()
