"""試聴と小説生成（台本 → 音声 → 一括）のための Gradio UI。

    python -m irodori_speaker.webui [-c config.yaml] [--host 127.0.0.1] [--port 7860]

gradio / torch / irodori_tts は build_app と各ハンドラの中で遅延 import する。
"""

from __future__ import annotations

import argparse
import inspect
import math
import queue
import random
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

from .config import EMOTIONS, Config, load_config
from .lora import normalize_lora
from .pipeline import (
    TSV_FIELDS,
    book_dir,
    build_script,
    load_book,
    parse_chapter_range,
    synthesize_book,
    write_script,
)
from .readings import load_readings
from .voices import AUDIO_EXTS, NO_REF_VOICE, is_no_ref, list_voices

CHECKPOINT_CHOICES = [
    "Aratako/Irodori-TTS-v4.1-Small",
    "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
    "Aratako/Irodori-TTS-v4.1-Small-MF",
]
DEVICE_CHOICES = ["cuda", "mps", "cpu"]
PRECISION_CHOICES = ["bf16", "fp32"]
DEFAULT_TEXT = "ねえ、聞いて。ずっと探していたものが、やっと見つかったの。"
TTS_MISSING = (
    "音声合成には torch と irodori_tts が必要です。"
    "GPU マシンで `uv sync --frozen --extra tts --extra cu128 --extra webui` を実行してください（{err}）"
)


# ---- 純粋なヘルパー（gradio / torch 不要） ----


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def _to_int(value) -> int:
    return int(float(str(value).strip()))


def parse_seed(value) -> int | None:
    """-1（負の値）・空・None はランダム（None）。"""
    if _blank(value):
        return None
    n = _to_int(value)
    return n if n >= 0 else None


def parse_steps(value) -> int | None:
    """空・0 以下は既定（None）。"""
    if _blank(value):
        return None
    n = _to_int(value)
    return n if n > 0 else None


def llm_overrides_from_ui(api, base_url, model, think, concurrency) -> dict:
    """UI の LLM 設定を build_script の llm_overrides にする。空欄は config の値を使う。"""
    out: dict = {}
    for key, value in (("api", api), ("base_url", base_url), ("model", model)):
        if not _blank(value):
            out[key] = str(value).strip()
    out["think"] = bool(think)
    if not _blank(concurrency):
        out["concurrency"] = max(1, _to_int(concurrency))
    return out


def dataframe_rows(rows: list[dict]) -> list[list]:
    return [[r[f] for f in TSV_FIELDS] for r in rows]


def _key(chapter, para, seg) -> tuple[int, int, int]:
    return (_to_int(chapter), _to_int(para), _to_int(seg))


def rows_from_dataframe(df_rows, original_rows: list[dict]) -> list[dict]:
    """Dataframe の text 列だけを取り込み、他の列は読み込み時の値を使う。"""
    if hasattr(df_rows, "values"):  # pandas.DataFrame
        df_rows = df_rows.values.tolist()
    df_rows = [list(r) for r in df_rows]
    if len(df_rows) != len(original_rows):
        raise ValueError(
            f"台本の行数が読み込み時と違います（表 {len(df_rows)} 行 / 台本 {len(original_rows)} 行）。"
            "台本を読み込み直してください"
        )
    by_key = {_key(r["chapter"], r["para"], r["seg"]): r for r in original_rows}
    text_col = TSV_FIELDS.index("text")
    result = []
    for r in df_rows:
        k = _key(r[0], r[1], r[2])
        original = by_key.get(k)
        if original is None:
            raise ValueError(f"台本に無い行です（章 {k[0]} / 段落 {k[1]} / 文 {k[2]}）。台本を読み込み直してください")
        text = r[text_col]
        blank = _blank(text) or original.get("kind") == "break"  # 区切りは常に読まない
        result.append({**original, "text": "" if blank else str(text)})
    return result


def emoji_prefixed(text: str, emoji: str) -> str:
    """文頭に絵文字を置く。「『（ で始まるなら開き括弧の直後。"""
    from .emotion import OPEN_BRACKETS

    if not emoji:
        return text
    pos = 1 if text and text[0] in OPEN_BRACKETS else 0
    return text[:pos] + emoji + text[pos:]


def split_preview_text(text: str, max_chars: int) -> list[list[str]]:
    """試聴の本文を、小説生成と同じ規則で段落ごとの文のリストに分ける。"""
    from .text_splitter import split_paragraphs, split_sentences

    paragraphs = (split_sentences(para, max_chars) for para in split_paragraphs(text))
    return [sentences for sentences in paragraphs if sentences]


def join_audio(pieces, sr: int, sentence_gap: float, paragraph_gap: float):
    """段落ごとの文の音声を、文の間・段落の間に無音を挟んで 1 本にする。先頭と末尾には足さない。"""
    import numpy as np

    parts = []
    for paragraph in pieces:
        for s_idx, audio in enumerate(paragraph):
            if parts:
                gap = sentence_gap if s_idx else paragraph_gap
                parts.append(np.zeros(int(round(gap * sr)), dtype=np.float32))
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def synthesize_split(synth, paragraphs, overrides, sentence_gap, paragraph_gap, prefix=None):
    """分けた文を 1 文ずつ合成して結合する。文ごとに (i, n) を yield し、(sr, audio, warnings) を返す。

    prefix は文ごとに本文へ手を入れる関数（総当たり生成の絵文字用）。
    """
    n = sum(len(p) for p in paragraphs)
    rows, warnings, srs, done = [], [], [], 0
    for paragraph in paragraphs:
        row = []
        for sentence in paragraph:
            done += 1
            yield done, n
            sr, audio, warns = synth.generate_once(prefix(sentence) if prefix else sentence, **overrides)
            srs.append(sr)
            row.append(audio)
            warnings.extend(warns)
        rows.append(row)
    if len(set(srs)) != 1:
        raise ValueError(f"文ごとのサンプリングレートが一致しません: {sorted(set(srs))}")
    return srs[0], join_audio(rows, srs[0], sentence_gap, paragraph_gap), warnings


def carry_emoji(paragraphs: list[list[str]]) -> list[list[str]]:
    """絵文字で始まらない文の先頭に、直前までに最後に現れた絵文字の連なりを付ける。

    文ごとに合成すると 2 文目以降は感情の指定が切れるため、台本作成と同じく文頭に必ず置く。
    """
    from .emoji import emoji_runs
    from .emotion import OPEN_BRACKETS

    last = ""
    result = []
    for paragraph in paragraphs:
        row = []
        for sentence in paragraph:
            runs = emoji_runs(sentence)
            head = 1 if sentence and sentence[0] in OPEN_BRACKETS else 0
            if last and not (runs and runs[0][0] == head):
                sentence_out = emoji_prefixed(sentence, last)
            else:
                sentence_out = sentence
            row.append(sentence_out)
            out_runs = emoji_runs(sentence_out)
            if out_runs:
                last = sentence_out[slice(*out_runs[-1])]
        result.append(row)
    return result


def voice_choices(voices_dir) -> list[str]:
    """声の候補。先頭は参照音声なしの none（voices/none.wav があっても重複させない）。"""
    return [NO_REF_VOICE] + [v for v in list_voices(Path(voices_dir)) if v != NO_REF_VOICE]


def add_voice(src, voice_id: str, voices_dir) -> Path:
    """アップロードされた音声を voices/<id><拡張子> にコピーする。"""
    voice_id = (voice_id or "").strip()
    if not voice_id or "/" in voice_id or "\\" in voice_id or voice_id.startswith("."):
        raise ValueError("声の ID には、空でなく / \\ を含まず . で始まらない名前を指定してください")
    if voice_id == NO_REF_VOICE:
        raise ValueError(f"{NO_REF_VOICE} は参照音声を使わない設定の予約語なので、声の ID には使えません")
    if not src:
        raise ValueError("先に参照音声をアップロードしてください")
    src = Path(src)
    suffix = src.suffix.lower() if src.suffix.lower() in AUDIO_EXTS else ".wav"
    voices_dir = Path(voices_dir)
    if voice_id in list_voices(voices_dir):
        raise ValueError(f"声 {voice_id!r} は既にあります。別の ID にするか、voices/ のファイルを削除してください")
    voices_dir.mkdir(parents=True, exist_ok=True)
    dest = voices_dir / f"{voice_id}{suffix}"
    shutil.copyfile(src, dest)
    return dest


def stream_progress(fn):
    """fn(progress) を別スレッドで実行し、progress(done, total, msg) を流す generator。

    `result = yield from stream_progress(fn)` で fn の戻り値を得る。fn の例外は再送出する。
    """
    q: queue.Queue = queue.Queue()
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = fn(lambda done, total, msg: q.put((done, total, msg)))
        except BaseException as e:  # noqa: BLE001 呼び出し側スレッドで再送出する
            box["error"] = e
        finally:
            q.put(None)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    finished = False
    while not finished:
        latest = q.get()
        if latest is None:
            break
        while True:  # 溜まった分は最新だけ流す
            try:
                nxt = q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                finished = True
                break
            latest = nxt
        yield latest
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _novel_path(novel_file, novel_text) -> Path:
    raw = novel_file if novel_file else (novel_text or "").strip()
    if not raw:
        raise ValueError("小説のファイルをアップロードするか、パスを入力してください")
    path = Path(getattr(raw, "name", raw)).expanduser()
    if not path.is_file():
        raise ValueError(f"小説のファイルが見つかりません: {path}")
    return path


def should_reload(path, loaded) -> bool:
    """直近に表へ読み込んだ小説（loaded）と違うときだけ、台本を読み直す。"""
    return loaded is None or str(path) != str(loaded)


def _existing_rows(out_dir: Path) -> list[dict]:
    try:
        return load_book(out_dir)[1]
    except ValueError:
        return []


# ---- Engine ----


class Engine:
    """(checkpoint, device, precision, voice) が変わったら Synthesizer を作り直す。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._synth = None
        self._key: tuple | None = None
        self._ready = False

    def key_for(self, checkpoint, device, precision, voice, lora_adapter=None) -> tuple:
        voice = (voice or "").strip()
        if not voice:
            raise ValueError("声を選んでください（voices/ に音声を置くか、画面からアップロードしてください）")
        return (
            (checkpoint or "").strip() or self.cfg.checkpoint,
            device or self.cfg.device,
            precision or self.cfg.precision,
            voice,
            # 空欄は config、入力があれば none 系は None（config を打ち消す）
            normalize_lora(lora_adapter) if (lora_adapter or "").strip() else normalize_lora(self.cfg.lora_adapter),
        )

    def cfg_for(self, key: tuple) -> Config:
        checkpoint, device, precision, voice, lora_adapter = key
        return replace(
            self.cfg, checkpoint=checkpoint, device=device, precision=precision, voice=voice,
            lora_adapter=lora_adapter,
        )

    def is_loaded(self, key: tuple) -> bool:
        return self._ready and self._key == key

    def mark_unloaded(self) -> None:
        """runtime を GPU から解放した後に呼ぶ。次の利用時に ensure_loaded が読み直す。"""
        self._ready = False

    def invalidate(self) -> None:
        """声ファイルを入れ替えたときなど、Synthesizer を作り直させる。"""
        self._synth = None
        self._key = None
        self._ready = False

    def get(
        self, key: tuple, segments_dir: Path, warnings_path: Path | None = None, out_dir: Path | None = None
    ):
        # torch / irodori_tts が無い環境でも画面だけは出せるよう遅延 import
        from .synth import Synthesizer
        from .voices import describe_voice, resolve_voice

        cfg = self.cfg_for(key)
        if self._synth is None or self._key != key:
            if is_no_ref(cfg.voice):
                voice_paths = []
                print("[webui] 使う声: なし（参照音声を使わない）")
            else:
                voice_paths = resolve_voice(Path(cfg.voices_dir), cfg.voice)
                print(f"[webui] 使う声: {describe_voice(voice_paths)}")
            self._synth = Synthesizer(
                cfg, segments_dir, voice_paths, Path(cfg.output_dir) / ".voices", warnings_path
            )
            self._key = key
            self._ready = False
        synth = self._synth
        synth.refresh_lora()  # 同じパスのアダプタが学習し直されていたら検出する
        synth.segments_dir = Path(segments_dir)
        synth.segments_dir.mkdir(parents=True, exist_ok=True)
        synth.warnings_path = Path(warnings_path) if warnings_path else None
        synth.readings = load_readings(cfg, out_dir)  # Synthesizer は使い回すが、辞書は本ごと
        return synth

    def load(
        self, key: tuple, segments_dir: Path, warnings_path: Path | None = None, out_dir: Path | None = None
    ):
        synth = self.get(key, segments_dir, warnings_path, out_dir)
        if not self._ready:
            synth.runtime  # noqa: B018 モデルを読み込む（キーが変わるときは VRAM を確認する）
            synth.latent_path  # noqa: B018 参照音声を潜在表現に変換する
            self._ready = True
        return synth


# ---- 画面 ----


def build_app(cfg: Config):
    import gradio as gr

    engine = Engine(cfg)
    stop_flag = threading.Event()
    preview_dir = Path(cfg.output_dir) / ".preview"
    voices_dir = Path(cfg.voices_dir)
    gpu = {"concurrency_limit": 1, "concurrency_id": "gpu"}  # LLM / TTS を同時に動かさない

    css = ""
    palette = None
    try:
        from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette

        css, palette = EMOJI_PALETTE_CSS, build_emoji_palette
    except ImportError:
        pass

    def fail(prefix: str, e: Exception):
        print(f"[webui] {prefix}: {e}")
        if isinstance(e, ImportError):
            return gr.Error(TTS_MISSING.format(err=e))
        return gr.Error(f"{prefix}: {e}")

    def ensure_loaded(key, segments_dir, warnings_path=None, out_dir=None):
        """読み込みが必要なら状態を yield してから読み込み、Synthesizer を返す。"""
        if not engine.is_loaded(key):
            yield f"⏳ モデルを読み込み中: {key[0]}（初回は数分かかります。完了まで他の操作は待ってください）"
            t0 = time.perf_counter()
            synth = engine.load(key, segments_dir, warnings_path, out_dir)
            yield f"✅ 読み込み完了 ({time.perf_counter() - t0:.0f}秒): {key[0]} / {key[1]} / {key[2]} / 声 {key[3]}"
            return synth
        return engine.load(key, segments_dir, warnings_path, out_dir)

    def release_tts() -> str:
        """LLM を動かす前に TTS の runtime を GPU から外す（8GB では両方載らない）。"""
        from . import synth as synth_mod

        synth_mod.release_runtime()
        engine.mark_unloaded()
        print("[webui] TTS を解放しました")
        return "🧹 TTS を解放しました"

    def tts_key(checkpoint, device, precision, voice, lora_adapter=None):
        try:
            return engine.key_for(checkpoint, device, precision, voice, lora_adapter)
        except ValueError as e:
            raise gr.Error(str(e)) from e

    # -- TTS 設定 --

    def refresh_voices(current):
        return gr.update(choices=voice_choices(voices_dir), value=current)

    def upload_voice(audio_path, voice_id):
        try:
            dest = add_voice(audio_path, voice_id, voices_dir)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        engine.invalidate()
        print(f"[webui] 声を追加しました: {dest}")
        return gr.update(choices=voice_choices(voices_dir), value=dest.stem), f"✅ 声を追加しました: {dest}"

    def load_model(checkpoint, device, precision, voice, lora_adapter=None):
        key = tts_key(checkpoint, device, precision, voice, lora_adapter)
        try:
            for msg in ensure_loaded(key, preview_dir):
                yield msg
        except Exception as e:
            raise fail("読み込みに失敗しました", e) from e
        yield f"✅ 読み込み済み: {key[0]} / {key[1]} / {key[2]} / 声 {key[3]}"

    # -- LLM 設定 --

    def fetch_models(api, base_url, model):
        from .emotion import list_models

        try:
            names = list_models(replace(cfg.llm, api=api, base_url=(base_url or "").strip() or cfg.llm.base_url))
        except Exception as e:
            print(f"[webui] モデル一覧の取得に失敗: {e}")
            return gr.update(), f"❌ モデル一覧の取得に失敗しました: {e}"
        return gr.update(choices=names, value=model), f"✅ {len(names)} 件のモデルを取得しました"

    # -- 試聴 --

    def insert_emoji(text, emotion, intensity):
        from .emotion import Annotation, to_emoji

        return emoji_prefixed(text, to_emoji(Annotation(emotion, int(intensity)), cfg.emotion_emoji))

    def annotate_text(text, api, base_url, model, think, concurrency):
        from .emotion import (
            Annotation,
            LLMAnnotator,
            LLMServer,
            apply_shift,
            build_spoken_text,
            split_clauses,
        )
        from .text_splitter import Segment

        body = (text or "").strip()
        if not body:
            raise gr.Error("本文を入力してください")
        llm_cfg = replace(cfg.llm, **llm_overrides_from_ui(api, base_url, model, think, concurrency))
        kind = "dialogue" if body[0] in "「『（(" else "narration"
        seg = Segment(0, 0, 0, kind, body)
        yield gr.update(), release_tts() + "。⏳ LLM で絵文字を付けています …"
        try:
            with LLMServer(llm_cfg):
                annotator = LLMAnnotator(llm_cfg, preview_dir / "annotations.jsonl")
                try:
                    anns = annotator.annotate_many([([seg], None, None)])[0][0]
                finally:
                    annotator.unload()
        except Exception as e:
            raise fail("LLM での推定に失敗しました", e) from e
        shifted = []
        for a in anns:
            a = apply_shift(a, kind, cfg.narration_intensity_shift)
            shifted.append(a if a.emotion in EMOTIONS else Annotation())
        spoken = build_spoken_text(split_clauses(body), shifted, cfg.emotion_emoji)
        yield spoken, "✅ 絵文字を付けました（LLM を解放しました）"

    def sampling_args(steps, cfg_text, cfg_speaker, duration, seed):
        seed_value = parse_seed(seed)
        if seed_value is None:
            seed_value = random.randrange(2**31)
        return seed_value, {
            "num_steps": parse_steps(steps),
            "cfg_scale_text": float(cfg_text),
            "cfg_scale_speaker": float(cfg_speaker),
            "duration_scale": float(duration),
            "seed": seed_value,
        }

    def warn_note(warnings):
        return f"（警告 {len(warnings)} 件: {warnings[-1]}）" if warnings else ""

    def generate(
        text, checkpoint, device, precision, voice, steps, cfg_text, cfg_speaker, duration, seed,
        lora_adapter=None,
    ):
        if not (text or "").strip():
            raise gr.Error("本文を入力してください")
        key = tts_key(checkpoint, device, precision, voice, lora_adapter)
        seed_value, overrides = sampling_args(steps, cfg_text, cfg_speaker, duration, seed)
        try:
            synth = yield from _with_status(ensure_loaded(key, preview_dir))
            paragraphs = carry_emoji(split_preview_text(text, cfg.split.max_chars))
            n = sum(len(p) for p in paragraphs)
            yield gr.update(), "🎙 生成中…"
            t0 = time.perf_counter()
            work = synthesize_split(synth, paragraphs, overrides, cfg.silence.sentence, cfg.silence.paragraph)
            try:
                while True:
                    i, _ = next(work)
                    if n >= 2:
                        yield gr.update(), f"🎙 生成中… ({i}/{n})"
            except StopIteration as stop:
                sr, audio, warnings = stop.value
        except Exception as e:
            raise fail("生成に失敗しました", e) from e
        split_note = f"・{n} 文に分けて合成" if n >= 2 else ""
        yield (sr, audio), f"✅ 生成完了 seed={seed_value}・{time.perf_counter() - t0:.1f}秒{split_note} {warn_note(warnings)}"

    def _with_status(step):
        """ensure_loaded の状態を (gr.update(), msg) として流し、Synthesizer を返す。"""
        try:
            while True:
                yield gr.update(), next(step)
        except StopIteration as stop:
            return stop.value

    def compare_emotions(
        text, checkpoint, device, precision, voice, steps, cfg_text, cfg_speaker, duration, seed, intensity,
        lora_adapter=None, progress=gr.Progress(),
    ):
        # 同じ seed・同じ本文で、文頭の絵文字だけ変える。差が出なければ絵文字が効いていない
        from .emotion import Annotation, to_emoji

        if not (text or "").strip():
            raise gr.Error("本文を入力してください")
        key = tts_key(checkpoint, device, precision, voice, lora_adapter)
        seed_value, overrides = sampling_args(steps, cfg_text, cfg_speaker, duration, seed)
        n = len(EMOTIONS)
        results = [gr.update()] * n
        paragraphs = split_preview_text(text, cfg.split.max_chars)
        try:
            step = ensure_loaded(key, preview_dir)
            try:
                while True:
                    yield [*results, next(step)]
            except StopIteration as stop:
                synth = stop.value
            for i, emotion in enumerate(EMOTIONS):
                emoji = to_emoji(Annotation(emotion, int(intensity)), cfg.emotion_emoji)
                msg = f"🎙 生成中… ({i + 1}/{n}) {emotion} {emoji}"
                progress((i, n), desc=msg)
                yield [*results, msg]
                work = synthesize_split(
                    synth, paragraphs, overrides, cfg.silence.sentence, cfg.silence.paragraph,
                    prefix=lambda piece, emoji=emoji: emoji_prefixed(piece, emoji),
                )
                try:
                    while True:
                        j, total = next(work)
                        if total >= 2:
                            yield [*results, f"{msg} [{j}/{total}]"]
                except StopIteration as stop:
                    sr, audio, _warnings = stop.value
                results[i] = gr.update(value=(sr, audio), label=f"{emotion}: {emoji or '(絵文字なし)'}")
        except Exception as e:
            raise fail("生成に失敗しました", e) from e
        yield [*results, f"✅ {n} 件の生成完了 seed={seed_value}"]

    # -- 小説生成 --

    def show_script(novel_file, novel_text, loaded):
        """選択中の小説が変わったとき、既存の script.tsv があれば表示する。"""
        keep = gr.update()
        try:
            novel = _novel_path(novel_file, novel_text)
        except ValueError:
            return keep, keep, keep, loaded
        if not should_reload(novel, loaded):
            return keep, keep, keep, loaded  # 同じ小説なら未保存の編集を消さない
        rows = _existing_rows(book_dir(cfg, novel))
        if not rows:
            return gr.update(value=[]), [], keep, str(novel)
        status = f"既存の script.tsv を読み込みました（{len(rows)} 行）"
        return gr.update(value=dataframe_rows(rows)), rows, status, str(novel)

    def save_script(df, original, novel_file, novel_text):
        try:
            out_dir = book_dir(cfg, _novel_path(novel_file, novel_text))
            rows = rows_from_dataframe(df, original or _existing_rows(out_dir))
            write_script(out_dir / "script.tsv", rows)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        return rows, f"✅ script.tsv を保存しました: {out_dir / 'script.tsv'}（{len(rows)} 行）"

    def stage_script(novel, out_dir, chapters, no_emotion, reannotate, overrides):
        selected = parse_chapter_range(chapters)
        if not no_emotion and cfg.emotion_mode != "none":
            yield release_tts()

        def work(progress):
            return build_script(
                novel,
                cfg,
                out_dir,
                chapters=selected,
                use_emotion=not no_emotion,
                reannotate=reannotate,
                llm_overrides=overrides,
                progress=progress,
            )

        step = stream_progress(work)
        try:
            while True:
                done, total, msg = next(step)
                yield f"📝 台本を作成中 {done}/{total}: {msg}"
        except StopIteration:
            pass
        return load_book(out_dir)[1]

    def stage_synth(
        out_dir, chapters, force, checkpoint, device, precision, voice, df, original, progress,
        lora_adapter=None,
    ):
        """(status, outputs) を yield する。outputs は終了時のみ (章 WAV, 全体 WAV)。"""
        key = tts_key(checkpoint, device, precision, voice, lora_adapter)
        script_path = out_dir / "script.tsv"
        if df is not None:
            rows = rows_from_dataframe(df, original or _existing_rows(out_dir))
            write_script(script_path, rows)
        titles, rows = load_book(out_dir)
        selected = parse_chapter_range(chapters)
        synth_cfg = engine.cfg_for(key)
        try:
            synth = yield from _bare(
                ensure_loaded(key, out_dir / "segments", out_dir / "warnings.log", out_dir)
            )
        except RuntimeError as e:  # VRAM 不足など
            print(f"[webui] 中止: {e}")
            yield f"❌ {e}", None
            return
        session_warnings: list[str] = []

        def work(cb):
            def on_progress(done, total, msg):
                session_warnings.extend(synth.last_warnings)
                cb(done, total, msg)

            return synthesize_book(
                rows, titles, synth_cfg, out_dir, synth,
                chapters=selected, force=force, progress=on_progress, should_stop=stop_flag.is_set,
            )

        step = stream_progress(work)
        try:
            while True:
                done, total, msg = next(step)
                progress(done / max(total, 1), desc=msg)
                yield f"🎙 {msg}", None
        except StopIteration as stop:
            result = stop.value
        note = warn_note(session_warnings)
        files = [str(p) for p in result.chapter_files]
        if result.stopped:
            yield f"⏹ 停止しました。再実行すると続きから進みます {note}", (files, None)
        elif result.full_path:
            yield f"✅ 完了: {result.full_path} {note}", (files, str(result.full_path))
        else:
            stale = (
                "。台本と一致しない章: " + ", ".join(str(n) for n in result.stale_chapters)
                if result.stale_chapters
                else ""
            )
            yield f"✅ 章ごとの WAV を出力しました（全章が揃っていないため通しは未作成{stale}） {note}", (files, None)

    def _bare(step):
        """ensure_loaded の状態を (msg, None) として流し、Synthesizer を返す。"""
        try:
            while True:
                yield next(step), None
        except StopIteration as stop:
            return stop.value

    def run_novel(mode, novel_file, novel_text, chapters, no_emotion, reannotate, force, checkpoint, device,
                  precision, voice, api, base_url, model, think, concurrency, df, original,
                  lora_adapter=None, progress=gr.Progress()):
        """mode: script | synth | all。出力は (状態, 表, 台本の行, 章 WAV, 全体 WAV)。"""
        keep = gr.update()
        stop_flag.clear()
        df = [r for r in (df or []) if any(not _blank(c) for c in r)]  # 空の表は「未表示」として扱う
        try:
            novel = _novel_path(novel_file, novel_text)
            out_dir = book_dir(cfg, novel)
            table, rows_state = keep, keep
            if mode in ("script", "all"):
                overrides = llm_overrides_from_ui(api, base_url, model, think, concurrency)
                rows = yield from _script_events(
                    stage_script(novel, out_dir, chapters, no_emotion, reannotate, overrides), keep
                )
                table, rows_state = gr.update(value=dataframe_rows(rows)), rows
                yield f"✅ 台本を作りました: {out_dir / 'script.tsv'}（{len(rows)} 行）", table, rows_state, keep, keep
                if mode == "all":
                    df, original = None, None  # 直前に作った script.tsv をそのまま使う
                    if stop_flag.is_set():
                        yield "⏹ 停止しました（台本は作成済みです）", table, rows_state, keep, keep
                        return
            if mode in ("synth", "all"):
                if mode == "synth" and not df:
                    if not _existing_rows(out_dir):
                        raise ValueError(f"先に台本を作ってください: {out_dir / 'script.tsv'}")
                    df, original = None, None  # 表が空なら script.tsv をそのまま使う
                for status, outputs in stage_synth(
                    out_dir, chapters, force, checkpoint, device, precision, voice, df, original, progress,
                    lora_adapter=lora_adapter,
                ):
                    files, full = outputs if outputs else (keep, keep)
                    yield status, table, rows_state, files, full
        except gr.Error:
            raise
        except Exception as e:
            raise fail("失敗しました", e) from e

    def _script_events(step, keep):
        try:
            while True:
                yield next(step), keep, keep, keep, keep
        except StopIteration as stop:
            return stop.value

    def make_handler(mode):
        def handler(*args, progress=gr.Progress()):
            for out in run_novel(mode, *args, progress=progress):
                # 表を更新したら、直近に読み込んだ小説もそれに合わせる（blur で消えないように）
                table_updated = mode != "synth" and isinstance(out[2], list)
                yield (*out, str(_novel_path(args[0], args[1])) if table_updated else gr.update())

        handler.__name__ = f"run_{mode}"
        return handler

    def stop_run():
        stop_flag.set()
        gr.Info("停止を要求しました。現在の文の合成が終わり次第止まります")

    # -- 画面の組み立て --

    blocks_kwargs = {"title": "irodori-speaker"}
    if css and "css" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["css"] = css
    with gr.Blocks(**blocks_kwargs) as app:
        gr.Markdown("# irodori-speaker\n試聴（声・絵文字・パラメータの確認）と、小説全体の音声生成。")
        with gr.Group():
            gr.Markdown("### TTS 設定")
            with gr.Row():
                checkpoint = gr.Dropdown(
                    CHECKPOINT_CHOICES, value=cfg.checkpoint, allow_custom_value=True,
                    label="チェックポイント（HF の ID / ローカルのパスも入力可）", scale=3,
                )
                device = gr.Dropdown(DEVICE_CHOICES, value=cfg.device, label="device", scale=1)
                precision = gr.Dropdown(PRECISION_CHOICES, value=cfg.precision, label="precision", scale=1)
            with gr.Row():
                voice = gr.Dropdown(
                    voice_choices(voices_dir), value=cfg.voice, allow_custom_value=True, label="声", scale=3
                )
                lora = gr.Textbox(
                    value=cfg.lora_adapter or "", label="話者 LoRA のアダプタ（空で config、none で無効）", scale=2
                )
                refresh_btn = gr.Button("更新", scale=1)
                load_btn = gr.Button("モデル読み込み", variant="primary", scale=1)
            with gr.Row():
                voice_audio = gr.Audio(type="filepath", sources=["upload"], label="参照音声をアップロード")
                voice_id = gr.Textbox(label="声の ID（voices/<ID>.wav になる）")
                add_voice_btn = gr.Button("声を追加")
            tts_status = gr.Textbox(
                label="状態", value="未読み込み（最初の生成時、または上のボタンで読み込みます）", interactive=False
            )
        with gr.Accordion("LLM 設定（台本の感情推定・絵文字付け）", open=False):
            with gr.Row():
                api = gr.Radio(["ollama", "openai"], value=cfg.llm.api, label="api")
                base_url = gr.Textbox(value=cfg.llm.base_url, label="base_url", scale=2)
            with gr.Row():
                model = gr.Dropdown(
                    [cfg.llm.model], value=cfg.llm.model, allow_custom_value=True, label="モデル", scale=3
                )
                fetch_btn = gr.Button("一覧を取得", scale=1)
            with gr.Row():
                think = gr.Checkbox(value=cfg.llm.think, label="思考モード（遅くなる）")
                concurrency = gr.Number(value=cfg.llm.concurrency, precision=0, minimum=1, label="並列数")
            llm_status = gr.Textbox(label="LLM の状態", interactive=False)
        llm_inputs = [api, base_url, model, think, concurrency]
        tts_inputs = [checkpoint, device, precision, voice]
        lora_inputs = [lora]  # 各 click の inputs の末尾に足す（ハンドラ末尾の kwarg に対応）

        refresh_btn.click(refresh_voices, voice, voice)
        add_voice_btn.click(upload_voice, [voice_audio, voice_id], [voice, tts_status])
        load_btn.click(load_model, [*tts_inputs, *lora_inputs], tts_status, **gpu)
        fetch_btn.click(fetch_models, [api, base_url, model], [model, llm_status])

        with gr.Tab("試聴"):
            text = gr.Textbox(label="本文", value=DEFAULT_TEXT, lines=3)
            if palette is not None:
                palette(text, open=False)
            else:
                with gr.Row():
                    emotion = gr.Dropdown(EMOTIONS, value="happy", label="感情")
                    intensity = gr.Slider(1, 3, value=2, step=1, label="強度")
                    insert_btn = gr.Button("絵文字を文頭に挿入")
                insert_btn.click(insert_emoji, [text, emotion, intensity], text)
            annotate_btn = gr.Button("LLM で絵文字を付ける")
            with gr.Row():
                steps = gr.Number(
                    value=cfg.sampling.num_steps, precision=0, label="num_steps（空で既定）"
                )
                cfg_text = gr.Number(value=cfg.sampling.cfg_scale_text, label="cfg_scale_text")
                cfg_speaker = gr.Number(value=cfg.sampling.cfg_scale_speaker, label="cfg_scale_speaker")
                duration = gr.Number(value=cfg.sampling.duration_scale, label="duration_scale")
                seed = gr.Number(
                    value=cfg.sampling.seed if cfg.sampling.seed is not None else -1,
                    precision=0, label="seed（-1 でランダム）",
                )
            gen_btn = gr.Button("生成", variant="primary")
            audio = gr.Audio(label="結果")
            preview_status = gr.Textbox(label="状態（seed・所要時間・警告）", interactive=False)
            sampling_inputs = [steps, cfg_text, cfg_speaker, duration, seed]
            annotate_btn.click(annotate_text, [text, *llm_inputs], [text, preview_status], **gpu)
            gen_btn.click(
                generate, [text, *tts_inputs, *sampling_inputs, *lora_inputs], [audio, preview_status], **gpu
            )
            gr.Markdown("### 感情プリセット総当たり\n同じ seed・同じ本文で、文頭に各感情の絵文字を付けて 10 本生成します。")
            brute_intensity = gr.Slider(1, 3, value=3, step=1, label="強度")
            brute_btn = gr.Button("総当たり生成")
            brute_outs = [gr.Audio(label=e) for e in EMOTIONS]
            brute_btn.click(
                compare_emotions,
                [text, *tts_inputs, *sampling_inputs, brute_intensity, *lora_inputs],
                [*brute_outs, preview_status],
                **gpu,
            )

        with gr.Tab("小説生成"):
            with gr.Row():
                novel_file = gr.File(label="小説（テキスト）", file_types=[".txt"], type="filepath")
                with gr.Column():
                    novel_text = gr.Textbox(label="小説のパス（アップロードがあればそちらを使う）")
                    chapters = gr.Textbox(label="章の範囲（例: 1-3,5。空で全章）")
            with gr.Row():
                no_emotion = gr.Checkbox(label="感情を付けない")
                reannotate = gr.Checkbox(label="感情を作り直す（reannotate）")
                force = gr.Checkbox(label="作り直す（force）")
            with gr.Row():
                script_btn = gr.Button("台本を作る（script.tsv）")
                synth_btn = gr.Button("音声にする", variant="primary")
                all_btn = gr.Button("一括実行", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")
            table_kwargs = {}
            if "static_columns" in inspect.signature(gr.Dataframe.__init__).parameters:
                table_kwargs["static_columns"] = [i for i, f in enumerate(TSV_FIELDS) if f != "text"]
            table = gr.Dataframe(
                headers=TSV_FIELDS, datatype="str", type="array", interactive=True, wrap=True,
                label="台本（text 列だけ編集できます）", **table_kwargs,
            )
            script_rows = gr.State([])
            loaded_novel = gr.State(None)
            save_btn = gr.Button("script.tsv を保存")
            novel_status = gr.Textbox(label="状態", interactive=False)
            chapter_files = gr.Files(label="章ごとの WAV")
            full_audio = gr.Audio(type="filepath", label="全体の WAV")

            novel_inputs = [
                novel_file, novel_text, chapters, no_emotion, reannotate, force,
                *tts_inputs, *llm_inputs, table, script_rows, *lora_inputs,
            ]
            novel_outputs = [novel_status, table, script_rows, chapter_files, full_audio, loaded_novel]
            for btn, mode in ((script_btn, "script"), (synth_btn, "synth"), (all_btn, "all")):
                btn.click(make_handler(mode), novel_inputs, novel_outputs, **gpu)
            stop_btn.click(stop_run, None, None)
            save_btn.click(save_script, [table, script_rows, novel_file, novel_text], [script_rows, novel_status])
            for trigger in (novel_file.change, novel_text.submit, novel_text.blur):
                trigger(
                    show_script, [novel_file, novel_text, loaded_novel], [table, script_rows, novel_status, loaded_novel]
                )

    app.irodori_css = css  # Gradio 6 以降は css を launch() に渡す
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="irodori_speaker.webui")
    ap.add_argument("-c", "--config", help="設定 YAML（初期値として読む。省略可）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    app = build_app(load_config(args.config))
    launch_kwargs = {"server_name": args.host, "server_port": args.port}
    if app.irodori_css and "css" in inspect.signature(app.launch).parameters:
        launch_kwargs["css"] = app.irodori_css
    print(f"[webui] 起動します: http://{args.host}:{args.port}")
    app.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
