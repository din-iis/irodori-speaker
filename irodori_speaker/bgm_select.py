from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import requests

from . import audio_io
from .bgm_types import Choice, Scene, Sound
from .config import BgmConfig, LLMConfig
from .emotion import chat_json
from .freesound import DailyLimit, FreesoundError, download_preview

GATE_PROMPT_VERSION = 1
GATE_SYSTEM_PROMPT_AMBIENT = (
    "You choose background ambience recordings for scenes in an audiobook. "
    "The ambience plays quietly under a narrator's voice for minutes, so the listener hears only its overall sound. "
    "For each candidate recording, decide from its name, tags and description whether it would sound right in the scene. "
    "Judge by how the recording sounds, not by where it was recorded: a recording from another city, "
    "or a spaceship interior hum for a moon base, is fine when it would sound the same in the scene. "
    "Reject a candidate if any of these hold: "
    "(1) it contains a prominent sound that would not be heard in the scene "
    "(for example breaking glass, birds, machinery, sirens, thunder or strong wind that the scene does not mention); "
    "(2) it contains speech that could be understood, including station or airport announcements, "
    "conversation, radio, TV or narration "
    "(an unintelligible crowd murmur is acceptable when the scene is a crowded place); "
    "(3) it contains music or jingles; "
    "(4) its time of day, weather, or indoor/outdoor setting contradicts the scene. "
    "When unsure, reject. Keep each reason under 15 words. Answer for every candidate id exactly once."
)
GATE_SYSTEM_PROMPT_MUSIC = (
    "You choose short instrumental music pieces played between scenes of an audiobook, with no narration over them. "
    "For each candidate, decide from its name, tags and description whether it fits the requested mood. "
    "Designed drones, pads and atmospheric textures count as music. "
    "Reject a candidate if any of these hold: "
    "(1) it has singing, vocals or speech; "
    "(2) it is a field recording of a real place or a sound effect rather than music; "
    "(3) its mood contradicts the requested mood. "
    "When unsure, reject. Keep each reason under 15 words. Answer for every candidate id exactly once."
)
# 環境音だけ、LLM の前に名前（タグは使わない）を単語で照らし合わせて外す。Task 0 で決めた
KEYWORD_NEGATION = re.compile(r"(no|without|sans)[\s_-]*music|music[\s_-]*free", re.I)
KEYWORD_REJECT = re.compile(
    r"announce|annouc|anounc|radio|televis|\btv\b|narrat|music|\bsong|singing|\bsinger|vocal|lyric|jingle|melod",
    re.I,
)
GATE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "accept": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["id", "accept", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["results"],
    "additionalProperties": False,
}

_DESC_CHARS = 300
_GATE_ATTEMPTS = 2
_CLAP_SR = 48000
_ONE_LINE_RE = re.compile(r"\s+")

# LLM 応答の壊れ方（emotion.LLMAnnotator と同じ範囲）。これらは「失敗」として扱う
_CHAT_ERRORS = (
    requests.RequestException,
    json.JSONDecodeError,
    KeyError,
    ValueError,
    TypeError,
    IndexError,
    AttributeError,
)


@dataclass
class Slot:
    chapter: int
    scene: int
    kind: str  # "ambient" | "intro" | "bridge"
    query: str
    duration: str
    clap_text: str
    gate_text: str

    @property
    def key(self) -> str:
        return f"{self.chapter}:{self.scene}:{self.kind}"


def build_slots(scenes: dict[int, list[Scene]], bgm: BgmConfig) -> list[Slot]:
    slots: list[Slot] = []
    for chapter, chapter_scenes in scenes.items():
        for sc in chapter_scenes:
            if sc.ambient_query is not None:
                slots.append(Slot(
                    sc.chapter, sc.index, "ambient", sc.ambient_query, bgm.ambient_duration,
                    "the sound of " + sc.description, sc.description,
                ))
            if sc.music_query:
                kind = "intro" if sc.index == 0 else "bridge"
                slots.append(Slot(
                    sc.chapter, sc.index, kind, sc.music_query, bgm.music_duration,
                    f"a {sc.mood} piece of music", f"mood: {sc.mood}",
                ))
    return slots


def _write_json(path: Path, data: object) -> None:
    """一時ファイル経由で JSON を置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------- 検索 ----------


def search_slots(
    slots: list[Slot], client, cache_path: Path, n: int
) -> tuple[dict[str, list[Sound]], dict[str, str]]:
    """slot ごとに検索する。結果は 1 slot ごとにキャッシュへ書き足す（止まっても残る）。"""
    cache = _read_json(cache_path)
    found: dict[str, list[Sound]] = {}
    failures: dict[str, str] = {}
    for s in slots:
        ck = f"{s.query}|{s.duration}|{n}"
        if ck not in cache:
            try:
                results = client.search(s.query, s.duration, n)
            except DailyLimit:
                raise
            except FreesoundError as e:
                print(f"[bgm] 検索に失敗しました（{s.key}）: {e}")
                failures[s.key] = "error"  # キャッシュしない（次の実行でやり直す）
                continue
            cache[ck] = [x.to_dict() for x in results]
            _write_json(cache_path, cache)
        found[s.key] = [Sound.from_dict(d) for d in cache[ck]]
    return found, failures


# ---------- ふるい分け ----------


def keyword_reject(sound: Sound) -> str | None:
    """名前に除外語があれば、当たった語（小文字）を返す。タグは見ない。"""
    name = KEYWORD_NEGATION.sub(" ", sound.name)
    m = KEYWORD_REJECT.search(name)
    return m.group(0).lower() if m else None


def _one_line(text: str) -> str:
    return _ONE_LINE_RE.sub(" ", text).strip()


def build_gate_prompt(gate_text: str, sounds: list[Sound]) -> str:
    lines = [
        f"id={s.id} | name={_one_line(s.name)} | tags={_one_line(' '.join(s.tags))}"
        f" | description={_one_line(s.description)[:_DESC_CHARS]}"
        for s in sounds
    ]
    return f"Scene: {_one_line(gate_text)}\n\nCandidates:\n" + "\n".join(lines)


def _validate_gate(resp: object, ids: set[int]) -> dict[int, tuple[bool, str]] | None:
    if not isinstance(resp, dict) or not isinstance(resp.get("results"), list):
        return None
    out: dict[int, tuple[bool, str]] = {}
    for item in resp["results"]:
        if not isinstance(item, dict):
            return None
        i, accept, reason = item.get("id"), item.get("accept"), item.get("reason")
        if isinstance(i, bool) or not isinstance(i, int):
            return None
        if not isinstance(accept, bool) or not isinstance(reason, str):
            return None
        if i in out:
            return None
        out[i] = (accept, reason)
    return out if set(out) == ids else None


class Gate:
    """候補の採否を LLM に判定させ、候補ごとに jsonl へキャッシュする。"""

    def __init__(self, cfg: LLMConfig, cache_path: Path, chat=None) -> None:
        self.cfg = cfg
        self.cache_path = cache_path
        self.chat = chat if chat is not None else chat_json
        self.cache: dict[str, tuple[bool, str]] = {}
        self._lock = threading.Lock()
        if cache_path.exists():
            broken = 0
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self.cache[row["hash"]] = (bool(row["accept"]), str(row["reason"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    broken += 1
            if broken:
                print(f"[bgm] ふるい分けのキャッシュの壊れた行を {broken} 件読み飛ばしました: {cache_path}")

    def _key(self, kind: str, gate_text: str, sound_id: int) -> str:
        raw = json.dumps(
            [self.cfg.model, kind, gate_text, sound_id, GATE_PROMPT_VERSION], ensure_ascii=False
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _needs_newline(self) -> bool:
        try:
            with self.cache_path.open("rb") as f:
                f.seek(-1, 2)
                return f.read(1) != b"\n"
        except OSError:
            return False

    def _store(self, rows: list[tuple[str, bool, str]]) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(
                json.dumps({"hash": h, "accept": a, "reason": r}, ensure_ascii=False) + "\n"
                for h, a, r in rows
            )
            if self._needs_newline():
                text = "\n" + text
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(text)
            for h, a, r in rows:
                self.cache[h] = (a, r)

    def judge(
        self, kind: str, gate_text: str, sounds: list[Sound]
    ) -> dict[int, tuple[bool, str]] | None:
        """id → (採否, 理由)。None は LLM の失敗（キャッシュしない）。"""
        keys = {s.id: self._key(kind, gate_text, s.id) for s in sounds}
        with self._lock:
            result = {i: self.cache[k] for i, k in keys.items() if k in self.cache}
        todo = [s for s in sounds if s.id not in result]
        if not todo:
            return result
        system = GATE_SYSTEM_PROMPT_AMBIENT if kind == "ambient" else GATE_SYSTEM_PROMPT_MUSIC
        prompt = build_gate_prompt(gate_text, todo)
        ids = {s.id for s in todo}
        judged = None
        for _ in range(_GATE_ATTEMPTS):
            try:
                judged = _validate_gate(self.chat(self.cfg, system, prompt, GATE_SCHEMA, "bgm_gate"), ids)
            except _CHAT_ERRORS as e:
                print(f"[bgm] ふるい分けの LLM 呼び出しに失敗しました: {e}")
                judged = None
            if judged is not None:
                break
        if judged is None:
            return None
        self._store([(keys[i], a, r) for i, (a, r) in judged.items()])
        result.update(judged)
        return result


def gate_slots(
    slots: list[Slot], found: dict[str, list[Sound]], gate: Gate, workers: int
) -> tuple[dict[str, list[Sound]], list[dict], dict[str, str]]:
    """slot ごとにふるい分ける。戻り値は（残った候補、gate.tsv の行、llm_error の slot）。"""

    def one(s: Slot):
        sounds = found.get(s.key, [])
        rows: list[dict] = []

        def row(sound: Sound, accept: bool, reason: str) -> dict:
            return {
                "chapter": s.chapter, "scene": s.scene, "kind": s.kind,
                "id": sound.id, "name": sound.name, "accept": accept, "reason": reason,
            }

        rest = sounds
        if s.kind == "ambient":
            rest = []
            for sound in sounds:
                word = keyword_reject(sound)
                if word is None:
                    rest.append(sound)
                else:
                    rows.append(row(sound, False, f"keyword: {word}"))
        if not rest:
            return s.key, [], rows, False
        verdict = gate.judge(s.kind, s.gate_text, rest)
        if verdict is None:
            return s.key, [], rows, True
        kept = []
        for sound in rest:
            accept, reason = verdict[sound.id]
            rows.append(row(sound, accept, reason))
            if accept:
                kept.append(sound)
        return s.key, kept, rows, False

    survivors: dict[str, list[Sound]] = {}
    all_rows: list[dict] = []
    llm_error: dict[str, str] = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for key, kept, rows, failed in pool.map(one, slots):
            survivors[key] = kept
            all_rows.extend(rows)
            if failed:
                llm_error[key] = "llm_error"
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return survivors, all_rows, llm_error


def write_gate_tsv(path: Path, rows: list[dict]) -> None:
    def cell(v: object) -> str:
        return _one_line(str(v))

    lines = ["\t".join(["chapter", "scene", "kind", "id", "name", "accept", "reason"])]
    for r in rows:
        lines.append("\t".join([
            cell(r["chapter"]), cell(r["scene"]), cell(r["kind"]), cell(r["id"]),
            cell(r["name"]), "accept" if r["accept"] else "reject", cell(r["reason"]),
        ]))
    _write_text(path, "\n".join(lines) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------- CLAP ----------


def window_starts(duration: float, window: float = 10.0, max_windows: int = 6) -> list[float]:
    """採点する区間の開始秒。0 から duration - window までを等分する。"""
    if duration <= window:
        return [0.0]
    n = min(max_windows, math.ceil(duration / window))
    last = duration - window
    return [last * i / (n - 1) for i in range(n)]


class ClapScorer:
    """CLAP の音声埋め込みと文の埋め込みの内積で点を付ける。torch はここでだけ import する。"""

    def __init__(self, model: str, device: str) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "bgm.device が cuda ですが、CUDA が使えません。"
                "bgm.device を cpu（Mac では mps）にするか、GPU のある環境で実行してください。"
            )
        self.model = model
        self.device = device
        self._torch = torch
        self._processor = ClapProcessor.from_pretrained(model)
        self._clap = ClapModel.from_pretrained(model).to(device).eval()
        self._text: dict[str, object] = {}

    @staticmethod
    def _tensor(out):
        return getattr(out, "pooler_output", out)

    def _text_embedding(self, text: str):
        emb = self._text.get(text)
        if emb is None:
            torch = self._torch
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self._clap.get_text_features(**inputs)
            emb = torch.nn.functional.normalize(self._tensor(out), dim=-1)[0]
            self._text[text] = emb
        return emb

    def score(self, path: Path, duration: float, text: str) -> float:
        torch = self._torch
        target = self._text_embedding(text)
        points: list[float] = []
        for start in window_starts(duration):
            audio = audio_io.decode(path, _CLAP_SR, 1, start, 10.0)
            if len(audio) == 0:
                continue
            inputs = self._processor(audio=audio, sampling_rate=_CLAP_SR, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._clap.get_audio_features(**inputs)
            emb = torch.nn.functional.normalize(self._tensor(out), dim=-1)[0]
            points.append(float((emb * target).sum()))
        if not points:
            raise ValueError(f"音声を読めませんでした: {path}")
        return sum(points) / len(points)


# ---------- 選択 ----------


def choose(
    slots: list[Slot],
    found: dict[str, list[Sound]],
    survivors: dict[str, list[Sound]],
    failures: dict[str, str],
    scorer,
    scores_path: Path,
    exclude: set[int],
    tmp_dir: Path,
    workers: int,
    download=download_preview,
) -> list[Choice]:
    """slot ごとに、ふるい分けで残った候補から CLAP の最高点を選ぶ。"""
    scores = _read_json(scores_path)
    model = getattr(scorer, "model", "")
    choices: list[Choice] = []

    def make(s: Slot, sound: Sound | None, score: float | None, reason: str | None) -> Choice:
        return Choice(s.chapter, s.scene, s.kind, s.query, sound, score, reason)

    for s in slots:
        if failures.get(s.key) == "error":
            choices.append(make(s, None, None, "error"))
            continue
        if not found.get(s.key):
            choices.append(make(s, None, None, "no_result"))
            continue
        if failures.get(s.key) == "llm_error":
            choices.append(make(s, None, None, "llm_error"))
            continue
        cands = [x for x in survivors.get(s.key, []) if x.id not in exclude]
        if not cands:
            choices.append(make(s, None, None, "all_rejected"))
            continue

        def ck(sound: Sound) -> str:
            return f"{sound.id}|{s.clap_text}|{model}"

        todo = [x for x in cands if ck(x) not in scores]
        if todo:
            tmp_dir.mkdir(parents=True, exist_ok=True)

            def fetch(sound: Sound) -> Path | None:
                dst = tmp_dir / f"{sound.id}.mp3"
                try:
                    download(sound.preview, dst)
                except (FreesoundError, OSError) as e:
                    print(f"[bgm] 試聴音声を取得できませんでした（{sound.id}）: {e}")
                    return None
                return dst

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
            try:
                paths = list(pool.map(fetch, todo))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            for sound, path in zip(todo, paths):
                if path is None:
                    continue
                try:
                    scores[ck(sound)] = float(scorer.score(path, sound.duration, s.clap_text))
                except Exception as e:  # 壊れた音声 1 つで全体を止めない
                    print(f"[bgm] 採点できませんでした（{sound.id}）: {e}")
                finally:
                    path.unlink(missing_ok=True)
            _write_json(scores_path, scores)

        scored = [(scores[ck(x)], x) for x in cands if ck(x) in scores]
        if not scored:
            choices.append(make(s, None, None, "error"))
            continue
        best_score, best = max(scored, key=lambda t: t[0])  # 同点は先頭
        choices.append(make(s, best, best_score, None))
    return choices


def read_exclude(path: Path) -> set[int]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    ids: set[int] = set()
    for line in text.splitlines():
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        try:
            ids.add(int(token))
        except ValueError:
            print(f"[bgm] exclude.txt の行を読み飛ばしました（ID は整数）: {token!r}")
    return ids


def write_selection(path: Path, choices: list[Choice], keep_chapters: set[int]) -> None:
    """今回の章を置き換え、keep_chapters に無い章を落とす。"""
    data = _read_json(path)
    new: dict[str, list[dict]] = {}
    for c in choices:
        new.setdefault(str(c.chapter), []).append(c.to_dict())
    data.update(new)
    data = {k: v for k, v in data.items() if k.lstrip("-").isdigit() and int(k) in keep_chapters}
    _write_json(path, dict(sorted(data.items(), key=lambda kv: int(kv[0]))))


def load_selection(path: Path) -> list[Choice]:
    data = _read_json(path)
    out: list[Choice] = []
    for key in sorted(data, key=int):
        out.extend(Choice.from_dict(d) for d in data[key])
    return out


def write_missing(path: Path, choices: list[Choice], scenes: dict[int, list[Scene]]) -> None:
    lines = ["\t".join(["chapter", "scene", "start_para", "kind", "description", "query", "reason"])]
    for c in choices:
        if c.reason is None:
            continue
        sc = next((x for x in scenes.get(c.chapter, []) if x.index == c.scene), None)
        if sc is None or (c.kind == "ambient" and sc.ambient_query is None):
            continue  # 意図した無音
        lines.append("\t".join([
            str(c.chapter), str(c.scene), str(sc.start_para), c.kind,
            _one_line(sc.description), _one_line(c.query), c.reason,
        ]))
    _write_text(path, "\n".join(lines) + "\n")
