from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

from .bgm_types import Sound

API_BASE = "https://freesound.org/apiv2"
FIELDS = "id,name,tags,description,duration,license,previews,username,url"
KEY_NAME = "FREESOUND_API_KEY"

_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER = 120.0
_DEFAULT_RETRY_AFTER = 60.0
_TIMEOUT = 30
_DAILY_RE = re.compile(r"\bday\b|daily", re.I)

_KEY_MISSING = """Freesound の API キーが見つかりません: {path}
自分の端末で次を実行して作ってください（キーは画面に表示されません）:
  mkdir -p ~/.config/irodori-speaker
  read -rsp 'Freesound API key: ' k; echo
  printf 'FREESOUND_API_KEY=%s\\n' "$k" > ~/.config/irodori-speaker/secrets.env; unset k
  chmod 600 ~/.config/irodori-speaker/secrets.env"""


class KeyMissing(Exception):
    pass


class DailyLimit(Exception):
    pass


class FreesoundError(Exception):
    pass


def load_key(path: str | Path) -> str:
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        raise KeyMissing(_KEY_MISSING.format(path=p)) from None
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == KEY_NAME and value.strip():
            return value.strip()
    raise KeyMissing(_KEY_MISSING.format(path=p))


def _retry_after(resp) -> float | None:
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _is_daily(resp, retry_after: float | None) -> bool:
    if retry_after is not None and retry_after > _MAX_RETRY_AFTER:
        return True
    try:
        detail = str(resp.json().get("detail", ""))
    except Exception:
        return False
    return bool(_DAILY_RE.search(detail))


def _to_sound(r: dict) -> Sound | None:
    preview = (r.get("previews") or {}).get("preview-hq-mp3")
    if not preview:
        return None
    return Sound(
        id=int(r["id"]),
        name=r.get("name", ""),
        tags=list(r.get("tags") or []),
        description=r.get("description") or "",
        duration=float(r.get("duration") or 0.0),
        license=r.get("license", ""),
        username=r.get("username", ""),
        url=r.get("url", ""),
        preview=preview,
    )


class FreesoundClient:
    def __init__(
        self,
        key: str,
        *,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        min_interval: float = 1.0,
    ):
        self._key = key
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._min_interval = min_interval
        self._last: float | None = None

    def __repr__(self) -> str:
        return "FreesoundClient()"

    def _wait_interval(self) -> None:
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def _get(self, path: str, params: dict):
        """1 回の API 呼び出し（429 の待ち直しを含む）。404 も応答として返す。"""
        for attempt in range(_MAX_ATTEMPTS):
            self._wait_interval()
            try:
                resp = self._session.get(
                    API_BASE + path,
                    params=params,
                    headers={"Authorization": f"Token {self._key}"},
                    timeout=_TIMEOUT,
                )
            except requests.RequestException:
                raise FreesoundError(
                    f"Freesound の検索に失敗しました: 通信エラー（{path}）"
                ) from None
            if resp.status_code != 429:
                return resp
            after = _retry_after(resp)
            if _is_daily(resp, after):
                raise DailyLimit("Freesound の 1 日の上限に達しました")
            if attempt == _MAX_ATTEMPTS - 1:
                break
            self._sleep(min(after, _MAX_RETRY_AFTER) if after is not None else _DEFAULT_RETRY_AFTER)
        raise FreesoundError(f"Freesound の検索に失敗しました: HTTP 429（{path}）")

    def search(self, query: str, duration: str, n: int) -> list[Sound]:
        params = {
            "query": query,
            "filter": f"duration:{duration}",
            "fields": FIELDS,
            "page_size": n,
        }
        path = "/search/"
        resp = self._get(path, params)
        if resp.status_code == 404:
            path = "/search/text/"
            resp = self._get(path, params)
        if resp.status_code != 200:
            raise FreesoundError(
                f"Freesound の検索に失敗しました: HTTP {resp.status_code}（{path}）"
            )
        try:
            results = resp.json().get("results", [])
        except Exception:
            raise FreesoundError(
                f"Freesound の検索に失敗しました: 応答が不正です（{path}）"
            ) from None
        sounds = (_to_sound(r) for r in results)
        return [s for s in sounds if s is not None]


def download_preview(url: str, dst: Path, session: requests.Session | None = None) -> None:
    """試聴用 MP3 を取得する。キーは付けない（API 以外のホストに送らない）。"""
    session = session if session is not None else requests.Session()
    dst = Path(dst)
    try:
        resp = session.get(url, timeout=_TIMEOUT)
    except requests.RequestException:
        raise FreesoundError("試聴音声の取得に失敗しました: 通信エラー") from None
    if resp.status_code != 200:
        raise FreesoundError(f"試聴音声の取得に失敗しました: HTTP {resp.status_code}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_bytes(resp.content)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def cached_preview(sound: Sound, cache_dir: Path, download=download_preview) -> Path:
    """試聴 MP3 を本をまたぐキャッシュ（<cache_dir>/<ID>.mp3）に置いて返す。あれば取得しない。"""
    dst = Path(cache_dir) / f"{sound.id}.mp3"
    if not dst.exists():
        download(sound.preview, dst)  # 一時ファイル経由で書くので、失敗しても壊れたファイルは残らない
    return dst
