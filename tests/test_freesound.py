import pytest
import requests

from irodori_speaker.bgm_types import Sound
from irodori_speaker.freesound import (
    FIELDS,
    DailyLimit,
    FreesoundClient,
    FreesoundError,
    KeyMissing,
    cached_preview,
    download_preview,
    load_key,
)

KEY = "k"
SECRET = "s3cretKEYvalue"


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, content=b""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.content = content

    def json(self):
        return self._payload


class FakeSession:
    """requests.Session 互換の偽物。get の呼び出しを記録し、用意した応答を順に返す。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, **kw):
        self.calls.append({"url": url, "params": params, "headers": headers or {}})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def result(i, **over):
    d = {
        "id": i,
        "name": f"n{i}.wav",
        "tags": ["a", "b"],
        "description": "d",
        "duration": 60.0,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "username": "u",
        "url": f"https://freesound.org/people/u/sounds/{i}/",
        "previews": {"preview-hq-mp3": f"https://cdn.freesound.org/p/{i}.mp3"},
    }
    d.update(over)
    return d


def make(responses, key=KEY, **kw):
    t = FakeTime()
    s = FakeSession(responses)
    c = FreesoundClient(key, session=s, sleep=t.sleep, clock=t.clock, **kw)
    return c, s, t


def ok(*ids):
    return FakeResponse(200, {"results": [result(i) for i in ids]})


def test_load_key_reads_line_and_never_echoes(tmp_path):
    p = tmp_path / "s.env"
    p.write_text("X=1\nFREESOUND_API_KEY=abc123\n")
    assert load_key(p) == "abc123"
    p.write_text("X=1\n")
    with pytest.raises(KeyMissing) as e:
        load_key(p)
    assert "abc123" not in str(e.value)
    assert str(p) in str(e.value)
    assert "chmod 600" in str(e.value)


def test_load_key_missing_file(tmp_path):
    p = tmp_path / "none.env"
    with pytest.raises(KeyMissing) as e:
        load_key(p)
    assert str(p) in str(e.value)


def test_load_key_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "s.env").write_text("FREESOUND_API_KEY=zzz\n")
    assert load_key("~/s.env") == "zzz"


def test_load_key_empty_value_is_missing(tmp_path):
    p = tmp_path / "s.env"
    p.write_text("FREESOUND_API_KEY=\n")
    with pytest.raises(KeyMissing):
        load_key(p)


def test_search_sends_token_header_and_fields():
    c, s, _ = make([ok(1, 2)], key=SECRET)
    sounds = c.search("rain", "[30 TO 900]", 12)
    call = s.calls[0]
    assert call["url"] == "https://freesound.org/apiv2/search/"
    assert call["params"]["query"] == "rain"
    assert call["params"]["filter"] == "duration:[30 TO 900]"
    assert call["params"]["fields"] == FIELDS
    assert call["params"]["page_size"] == 12
    assert call["headers"]["Authorization"] == f"Token {SECRET}"
    assert SECRET not in call["url"]
    assert SECRET not in str(call["params"])
    assert [x.id for x in sounds] == [1, 2]
    assert isinstance(sounds[0], Sound)
    assert sounds[0].preview == "https://cdn.freesound.org/p/1.mp3"
    assert sounds[0].license == "https://creativecommons.org/licenses/by/4.0/"
    assert sounds[0].tags == ["a", "b"]
    assert sounds[0].username == "u"


def test_search_skips_results_without_preview():
    r = FakeResponse(200, {"results": [result(1, previews={}), result(2)]})
    c, _, _ = make([r])
    assert [x.id for x in c.search("q", "[1 TO 2]", 5)] == [2]


def test_search_falls_back_to_text_endpoint_on_404():
    c, s, _ = make([FakeResponse(404), ok(7)])
    sounds = c.search("q", "[30 TO 900]", 3)
    assert [x.id for x in sounds] == [7]
    assert s.calls[1]["url"] == "https://freesound.org/apiv2/search/text/"
    assert s.calls[1]["headers"]["Authorization"] == "Token k"


def test_minute_limit_429_waits_and_retries():
    r429 = FakeResponse(429, {"detail": "Request was throttled (minute limit)"}, {"Retry-After": "5"})
    c, s, t = make([r429, ok(1)])
    sounds = c.search("q", "[1 TO 2]", 1)
    assert [x.id for x in sounds] == [1]
    assert 5 in t.sleeps
    assert len(s.calls) == 2


def test_minute_limit_429_without_retry_after_waits_60():
    r429 = FakeResponse(429, {"detail": "minute"})
    c, _, t = make([r429, ok(1)])
    c.search("q", "[1 TO 2]", 1)
    assert 60 in t.sleeps


def test_daily_limit_raises_immediately():
    r429 = FakeResponse(429, {"detail": "Your daily limit of 2000 requests per day was exceeded"})
    c, s, t = make([r429])
    with pytest.raises(DailyLimit):
        c.search("q", "[1 TO 2]", 1)
    assert t.sleeps == []
    assert len(s.calls) == 1


def test_retry_after_over_120_treated_as_daily():
    r429 = FakeResponse(429, {"detail": "throttled"}, {"Retry-After": "3600"})
    c, _, t = make([r429])
    with pytest.raises(DailyLimit):
        c.search("q", "[1 TO 2]", 1)
    assert t.sleeps == []


def test_three_minute_429s_raise_freesound_error():
    r = lambda: FakeResponse(429, {"detail": "minute"}, {"Retry-After": "2"})  # noqa: E731
    c, s, _ = make([r(), r(), r()])
    with pytest.raises(FreesoundError) as e:
        c.search("q", "[1 TO 2]", 1)
    assert not isinstance(e.value, DailyLimit)
    assert len(s.calls) == 3


def test_other_http_error_message_has_no_key():
    c, _, _ = make([FakeResponse(500)], key=SECRET)
    with pytest.raises(FreesoundError) as e:
        c.search("q", "[1 TO 2]", 1)
    msg = str(e.value)
    assert "HTTP 500" in msg
    assert "Token" not in msg
    assert SECRET not in msg


def test_request_exception_is_wrapped_without_key():
    # requests の例外の repr にヘッダーが入っていても、連鎖させない
    boom = requests.ConnectionError(f"Authorization: Token {SECRET}")
    c, _, _ = make([boom], key=SECRET)
    with pytest.raises(FreesoundError) as e:
        c.search("q", "[1 TO 2]", 1)
    assert SECRET not in str(e.value)
    assert e.value.__cause__ is None
    assert e.value.__suppress_context__ is True


def test_client_repr_has_no_key():
    c, _, _ = make([], key=SECRET)
    assert SECRET not in repr(c)
    assert SECRET not in str(c)


def test_min_interval_between_calls():
    c, _, t = make([ok(1), ok(2)])
    c.search("a", "[1 TO 2]", 1)
    assert t.sleeps == []
    c.search("b", "[1 TO 2]", 1)
    assert len(t.sleeps) == 1
    assert t.sleeps[0] == pytest.approx(1.0)


def test_min_interval_skips_sleep_when_enough_time_passed():
    c, _, t = make([ok(1), ok(2)])
    c.search("a", "[1 TO 2]", 1)
    t.now += 5
    c.search("b", "[1 TO 2]", 1)
    assert t.sleeps == []


def test_download_preview_sends_no_auth(tmp_path):
    s = FakeSession([FakeResponse(200, content=b"MP3DATA")])
    dst = tmp_path / "sub" / "1.mp3"
    download_preview("https://cdn.freesound.org/p/1.mp3", dst, session=s)
    assert dst.read_bytes() == b"MP3DATA"
    assert s.calls[0]["url"] == "https://cdn.freesound.org/p/1.mp3"
    assert "Authorization" not in s.calls[0]["headers"]
    assert not list(dst.parent.glob("*.tmp"))


def test_download_preview_failure_leaves_no_file(tmp_path):
    s = FakeSession([FakeResponse(404)])
    dst = tmp_path / "1.mp3"
    with pytest.raises(FreesoundError):
        download_preview("https://cdn.freesound.org/p/1.mp3", dst, session=s)
    assert not dst.exists()
    assert list(tmp_path.iterdir()) == []


def _sound(sid=7):
    return Sound(sid, "n", [], "", 1.0, "", "u", "https://freesound.org/s/7/", "https://cdn.freesound.org/p/7.mp3")


def test_cached_preview_downloads_once(tmp_path):
    calls = []

    def dl(url, dst):
        calls.append((url, dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")

    cache = tmp_path / "c" / "previews"
    p1 = cached_preview(_sound(), cache, dl)
    p2 = cached_preview(_sound(), cache, dl)
    assert p1 == p2 == cache / "7.mp3" and p1.read_bytes() == b"x"
    assert calls == [("https://cdn.freesound.org/p/7.mp3", cache / "7.mp3")]


def test_cached_preview_failure_leaves_no_file(tmp_path):
    def dl(url, dst):
        raise FreesoundError("x")

    with pytest.raises(FreesoundError):
        cached_preview(_sound(), tmp_path, dl)
    assert list(tmp_path.iterdir()) == []
