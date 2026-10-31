from irodori_speaker.bgm_types import Choice, Scene, Sound, license_label


def test_license_label():
    assert license_label("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0 1.0"
    assert license_label("https://creativecommons.org/licenses/by/4.0/") == "CC BY 4.0"
    assert license_label("http://creativecommons.org/licenses/by-nc/3.0/") == "CC BY-NC 3.0"
    assert license_label("https://example.com/x") == "https://example.com/x"


def test_choice_roundtrip():
    s = Sound(1, "n", ["a"], "d", 30.0, "http://creativecommons.org/licenses/by/4.0/", "u", "url", "p")
    c = Choice(0, 1, "ambient", "q", s, 0.3, None)
    assert Choice.from_dict(c.to_dict()) == c
    assert isinstance(c.to_dict()["sound"], dict)


def test_choice_roundtrip_without_sound():
    c = Choice(0, 1, "bridge", "q", None, None, "no_result")
    assert Choice.from_dict(c.to_dict()) == c


def test_scene_roundtrip_and_default_status():
    sc = Scene(0, 1, 2, 5, "a forest", None, "calm", "")
    assert sc.status == "ok"
    assert Scene.from_dict(sc.to_dict()) == sc
