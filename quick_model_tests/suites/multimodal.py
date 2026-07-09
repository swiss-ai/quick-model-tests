"""multimodal suite -- image & audio inputs. See SPEC.md 7.4.

Open question 1 RESOLVED (2026-06): the swissai endpoint accepts OpenAI-style
content parts:
  - image: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  - audio: {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
            (note: `audio_url`, not OpenAI's `input_audio`).

Determinism via embedded sentinels (SPEC.md 7.4): the fixture images carry a
numeric code (4827 / 1593) and the audio says a fixed pangram, so a pass means
the modality was actually read -- assert the sentinel/keyword appears in content.

Target a multimodal Apertus, e.g.
  --model swiss-ai/Apertus-1.5-8B-SFT-RL-DPO-SDPO-Mix-Less-Refuse-Feedback
(the whole Apertus-1.5 "omni" family reads images/audio). The suite HARD-FAILS
(via the `mm_supported` probe) when the model cannot read a sentinel image -- no
silent skips, matching the tools suite.
"""

import base64
import re
from pathlib import Path

import pytest

from quick_model_tests.client import ApiError, ChatClient
from quick_model_tests.suites.core import _discover_bos_from_chat, _leading_bos_count

pytestmark = pytest.mark.multimodal

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Special / control tokens that must never leak into user-visible content.
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>|</?(?:think|info|bash)\b", re.IGNORECASE)

MAX_TOKENS = 2 ** 14  # a generous budget for multimodal tests, see core.py / SPEC.md 7.6


def _data_url(name: str, mime: str) -> str:
    raw = (ASSETS / name).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _image(name: str, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": _data_url(name, mime)}}


def _audio(name: str, mime: str = "audio/wav") -> dict:
    return {"type": "audio_url", "audio_url": {"url": _data_url(name, mime)}}


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _content(resp: dict) -> str:
    return (ChatClient.content(resp) or "").strip()


@pytest.fixture(scope="session")
def mm_supported(client):
    """Probe once: can the model read a numeric sentinel out of an image?

    HARD FAIL (not skip) when the target model lacks vision -- pointing this
    suite at a text-only model is an error so the gate goes red, never silently
    green. Use a multimodal build (see module doc).
    """
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        _text("What number is written in this image? Digits only."),
                        _image("image_4827.png"),
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )
    except ApiError as e:
        pytest.fail(f"model {client.config.model!r} rejected image input ({e.status})")
    if "4827" not in _content(resp):
        pytest.fail(
            f"model {client.config.model!r} could not read the image sentinel "
            f"(no vision?): {_content(resp)!r}"
        )
    return True


def test_mm_image_small(client, mm_supported):
    """mm-image-small: small image with a sentinel -> sentinel present, no leak."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [
                    _text("What number is written in this image? Digits only."),
                    _image("image_4827.png"),
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp)
    assert "4827" in content, f"sentinel not read from image: {content!r}"
    assert not SPECIAL_TOKEN_RE.search(content), f"token leak: {content!r}"


def test_mm_image_large(client, mm_supported):
    """mm-image-large: a large image is accepted and answered well-formed."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [
                    _text("What number is written in this image? Digits only."),
                    _image("image_large.png"),
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp)
    assert content, "empty response for large image"
    assert "4827" in content, f"sentinel not read from large image: {content!r}"


def test_mm_image_multi(client, mm_supported):
    """mm-image-multi: two images with distinct sentinels -> both appear."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [
                    _text(
                        "Two images follow, each shows a number. Report BOTH numbers."
                    ),
                    _image("image_4827.png"),
                    _image("image_1593.png"),
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp)
    assert "4827" in content and "1593" in content, (
        f"both sentinels expected, got: {content!r}"
    )


def test_mm_audio_small(client, mm_supported):
    """mm-audio-small: short audio clip transcribed -> keyword sentinel present."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [_text("Transcribe the audio."), _audio("audio_fox.wav")],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp).lower()
    assert "fox" in content, f"audio sentinel 'fox' not transcribed: {content!r}"


def test_mm_audio_large(client, mm_supported):
    """mm-audio-large: a larger audio clip is accepted and transcribed."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [_text("Transcribe the audio."), _audio("audio_large.wav")],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp).lower()
    assert content, "empty response for large audio"
    assert "fox" in content, f"audio sentinel 'fox' not transcribed: {content!r}"


def test_mm_interleaved(client, mm_supported):
    """mm-interleaved: text + image + audio in one message -> well-formed, no leak."""
    resp = client.chat(
        [
            {
                "role": "user",
                "content": [
                    _text("Read the number in the image and transcribe the audio."),
                    _image("image_4827.png"),
                    _audio("audio_fox.wav"),
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    content = _content(resp)
    assert content, "empty response for interleaved image+audio"
    assert not SPECIAL_TOKEN_RE.search(content), f"token leak: {content!r}"


def _assert_single_bos_in_mm_chat(client, part, kind):
    """Tokenize a `<text> + <modality>` chat through the server's own template and
    assert exactly one leading BOS. The path apertus-program #420 was reported on:
    vLLM's multimodal tokenization restores ``add_special_tokens=True`` (via
    ``mm_processor.info.default_tok_params``), so if the chat template also emits
    ``{{ bos_token }}`` the rendered prompt starts ``<s><s>``. Skips if
    ``/tokenize`` does not accept multimodal messages or the model has no BOS.

    (Best-effort proxy: ``/tokenize`` may not traverse the same mm code path as
    generation, but it is the only observable surface for the rendered mm prompt.)
    """
    bos_id, bos_str, _ = _discover_bos_from_chat(client)
    messages = [{"role": "user", "content": [_text("Describe this input."), part]}]
    try:
        ids = client.tokenize_chat(messages)
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept multimodal messages: {exc}")
    if not ids:
        pytest.skip("multimodal chat tokenization returned no tokens")
    count = _leading_bos_count(ids, bos_id)
    assert count == 1, (
        f"{kind} chat prompt must begin with exactly one BOS (id {bos_id}, "
        f"{bos_str!r}), got {count} (first ids {ids[:6]}). Two is the multimodal "
        f"double-BOS from `default_tok_params` restoring add_special_tokens=True "
        f"(apertus-program #420)."
    )


def test_mm_bos_single_in_chat(client):
    """mm-bos-single-in-chat: an image+text chat prompt begins with exactly one BOS.

    The mm counterpart to ``core-no-double-bos-chat`` -- the exact path #420 was
    reported on. See ``_assert_single_bos_in_mm_chat``.
    """
    _assert_single_bos_in_mm_chat(client, _image("image_4827.png"), "image")


def test_mm_bos_single_in_chat_audio(client):
    """mm-bos-single-in-chat-audio: an audio+text chat prompt begins with one BOS.

    Audio triggers the same ``default_tok_params`` mm path as image, so it gets its
    own guard. See ``_assert_single_bos_in_mm_chat``.
    """
    _assert_single_bos_in_mm_chat(client, _audio("audio_fox.wav"), "audio")
