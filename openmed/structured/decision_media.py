"""Verified local decision media. No remote resolution, raw-media files or logging."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from openmed.multimodal.asset_limits import MOBILE_V1
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.media_type import detect_media_type
from openmed.multimodal.preflight import PreflightStatus, preflight_asset

MAX_MEDIA_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 16 * 1024 * 1024
MAX_MEDIA_ITEMS = 4
_LIMITS = MOBILE_V1.with_limits(
    max_byte_size=MAX_MEDIA_BYTES,
    max_pixels=2_000_000,
    max_total_pixels=16_000_000,
    max_frames=64,
    max_duration_seconds=30,
)
SUPPORTED_TYPES = frozenset({"image/png", "image/jpeg", "video/mp4"})


class DecisionMediaError(ValueError):
    """Fixed-code refusal containing no media bytes, URLs or local paths."""


def _need(condition: bool, code: str = "decision_media_invalid") -> None:
    if not condition:
        raise DecisionMediaError(code)


def _admit(data: bytes, mime: str, digest: str, **metadata: Any) -> None:
    manifest = AssetManifest(
        asset_id="decision-media",
        media_type=mime,
        sha256=digest,
        byte_size=len(data),
        **metadata,
    )
    report = preflight_asset(manifest, data, limit_profile=_LIMITS)
    _need(report.status is PreflightStatus.ACCEPT, "decision_media_preflight_failed")


def _pipe(tool: str, args: list[str], data: bytes, cap: int) -> bytes:
    """Bound input, stdout, elapsed time and process lifetime; discard decoder logs."""
    exe = shutil.which(tool)
    _need(exe is not None, "decision_media_video_tools_unavailable")
    assert exe is not None
    # Never inherit FFREPORT or protocol/codec environment overrides from a host.
    # Commands contain only fixed arguments and validated counts, never media or paths.
    env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR") if k in os.environ}
    proc = subprocess.Popen(
        [exe, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def write() -> None:
        assert proc.stdin is not None
        try:
            with proc.stdin:
                proc.stdin.write(data)
        except (OSError, ValueError):
            pass

    writer = threading.Thread(target=write, daemon=True)
    watchdog = threading.Timer(15, proc.kill)
    watchdog.daemon = True
    watchdog.start()
    writer.start()
    try:
        with proc.stdout:
            result = proc.stdout.read(cap + 1)
        _need(len(result) <= cap, "decision_media_output_limit")
        _need(proc.wait() == 0, "decision_media_video_decode_failed")
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        writer.join()
    return result


@dataclass(frozen=True, slots=True)
class _Video:
    width: int
    height: int
    frames: int
    fps: Fraction


_INPUT = [
    "-v",
    "error",
    "-max_alloc",
    "67108864",
    "-protocol_whitelist",
    "pipe",
    "-f",
    "mov",
    "-enable_drefs",
    "0",
    "-use_absolute_path",
    "0",
]


def _video_info(data: bytes) -> _Video:
    blob = _pipe(
        "ffprobe",
        [
            *_INPUT,
            "-show_packets",
            "-show_streams",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,time_base,sample_aspect_ratio:stream_side_data=rotation:packet=pts,duration,stream_index:stream_tags=:format_tags=:packet_side_data=",
            "-of",
            "json",
            "pipe:0",
        ],
        data,
        32768,
    )
    info = json.loads(blob)
    _need(len(info["streams"]) == 1, "decision_media_video_tracks")
    stream = info["streams"][0]
    _need(
        stream.get("codec_type") == "video"
        and stream.get("codec_name") in {"h264", "mpeg4", "mjpeg"},
        "decision_media_unsupported",
    )
    _need(
        stream.get("sample_aspect_ratio", "1:1") in {"1:1", "N/A"}
        and not any(
            row.get("rotation", 0) != 0 for row in stream.get("side_data_list", ())
        ),
        "decision_media_video_orientation",
    )
    w, h = stream["width"], stream["height"]
    packets = info["packets"]
    _need(
        type(w) is int
        and type(h) is int
        and 0 < w <= 4096
        and 0 < h <= 4096
        and type(packets) is list
        and 1 <= len(packets) <= 64
        and w * h <= _LIMITS.max_pixels
        and w * h * len(packets) <= _LIMITS.max_total_pixels,
        "decision_media_limit",
    )
    rate = Fraction(stream["avg_frame_rate"])
    units = Fraction(stream["time_base"])
    _need(
        0 < rate <= 60 and units > 0 and len(packets) / rate <= 30,
        "decision_media_video_timing",
    )
    step = 1 / (units * rate)
    times = sorted(p["pts"] for p in packets)
    _need(
        all(
            p.get("stream_index") == 0
            and type(p.get("duration")) is int
            and p["duration"] == step
            for p in packets
        )
        and times == [i * step for i in range(len(times))],
        "decision_media_video_timing",
    )
    return _Video(w, h, len(packets), rate)


def _rgb(data: bytes, info: _Video) -> bytes:
    size = info.width * info.height * info.frames * 3
    pixels = _pipe(
        "ffmpeg",
        [
            *_INPUT,
            "-threads",
            "1",
            "-noautorotate",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-threads",
            "1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        data,
        size,
    )
    _need(len(pixels) == size, "decision_media_video_frame_mismatch")
    return pixels


def _scrub(data: bytes, mime: str, digest: str) -> tuple[bytes, str]:
    if mime.startswith("image/"):
        from PIL import Image

        from openmed.multimodal.metadata_scrub import scrub_raster_pixels

        with Image.open(io.BytesIO(data), formats=["PNG", "JPEG"]) as img:
            _admit(data, mime, digest, width=img.width, height=img.height)
            _need(
                getattr(img, "n_frames", 1) == 1
                and img.mode in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"},
                "decision_media_unsupported",
            )
            # Apply EXIF orientation before detaching all source metadata.
            clean = scrub_raster_pixels(img)
            out = io.BytesIO()
            clean.save(out, format="PNG")
            result = out.getvalue()
            _need(len(result) <= MAX_MEDIA_BYTES, "decision_media_output_limit")
            return result, "image/png"
    info = _video_info(data)
    _admit(
        data,
        mime,
        digest,
        width=info.width,
        height=info.height,
        frames=info.frames,
        duration_seconds=float(info.frames / info.fps),
    )
    pixels = _rgb(data, info)
    # Rebuild from pixels and the verified constant frame clock, never remux
    # unknown metadata, auxiliary tracks, codec user-data or file references.
    clean_video = _pipe(
        "ffmpeg",
        [
            "-v",
            "error",
            "-protocol_whitelist",
            "pipe",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{info.width}x{info.height}",
            "-r",
            str(info.fps),
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-an",
            "-sn",
            "-dn",
            "-threads",
            "1",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-bf",
            "0",
            "-movflags",
            "empty_moov+frag_keyframe+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ],
        pixels,
        MAX_MEDIA_BYTES,
    )
    verified = _video_info(clean_video)
    _need(
        verified == info and _rgb(clean_video, verified) == pixels,
        "decision_media_video_verification_failed",
    )
    return clean_video, "video/mp4"


@dataclass(frozen=True, slots=True, init=False)
class DecisionMedia:
    """A size- and digest-checked, metadata-scrubbed local decision input."""

    media_type: str
    sha256: str
    byte_size: int
    _payload: bytes = field(repr=False)

    def __init__(
        self, source: bytes | Path, *, media_type: str, sha256: str, byte_size: int
    ):
        _need(
            type(byte_size) is int and 0 < byte_size <= MAX_MEDIA_BYTES,
            "decision_media_limit",
        )
        _need(
            type(media_type) is str and media_type in SUPPORTED_TYPES,
            "decision_media_unsupported",
        )
        _need(type(sha256) is str and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None)
        try:
            if isinstance(source, Path):
                _need(
                    source.is_absolute() and not str(source).startswith(("//", "\\\\")),
                    "decision_media_local_source_required",
                )
                with os.fdopen(
                    os.open(
                        source,
                        os.O_RDONLY
                        | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    ),
                    "rb",
                ) as f:
                    st = os.fstat(f.fileno())
                    _need(stat.S_ISREG(st.st_mode) and st.st_size == byte_size)
                    data = f.read(byte_size + 1)
            else:
                _need(type(source) is bytes, "decision_media_local_source_required")
                data = source
            _need(
                len(data) == byte_size and hashlib.sha256(data).hexdigest() == sha256,
                "decision_media_digest_mismatch",
            )
            _need(detect_media_type(data) == media_type, "decision_media_type_mismatch")
            blob, kind = _scrub(data, media_type, sha256)
        except DecisionMediaError:
            raise
        except Exception:
            pass
        else:
            object.__setattr__(self, "media_type", kind)
            object.__setattr__(self, "_payload", blob)
            object.__setattr__(self, "byte_size", len(blob))
            object.__setattr__(self, "sha256", hashlib.sha256(blob).hexdigest())
            return
        raise DecisionMediaError("decision_media_invalid")

    def safe_metadata(self) -> dict[str, Any]:
        """Describe only the sanitized bytes that will actually be sent."""
        return {
            "media_type": self.media_type,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }

    def protected_bytes(self) -> bytes:
        """Return the sanitized protected payload to a trusted local backend."""
        return self._payload

    def data_url(self) -> str:
        """Encode protected media for the authorized local wire call."""
        return f"data:{self.media_type};base64," + base64.b64encode(
            self._payload
        ).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        """Serialize protected inline transport; never use for audit or logs."""
        return {
            **self.safe_metadata(),
            "data": base64.b64encode(self._payload).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> DecisionMedia:
        """Parse inline bytes only. Never open filesystem paths from a wire request."""
        _need(
            type(row) is dict
            and set(row) == {"data", "byte_size", "media_type", "sha256"}
        )
        n = row["byte_size"]
        b64 = row["data"]
        _need(
            type(n) is int
            and 0 < n <= MAX_MEDIA_BYTES
            and type(b64) is str
            and len(b64) == 4 * ((n + 2) // 3),
            "decision_media_limit",
        )
        try:
            data = base64.b64decode(b64, validate=True)
        except ValueError:
            pass
        else:
            return cls(
                data, media_type=row["media_type"], sha256=row["sha256"], byte_size=n
            )
        raise DecisionMediaError("decision_media_invalid")


def checked_media(items: tuple[DecisionMedia, ...]) -> tuple[DecisionMedia, ...]:
    """Validate the total request budget before a backend is selected."""
    _need(
        type(items) is tuple
        and len(items) <= MAX_MEDIA_ITEMS
        and all(type(i) is DecisionMedia for i in items)
        and sum(i.byte_size for i in items) <= MAX_TOTAL_MEDIA_BYTES,
        "decision_media_limit",
    )
    return items


def parse_media(rows: Any) -> tuple[DecisionMedia, ...]:
    """Bound item counts and declared total before decoding any transport payload."""
    _need(
        type(rows) in {list, tuple}
        and len(rows) <= MAX_MEDIA_ITEMS
        and all(
            type(r) is dict
            and type(r.get("byte_size")) is int
            and 0 < r["byte_size"] <= MAX_MEDIA_BYTES
            for r in rows
        )
        and sum(r["byte_size"] for r in rows) <= MAX_TOTAL_MEDIA_BYTES,
        "decision_media_limit",
    )
    return checked_media(tuple(DecisionMedia.from_dict(r) for r in rows))


def system_one_media(body: Mapping[str, Any]) -> tuple[DecisionMedia, ...]:
    """Verify inline System One data URLs. Remote URLs and local paths are never opened."""
    pending: list[tuple[str, str]] = []
    size = 0
    for key in ("images", "videos"):
        rows = body.get(key, [])
        _need(
            type(rows) is list and len(pending) + len(rows) <= MAX_MEDIA_ITEMS,
            "decision_media_limit",
        )
        for item in rows:
            _need(
                type(item) is str
                and len(item) <= 64 + 4 * ((MAX_MEDIA_BYTES + 2) // 3),
                "decision_media_limit",
            )
            prefix, separator, encoded = item.partition(",")
            allowed = ("image/png", "image/jpeg") if key == "images" else ("video/mp4",)
            mime = next((t for t in allowed if prefix == "data:" + t + ";base64"), None)
            _need(
                mime is not None and separator == ",",
                "decision_media_local_source_required",
            )
            assert mime is not None
            _need(
                0 < len(encoded) <= 4 * ((MAX_MEDIA_BYTES + 2) // 3),
                "decision_media_limit",
            )
            size += (len(encoded) // 4) * 3 - min(
                2, len(encoded) - len(encoded.rstrip("="))
            )
            _need(size <= MAX_TOTAL_MEDIA_BYTES, "decision_media_limit")
            pending.append((mime, encoded))
    media = []
    for mime, encoded in pending:
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            pass
        else:
            media.append(
                DecisionMedia(
                    data,
                    media_type=mime,
                    sha256=hashlib.sha256(data).hexdigest(),
                    byte_size=len(data),
                )
            )
            continue
        raise DecisionMediaError("decision_media_invalid")
    return checked_media(tuple(media))
