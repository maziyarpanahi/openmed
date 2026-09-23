"""Bounded Tesseract OCR over stdin/stdout without temporary image/text files."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence

from .exceptions import MissingDependencyError
from .ocr import OcrResult, OcrWord, tesseract_language

_TSV_HEADER = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def _execute(
    command: list[str],
    payload: bytes,
    *,
    timeout: float,
    max_output_bytes: int,
    cancel_check: Callable[[], bool] | None,
) -> bytes:
    """Drain output while feeding input; reap the owned process on every exit."""
    try:
        cancelled = cancel_check is not None and cancel_check()
    except Exception:
        cancelled = True
    if cancelled:
        raise RuntimeError("ocr_cancelled")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "SystemRoot",
            "WINDIR",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "TESSDATA_PREFIX",
        }
    }
    environment.update(OMP_THREAD_LIMIT="1", LC_ALL="C")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
    except OSError:
        raise RuntimeError("ocr_process_unavailable") from None
    assert process.stdin is not None and process.stdout is not None
    stopped = threading.Event()
    reason: list[str] = []
    reason_lock = threading.Lock()
    deadline = time.monotonic() + timeout

    def stop(code: str) -> None:
        with reason_lock:
            if not reason:
                reason.append(code)
        if process.poll() is None:
            process.kill()

    def feed() -> None:
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def watch() -> None:
        while not stopped.wait(0.02):
            try:
                if cancel_check is not None and cancel_check():
                    stop("ocr_cancelled")
                    return
                if time.monotonic() >= deadline:
                    stop("ocr_timeout")
                    return
            except Exception:
                stop("ocr_cancelled")
                return

    writer = threading.Thread(target=feed, name="openmed-ocr-input")
    watchdog = threading.Thread(target=watch, name="openmed-ocr-watchdog")
    output = bytearray()
    try:
        try:
            writer.start()
            watchdog.start()
        except (RuntimeError, OSError):
            raise RuntimeError("ocr_process_unavailable") from None
        while chunk := process.stdout.read1(65_536):
            if len(output) + len(chunk) > max_output_bytes:
                stop("ocr_output_limit")
                break
            output.extend(chunk)
        process.wait()
        stopped.set()
        if reason:
            raise RuntimeError(reason[0])
        if process.returncode != 0:
            raise RuntimeError("ocr_execution_failed")
        return bytes(output)
    finally:
        stopped.set()
        if process.poll() is None:
            process.kill()
        process.wait()
        if writer.ident is not None:
            writer.join()
        else:
            process.stdin.close()
        if watchdog.ident is not None:
            watchdog.join()
        process.stdout.close()


def _png(image: Any, *, max_pixels: int) -> tuple[bytes, int, int]:
    try:
        from PIL import Image
    except ImportError:
        raise MissingDependencyError(
            dependency="Pillow", instruction="Install Pillow to encode OCR input."
        ) from None
    owned = not isinstance(image, Image.Image)
    loaded = None
    try:
        loaded = (
            Image.open(BytesIO(image) if isinstance(image, bytes) else image)
            if owned
            else image
        )
        width, height = loaded.size
        if getattr(loaded, "n_frames", 1) != 1:
            raise ValueError("ocr_multiple_frames")
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError("ocr_pixel_limit")
        with (
            loaded.convert("RGBA") as rgba,
            Image.new("RGB", loaded.size, "white") as pixels,
        ):
            with rgba.getchannel("A") as alpha:
                pixels.paste(rgba, mask=alpha)
            output = BytesIO()
            pixels.save(output, format="PNG")
            return output.getvalue(), width, height
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc) in {
            "ocr_multiple_frames",
            "ocr_pixel_limit",
        }:
            raise
        raise ValueError("ocr_invalid_image") from None
    finally:
        if owned and loaded is not None:
            loaded.close()


def _parse_tsv(
    payload: bytes,
    *,
    width: int,
    height: int,
    max_words: int,
    max_text_chars: int,
) -> OcrResult:
    try:
        lines = payload.decode("utf-8").splitlines()
        if not lines or tuple(lines[0].split("\t")) != _TSV_HEADER:
            raise ValueError
        words = []
        line_ids = []
        word_ids = set()
        text_chars = 0
        for line in lines[1:]:
            if not line:
                continue
            cells = line.split("\t", 11)
            if len(cells) != 12:
                raise ValueError
            level, page, block, paragraph, line_number, word_number = map(
                int, cells[:6]
            )
            if (
                not 1 <= level <= 5
                or page != 1
                or min(block, paragraph, line_number, word_number) < 0
            ):
                raise ValueError
            if level != 5:
                continue
            left, top, box_width, box_height = map(int, cells[6:10])
            confidence = float(cells[10])
            raw_text = cells[11]
            text = raw_text.strip()
            if (
                min(left, top) < 0
                or min(box_width, box_height) <= 0
                or left + box_width > width
                or top + box_height > height
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 100
                or min(block, paragraph, line_number, word_number) <= 0
                or any(ord(char) < 32 or ord(char) == 127 for char in raw_text)
            ):
                raise ValueError
            if not text:
                continue
            word_id = (block, paragraph, line_number, word_number)
            if word_id in word_ids:
                raise ValueError
            word_ids.add(word_id)
            text_chars += len(text) + bool(words)
            if len(words) >= max_words or text_chars > max_text_chars:
                raise RuntimeError("ocr_text_limit")
            words.append(
                OcrWord(
                    text,
                    (left, top, left + box_width, top + box_height),
                    confidence / 100,
                )
            )
            line_ids.append((block, paragraph, line_number))
        return OcrResult(
            tuple(words),
            metadata={
                "engine": "tesseract-stream",
                "transport": "stdin-stdout",
                "width": width,
                "height": height,
                "line_ids": tuple(line_ids),
                "qualification": "preview",
            },
        )
    except (UnicodeDecodeError, ValueError, IndexError):
        raise ValueError("ocr_invalid_output") from None


class StreamingTesseractEngine:
    """Explicit OCR engine for processing one image entirely through memory.

    Requires Pillow and a locally installed Tesseract executable/language data.
    It never selects or downloads a model. OCR results remain unqualified.
    """

    name = "tesseract-stream"

    def __init__(
        self,
        *,
        executable: str = "tesseract",
        tessdata_dir: str | Path | None = None,
        timeout_seconds: float = 30,
        max_pixels: int = 16_000_000,
        max_output_bytes: int = 8_000_000,
        max_words: int = 25_000,
        max_text_chars: int = 100_000,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Configure trusted runtime limits, independent of document contents.

        Args:
            executable: Tesseract executable name or path, selected by the host.
            tessdata_dir: Optional host-selected directory of installed language data.
            timeout_seconds: Native execution deadline, between zero and 120 seconds.
            max_pixels: Maximum pixels accepted before image decoding/conversion.
            max_output_bytes: Maximum native TSV output retained in memory.
            max_words: Maximum recognized words accepted in one image.
            max_text_chars: Maximum characters in the space-joined OCR result.
            cancel_check: Optional thread-safe cancellation callback.
        """
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 120
        ):
            raise ValueError("invalid OCR timeout")
        for limit in (max_pixels, max_output_bytes, max_words, max_text_chars):
            if type(limit) is not int or limit < 1:
                raise ValueError("invalid OCR limit")
        if cancel_check is not None and not callable(cancel_check):
            raise ValueError("invalid OCR cancellation callback")
        self.executable, self.tessdata_dir = executable, tessdata_dir
        self.timeout_seconds, self.max_pixels = timeout_seconds, max_pixels
        self.max_output_bytes, self.max_words = max_output_bytes, max_words
        self.max_text_chars, self.cancel_check = max_text_chars, cancel_check

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        """Recognize one image with validated word geometry and confidence.

        Args:
            image: A Pillow image, encoded image bytes, image path or binary stream.
            languages: OpenMed OCR language codes; defaults to English.

        Returns:
            Recognized words and numeric layout metadata, without temporary files.

        Raises:
            MissingDependencyError: Pillow or the Tesseract executable is unavailable.
            ValueError: Invalid input, geometry, frame count, limits or TSV output.
            RuntimeError: Cancellation, deadline, native failure or output limits.
        """
        try:
            language = tesseract_language(languages)
        except ValueError:
            raise ValueError("ocr_unsupported_language") from None
        executable = shutil.which(self.executable)
        if executable is None:
            raise MissingDependencyError(
                dependency="tesseract",
                instruction="Install Tesseract and the requested language data on the host.",
            )
        payload, width, height = _png(image, max_pixels=self.max_pixels)
        command = [executable, "stdin", "stdout", "-l", language]
        if self.tessdata_dir is not None:
            command += ["--tessdata-dir", str(self.tessdata_dir)]
        command += ["-c", "tessedit_write_images=0", "tsv"]
        output = _execute(
            command,
            payload,
            timeout=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            cancel_check=self.cancel_check,
        )
        return _parse_tsv(
            output,
            width=width,
            height=height,
            max_words=self.max_words,
            max_text_chars=self.max_text_chars,
        )
