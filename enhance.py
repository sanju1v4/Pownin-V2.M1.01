"""DPDFNet speech enhancement, 16 kHz mono, multi-minute safe.

    from enhance import enhance
    y = enhance("meeting.m4a")        # -> float32 np.ndarray @ 16 kHz

Long-audio note: dpdfnet's frame loop threads the RNN state across every frame
of the input, so a 3-minute file is already processed as one continuous stream.
We deliberately do NOT chunk -- chunking is what would create boundary
artefacts, not what avoids them. Memory is linear in duration and small
(~12 MB of spectra per minute), so there is nothing to buy by splitting.

What we do fix: dpdfnet.enhance() rebuilds the ONNX session on every call.
We build it once and reuse it.
"""

import os
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import soxr

import dpdfnet
from dpdfnet.models import resolve_model
from dpdfnet.onnx_backend import build_runtime_model

SR = 16000
MODEL = "dpdfnet4"
ATTN_LIMIT_DB = 12

_runtime = None
_model_sr = None


def _get_runtime():
    """Build the ONNX session once; ~5 s the first time, free thereafter."""
    global _runtime, _model_sr
    if _runtime is None:
        resolved = resolve_model(model=MODEL, onnx_path=None, auto_download=True)
        _runtime = build_runtime_model(resolved.onnx_path)
        _model_sr = resolved.info.sample_rate
    return _runtime, _model_sr


def warmup() -> None:
    """Preload the model. Call at server startup or the first request pays ~8 s."""
    enhance_array(np.zeros(SR, dtype=np.float32))


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

# ffmpeg exits with its negative AVERROR code, which the OS then reports as an
# unsigned 32-bit integer -- hence the eye-watering "exit status 3199971767".
# Most AVERRORs are a negated four-character tag; the rest are negated errno.
_AVERROR_TAGS = {
    "INDA": "invalid data found when processing input (not a valid/parseable container)",
    "DEMU": "demuxer not found",
    "DECO": "decoder not found",
    "EOF ": "end of file reached prematurely (file is truncated)",
    "PAWN": "unknown protocol",
    "OPTN": "option not found",
}


def _explain_exit(returncode: int) -> str:
    """Turn an ffmpeg exit status back into something a human can act on."""
    raw = returncode if returncode < 0 else returncode - 2**32
    if raw >= 0:
        return f"exit status {returncode}"
    mk = -raw
    chars = [(mk >> s) & 0xFF for s in (0, 8, 16, 24)]
    if all(32 <= c < 127 for c in chars):
        tag = "".join(chr(c) for c in chars)
        meaning = _AVERROR_TAGS.get(tag, "unrecognised AVERROR tag")
        return f"exit status {returncode} = AVERROR '{tag}' -- {meaning}"
    if mk < 256:
        import errno
        return f"exit status {returncode} = errno {mk} ({errno.errorcode.get(mk, '?')})"
    return f"exit status {returncode} (unrecognised)"


def describe_media(path: str) -> str:
    """Size + container sniff + leading bytes. Safe to call on garbage."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"{path}: cannot stat ({exc})"

    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError as exc:
        return f"{path}: {size} bytes, cannot read ({exc})"

    bits = [f"{size} bytes"]
    if size == 0:
        bits.append("EMPTY")
    elif len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", "replace")
        first_box = int.from_bytes(head[:4], "big")
        bits.append(f"ISO-BMFF/MP4, ftyp brand '{brand}', first box {first_box} B")
    elif head[:4] == b"RIFF":
        bits.append("RIFF/WAV")
    elif head[:4] == b"OggS":
        bits.append("Ogg")
    elif head[:4] == b"\x1a\x45\xdf\xa3":
        # what Chrome/Firefox MediaRecorder produces
        bits.append("EBML/Matroska (WebM)")
    elif head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf1"):
        bits.append("MPEG audio / ADTS")
    elif head[:4] == b"\x00\x00\x00\x00":
        bits.append("leading NUL bytes -- looks zero-filled, not a container")
    else:
        bits.append("UNRECOGNISED container signature")

    bits.append("head=" + " ".join(f"{b:02x}" for b in head[:16]))
    bits.append("ascii=" + "".join(chr(b) if 32 <= b < 127 else "." for b in head[:16]))
    return f"{path}: " + ", ".join(bits)


def _ffmpeg_decode(path: str) -> np.ndarray:
    """Decode anything ffmpeg understands (m4a/aac/3gp/opus/...) to 16 kHz mono."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"cannot decode {path}: ffmpeg not on PATH")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        # Never swallow this. stderr is the only thing that says *why*.
        stderr = proc.stderr.decode("utf-8", "replace").strip() or "(ffmpeg wrote nothing to stderr)"
        raise RuntimeError(
            f"ffmpeg failed to decode the upload\n"
            f"  {_explain_exit(proc.returncode)}\n"
            f"  file: {describe_media(path)}\n"
            f"  ffmpeg stderr:\n"
            + "\n".join(f"    {line}" for line in stderr.splitlines())
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_16k_mono(path: str) -> np.ndarray:
    """Read any audio file -> float32 mono @ 16 kHz. soundfile first, ffmpeg fallback."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as sf_exc:
        try:
            return _ffmpeg_decode(path)
        except RuntimeError as ff_exc:
            # Both readers failed; the soundfile error is a useful second opinion.
            raise RuntimeError(f"{ff_exc}\n  soundfile also failed: {sf_exc}") from None
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------
# enhancement
# --------------------------------------------------------------------------

def enhance_array(audio: np.ndarray, on_progress=None) -> np.ndarray:
    """Denoise an in-memory float32 mono 16 kHz signal of any length.

    on_progress: optional callable(fraction_0_to_1), throttled to ~2 s. Called
    from whatever thread runs this, so keep it thread-safe.
    """
    runtime, model_sr = _get_runtime()

    cb = None
    if on_progress is not None:
        last = [0.0]

        def cb(done: int, total: int) -> None:
            now = time.perf_counter()
            if total and (done == total or now - last[0] > 2.0):
                last[0] = now
                on_progress(done / total)

    try:
        out = dpdfnet.api._enhance_with_runtime(
            audio,
            sample_rate=SR,
            runtime=runtime,
            model_sample_rate=model_sr,
            attn_limit_db=ATTN_LIMIT_DB,
            progress_callback=cb,
        )
    except AttributeError:
        # private helper gone in a future dpdfnet; public path rebuilds the
        # session per call but is otherwise identical
        out = dpdfnet.enhance(
            audio, sample_rate=SR, model=MODEL,
            attn_limit_db=ATTN_LIMIT_DB, progress_callback=cb,
        )
    return np.ascontiguousarray(out, dtype=np.float32)


def enhance(wav_path: str) -> np.ndarray:
    """Denoise an audio file. Returns float32 mono @ 16 kHz."""
    return enhance_array(load_16k_mono(wav_path))


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "enhanced.wav"

    raw = load_16k_mono(src)
    _get_runtime()
    t0 = time.perf_counter()
    out = enhance_array(raw, on_progress=lambda f: print(f"    enhance {100*f:5.1f}%", flush=True))
    ms = (time.perf_counter() - t0) * 1000

    sf.write(dst, out, SR)
    dur = len(raw) / SR
    print(f"in  {dur:6.2f}s  rms={np.sqrt((raw**2).mean()):.5f}  peak={np.abs(raw).max():.5f}")
    print(f"out {len(out)/SR:6.2f}s  rms={np.sqrt((out**2).mean()):.5f}  peak={np.abs(out).max():.5f}")
    print(f"enhance {ms/1000:.1f}s  (RTF {ms/1000/dur:.2f})  -> {dst}")
