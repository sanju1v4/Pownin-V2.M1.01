"""WER of raw vs DPDFNet-enhanced audio, per clip, using jiwer.

    python bench.py

Reference file: one line per clip, either

    12.wav<TAB>the reference sentence
    the reference sentence          (paired with wavs in sorted order)
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import soundfile as sf
import tempfile
from openai import OpenAI

from enhance import SR, enhance_array, load_16k_mono
from server import TRANSCRIBE_MODEL, load_env

WAV_DIR = Path(r"E:\Pawninmodule\bench_data\clips")
REFERENCE = Path(r"E:\Pawninmodule\bench_data\reference.txt")
WORKERS = 6

# Casing, punctuation and digit-vs-word spelling are transcription style, not
# noise robustness. Both paths get the identical treatment.
NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

NUMBERS = {
    "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
    "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
    "zehn": "10", "zwanzig": "20", "dreißig": "30", "fünfzig": "50",
    "hundert": "100", "fünfhundert": "500", "achthundert": "800",
}


def spell_numbers(text: str) -> str:
    """Map German number words to digits so '800' and 'achthundert' match."""
    return re.sub(
        r"\b\w+\b",
        lambda m: NUMBERS.get(m.group(0).lower(), m.group(0)),
        text,
    )


def load_reference() -> dict[str, str]:
    lines = [l for l in REFERENCE.read_text(encoding="utf-8").splitlines() if l.strip()]
    wavs = sorted(p.name for p in WAV_DIR.glob("*.wav"))
    refs: dict[str, str] = {}
    for i, line in enumerate(lines):
        if "\t" in line:
            name, _, text = line.partition("\t")
            refs[name.strip()] = text.strip()
        elif i < len(wavs):
            refs[wavs[i]] = line.strip()
    return refs


def transcribe(audio, client: OpenAI) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, SR, subtype="PCM_16")
        with open(path, "rb") as fh:
            r = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=fh, response_format="text"
            )
    finally:
        os.unlink(path)
    return (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()


def wer(ref: str, hyp: str) -> float:
    ref, hyp = spell_numbers(ref), spell_numbers(hyp)
    if not hyp.strip():
        return 1.0
    return jiwer.wer(ref, hyp, reference_transform=NORMALISE, hypothesis_transform=NORMALISE)


def run_clip(item, client):
    name, ref = item
    raw = load_16k_mono(str(WAV_DIR / name))
    enh = enhance_array(raw)
    return name, ref, transcribe(raw, client), transcribe(enh, client)


def main() -> None:
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (env or .env)")

    refs = load_reference()
    items = [(n, r) for n, r in refs.items() if (WAV_DIR / n).exists()]
    if not items:
        raise SystemExit(f"no wavs matching {REFERENCE} found in {WAV_DIR}")

    client = OpenAI()
    print(f"{len(items)} clips from {WAV_DIR}\n")
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(lambda it: run_clip(it, client), items))

    print(f"{'clip':<12}{'words':>6}{'WER raw':>10}{'WER enh':>10}{'delta':>9}")
    print("-" * 47)
    raws, enhs = [], []
    for name, ref, raw_text, enh_text in rows:
        wr, we = wer(ref, raw_text), wer(ref, enh_text)
        raws.append(wr)
        enhs.append(we)
        d = we - wr
        print(f"{name:<12}{len(ref.split()):>6}{wr:>9.1%}{we:>10.1%}{d:>+9.1%}")

    mr, me = sum(raws) / len(raws), sum(enhs) / len(enhs)
    print("-" * 47)
    print(f"{'mean':<12}{'':>6}{mr:>9.1%}{me:>10.1%}{me - mr:>+9.1%}")
    if mr:
        print(f"\nrelative WER reduction: {(mr - me) / mr:+.1%}")
    better = sum(1 for a, b in zip(raws, enhs) if b < a - 1e-9)
    worse = sum(1 for a, b in zip(raws, enhs) if b > a + 1e-9)
    print(f"clips improved: {better}   unchanged: {len(rows)-better-worse}   worse: {worse}")


if __name__ == "__main__":
    main()
