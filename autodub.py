#!/usr/bin/env python3
"""Batch-dub English videos into Indonesian, fully local and open source.

Pipeline per video: ffmpeg extract audio -> faster-whisper (ASR + timestamps)
-> NLLB-200 (translation) -> Piper TTS (Indonesian voice) -> ffmpeg atempo
time-fit per segment -> reassemble audio track -> mux back onto original video.

Usage:
    .venv/bin/python autodub.py --source ./videos_in --output ./videos_out
"""

import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # progress prints visible immediately even when redirected to a log file

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}

PIPER_VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "id/id_ID/news_tts/medium/"
)
PIPER_VOICE_NAME = "id_ID-news_tts-medium"

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "ind_Latn"

SAMPLE_RATE = 22050  # Piper's id_ID-news_tts-medium output rate


def ensure_piper_voice(models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{PIPER_VOICE_NAME}.onnx"
    config_path = models_dir / f"{PIPER_VOICE_NAME}.onnx.json"
    for path in (onnx_path, config_path):
        if not path.exists():
            url = PIPER_VOICE_BASE_URL + path.name
            print(f"[setup] downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
    return onnx_path, config_path


def extract_audio(video_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(wav_path),
        ],
        check=True, capture_output=True,
    )


def transcribe(wav_path: Path, whisper_model, ) -> list[dict]:
    segments_iter, _info = whisper_model.transcribe(
        str(wav_path), language="en", vad_filter=True,
    )
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_iter if seg.text.strip()
    ]
    return segments


def translate_segments(segments: list[dict], tokenizer, model) -> list[dict]:
    import torch

    tokenizer.src_lang = NLLB_SRC_LANG
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    batch_size = 8
    texts = [seg["text"] for seg in segments]
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            generated = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=256,
            )
        translations.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )

    for seg, translated in zip(segments, translations):
        seg["text_id"] = translated.strip()
    return segments


def _piper_executable() -> str:
    # Resolve relative to the running interpreter so this works whether or
    # not the venv has been "activated" (PATH may not include .venv/bin).
    candidate = Path(sys.executable).parent / "piper"
    return str(candidate) if candidate.exists() else "piper"


def synthesize_segment(text: str, onnx_path: Path, config_path: Path, out_wav: Path) -> None:
    subprocess.run(
        [
            _piper_executable(),
            "--model", str(onnx_path),
            "--config", str(config_path),
            "--output_file", str(out_wav),
        ],
        input=text.encode("utf-8"),
        check=True, capture_output=True,
    )


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def fit_duration(in_wav: Path, target_seconds: float, out_wav: Path) -> None:
    actual = wav_duration_seconds(in_wav)
    if actual <= 0 or target_seconds <= 0:
        out_wav.write_bytes(in_wav.read_bytes())
        return

    # atempo = actual / target: >1 speeds audio up (compress into a shorter
    # slot), <1 slows it down. Clamp to ffmpeg's per-filter [0.5, 2.0] range,
    # chaining filters for ratios outside that band.
    ratio = actual / target_seconds
    ratio = max(0.5, min(ratio, 4.0))

    factors = []
    remaining = ratio
    for _ in range(3):
        if 0.5 <= remaining <= 2.0:
            factors.append(remaining)
            remaining = 1.0
            break
        step = 2.0 if remaining > 1.0 else 0.5
        factors.append(step)
        remaining /= step
    if remaining != 1.0:
        factors.append(remaining)

    filter_str = ",".join(f"atempo={f:.4f}" for f in factors)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_wav), "-filter:a", filter_str, str(out_wav)],
        check=True, capture_output=True,
    )


def assemble_track(segments: list[dict], total_duration: float, temp_dir: Path) -> Path:
    from pydub import AudioSegment

    track = AudioSegment.silent(duration=int(total_duration * 1000) + 1000, frame_rate=SAMPLE_RATE)
    for idx, seg in enumerate(segments):
        fitted_path = temp_dir / f"seg_{idx:04d}_fitted.wav"
        if not fitted_path.exists():
            continue
        audio = AudioSegment.from_wav(fitted_path)
        track = track.overlay(audio, position=int(seg["start"] * 1000))

    out_path = temp_dir / "dubbed_track.wav"
    track.export(out_path, format="wav")
    return out_path


def mux_video(video_path: Path, dubbed_audio: Path, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(dubbed_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out_path),
        ],
        check=True, capture_output=True,
    )


def process_video(video_path: Path, out_path: Path, models: dict, keep_temp: bool) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="autodub_"))
    try:
        wav_path = temp_dir / "audio.wav"
        print(f"  [1/6] extracting audio ...")
        extract_audio(video_path, wav_path)

        print(f"  [2/6] transcribing (English) ...")
        segments = transcribe(wav_path, models["whisper"])
        print(f"        {len(segments)} segments detected")

        print(f"  [3/6] translating to Indonesian ...")
        segments = translate_segments(segments, models["nllb_tokenizer"], models["nllb_model"])

        print(f"  [4/6] synthesizing + time-fitting Indonesian speech ...")
        for idx, seg in enumerate(segments):
            raw_wav = temp_dir / f"seg_{idx:04d}_raw.wav"
            fitted_wav = temp_dir / f"seg_{idx:04d}_fitted.wav"
            synthesize_segment(seg["text_id"], models["piper_onnx"], models["piper_config"], raw_wav)
            fit_duration(raw_wav, seg["end"] - seg["start"], fitted_wav)

        total_duration = wav_duration_seconds(wav_path)
        print(f"  [5/6] assembling dubbed audio track ...")
        dubbed_audio = assemble_track(segments, total_duration, temp_dir)

        print(f"  [6/6] muxing into final video ...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mux_video(video_path, dubbed_audio, out_path)
    finally:
        if not keep_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"  (temp files kept at {temp_dir})")


def load_models(whisper_model_size: str, models_dir: Path) -> dict:
    print(f"[setup] loading faster-whisper ({whisper_model_size}, CPU, int8) ...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

    print(f"[setup] loading translation model ({NLLB_MODEL_NAME}) ...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

    print(f"[setup] preparing Piper voice ({PIPER_VOICE_NAME}) ...")
    piper_onnx, piper_config = ensure_piper_voice(models_dir)

    return {
        "whisper": whisper_model,
        "nllb_tokenizer": nllb_tokenizer,
        "nllb_model": nllb_model,
        "piper_onnx": piper_onnx,
        "piper_config": piper_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder berisi video sumber (Bahasa Inggris)")
    parser.add_argument("--output", required=True, help="Folder tujuan untuk video hasil dubbing")
    parser.add_argument("--whisper-model", default="medium", help="Ukuran model faster-whisper (default: medium)")
    parser.add_argument("--suffix", default="_ID", help="Sufiks nama file output (default: _ID)")
    parser.add_argument("--keep-temp", action="store_true", help="Simpan file sementara untuk debugging")
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    models_dir = Path(__file__).parent / "models" / "piper"

    if not source_dir.is_dir():
        sys.exit(f"Folder source tidak ditemukan: {source_dir}")

    def video_sort_key(path: Path) -> tuple:
        # Sort by the leading number in the title (e.g. "2. Foo" before
        # "10. Bar"), not plain alphabetical order.
        match = re.match(r"\s*(\d+)", path.stem)
        number = int(match.group(1)) if match else float("inf")
        return (number, path.name)

    videos = sorted(
        (p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
        key=video_sort_key,
    )
    if not videos:
        sys.exit(f"Tidak ada file video ({', '.join(sorted(VIDEO_EXTS))}) di {source_dir}")

    print(f"Ditemukan {len(videos)} video di {source_dir}")
    models = load_models(args.whisper_model, models_dir)

    ok, failed, skipped = 0, 0, 0
    for i, video_path in enumerate(videos, 1):
        rel = video_path.relative_to(source_dir)
        out_path = (output_dir / rel).with_name(rel.stem + args.suffix + ".mp4")

        print(f"\n[{i}/{len(videos)}] {rel}")
        if out_path.exists():
            print(f"  sudah ada, dilewati -> {out_path}")
            skipped += 1
            continue

        try:
            process_video(video_path, out_path, models, args.keep_temp)
            print(f"  selesai -> {out_path}")
            ok += 1
        except Exception as exc:
            print(f"  GAGAL: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nSelesai. Berhasil: {ok}, dilewati: {skipped}, gagal: {failed}")


if __name__ == "__main__":
    main()
