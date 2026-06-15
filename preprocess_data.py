#!/usr/bin/env python3
"""Preprocess audio and flow-curve files for CFBMNet training."""

import argparse
import json
import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config import Config

try:
    from scipy.interpolate import (
        Akima1DInterpolator,
        CubicSpline,
        PchipInterpolator,
        interp1d,
        make_interp_spline,
    )

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def unify_audio_length(audio: np.ndarray, sample_rate: int, target_duration: float = 3.0) -> np.ndarray:
    """Pad or trim an audio waveform to the target duration."""
    target_length = int(sample_rate * target_duration)
    current_length = len(audio)

    if current_length == target_length:
        return audio
    if current_length < target_length:
        pad_length = target_length - current_length
        last_value = audio[-1] if current_length > 0 else 0.0
        fade_out = np.linspace(last_value, 0.0, pad_length)
        return np.concatenate([audio, fade_out])
    return audio[:target_length]


def process_audio_file(
    input_path: Path,
    output_path: Path,
    target_duration: float = 3.0,
    target_sample_rate: int = 48000,
) -> bool:
    """Load, resample, normalize duration, and save one audio file."""
    try:
        audio, sample_rate = librosa.load(input_path, sr=None)
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        audio = unify_audio_length(audio, sample_rate, target_duration)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
        return True
    except Exception as exc:
        print(f"Failed to process audio file {input_path}: {exc}")
        return False


def prepare_audio_directory(
    input_dir: Path,
    output_dir: Path,
    target_duration: float = 3.0,
    target_sample_rate: int = 48000,
) -> dict:
    """Normalize every WAV file in a directory to a fixed duration and sample rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(input_dir.glob("*.wav"))
    stats = {
        "total_files": len(audio_files),
        "processed_files": 0,
        "failed_files": 0,
    }

    for audio_file in tqdm(audio_files, desc="Preparing audio"):
        output_path = output_dir / audio_file.name
        if process_audio_file(audio_file, output_path, target_duration, target_sample_rate):
            stats["processed_files"] += 1
        else:
            stats["failed_files"] += 1

    return stats


def calculate_time_column(volumes: np.ndarray, flows: np.ndarray) -> np.ndarray:
    """Estimate the time axis from volume and flow using trapezoidal increments."""
    if len(volumes) == 0:
        return np.asarray([], dtype=np.float64)

    times = [0.0 if flows[0] == 0 else float(volumes[0] / flows[0])]
    for idx in range(1, len(volumes)):
        delta_volume = float(volumes[idx] - volumes[idx - 1])
        average_flow = float((flows[idx] + flows[idx - 1]) / 2.0)
        delta_time = 0.0 if average_flow == 0.0 else delta_volume / average_flow
        times.append(times[-1] + delta_time)
    return np.asarray(times, dtype=np.float64)


def load_flow_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a raw flow CSV and return volume, flow, and optional time arrays."""
    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {csv_path}")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="any")
    numeric = numeric[(numeric.iloc[:, 0] >= 0) & (numeric.iloc[:, 1] >= 0)]
    if numeric.empty:
        raise ValueError(f"No valid non-negative rows in {csv_path}")

    numeric = numeric.sort_values(numeric.columns[0]).drop_duplicates(subset=numeric.columns[0], keep="first")
    volumes = numeric.iloc[:, 0].to_numpy(dtype=np.float64)
    flows = numeric.iloc[:, 1].to_numpy(dtype=np.float64)
    times = numeric.iloc[:, 2].to_numpy(dtype=np.float64) if numeric.shape[1] >= 3 else None
    return volumes, flows, times


def interpolate_flow_data(
    time_data: np.ndarray,
    flow_data: np.ndarray,
    target_time: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a flow curve onto the target time grid."""
    if len(time_data) < 2:
        return np.zeros_like(target_time)

    unique_indices = np.unique(time_data, return_index=True)[1]
    time_data = time_data[unique_indices]
    flow_data = flow_data[unique_indices]
    if len(time_data) < 2:
        return np.zeros_like(target_time)

    if method == "linear" or not SCIPY_AVAILABLE:
        return np.interp(target_time, time_data, flow_data)
    if method == "pchip" and len(time_data) >= 3:
        return PchipInterpolator(time_data, flow_data)(target_time)
    if method == "akima" and len(time_data) >= 3:
        return Akima1DInterpolator(time_data, flow_data)(target_time)
    if method == "cubic_spline" and len(time_data) >= 4:
        return CubicSpline(time_data, flow_data, bc_type="natural")(target_time)
    if method == "cubic_spline_clamped" and len(time_data) >= 4:
        return CubicSpline(time_data, flow_data, bc_type=((1, 0.0), (1, 0.0)))(target_time)
    if method == "bspline" and len(time_data) >= 4:
        return make_interp_spline(time_data, flow_data, k=3)(target_time)

    interpolator = interp1d(time_data, flow_data, bounds_error=False, fill_value=(flow_data[0], flow_data[-1]))
    return interpolator(target_time)


def process_flow_curve(
    flow_data: np.ndarray,
    time_data: np.ndarray,
    interpolation_method: str = "cubic_spline_clamped",
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one raw flow-time sequence to the fixed 0-3 s training grid."""
    target_time = np.linspace(0.0, 3.0, sequence_length, dtype=np.float64)
    valid_mask = np.isfinite(time_data) & np.isfinite(flow_data) & (time_data >= 0.0) & (time_data <= 3.0)
    time_data = time_data[valid_mask]
    flow_data = flow_data[valid_mask]

    if len(flow_data) == 0:
        return np.zeros(sequence_length, dtype=np.float64), target_time

    if time_data[0] > 0.0:
        time_data = np.concatenate([[0.0], time_data])
        flow_data = np.concatenate([[0.0], flow_data])
    else:
        flow_data[0] = 0.0

    if time_data[-1] < 3.0:
        decay_points = max(5, int((3.0 - time_data[-1]) * 10))
        decay_time = np.linspace(time_data[-1], 3.0, decay_points)[1:]
        decay_flow = np.linspace(flow_data[-1], 0.0, decay_points)[1:]
        time_data = np.concatenate([time_data, decay_time])
        flow_data = np.concatenate([flow_data, decay_flow])

    sorted_indices = np.argsort(time_data)
    time_data = time_data[sorted_indices]
    flow_data = flow_data[sorted_indices]

    processed_flow = interpolate_flow_data(time_data, flow_data, target_time, method=interpolation_method)
    processed_flow = np.nan_to_num(processed_flow, nan=0.0, posinf=0.0, neginf=0.0)
    processed_flow[0] = 0.0
    processed_flow[-1] = 0.0
    return processed_flow.astype(np.float64), target_time


def generate_mel_spectrogram(audio: np.ndarray, target_frames: int) -> np.ndarray:
    """Generate a Mel spectrogram aligned to the fixed flow-curve grid."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmax=Config.SAMPLE_RATE // 2,
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return librosa.util.fix_length(mel_spec, size=target_frames, axis=1)


def copy_label_file(input_csv_dir: Path, output_dir: Path) -> bool:
    """Copy or convert the label file into the processed data directory."""
    candidates = [
        input_csv_dir.parent / "label.csv",
        input_csv_dir.parent / "label.xlsx",
        Path("./data/label.csv"),
        Path("./data/label.xlsx"),
    ]

    for candidate in dict.fromkeys(candidates):
        if not candidate.exists():
            continue
        output_path = output_dir / "label.csv"
        if candidate.suffix.lower() == ".csv":
            shutil.copyfile(candidate, output_path)
        else:
            label_frame = pd.read_excel(candidate)
            if "id" in label_frame.columns:
                label_frame["id"] = label_frame["id"].apply(lambda value: f"{int(value):04d}" if pd.notna(value) else value)
            label_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved label file to {output_path}")
        return True

    print("Warning: no label.csv or label.xlsx file was found.")
    return False


def process_dataset(
    input_wav_dir: str | Path,
    input_csv_dir: str | Path,
    output_dir: str | Path,
    interpolation_method: str = "cubic_spline_clamped",
    sequence_length: int | None = None,
) -> dict:
    """Create processed Mel spectrograms, flow curves, metadata, and labels."""
    input_wav_dir = Path(input_wav_dir)
    input_csv_dir = Path(input_csv_dir)
    output_dir = Path(output_dir)
    sequence_length = int(sequence_length or Config.SEQUENCE_LENGTH)

    mel_dir = output_dir / "mel"
    csv_dir = output_dir / "csv"
    meta_dir = output_dir / "meta"
    mel_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_wav_dir.glob("*.wav"))
    stats = {
        "total_files": len(wav_files),
        "processed_files": 0,
        "failed_files": 0,
        "interpolation_method": interpolation_method,
        "sequence_length": sequence_length,
    }

    for wav_path in tqdm(wav_files, desc="Building processed dataset"):
        sample_id = wav_path.stem
        csv_path = input_csv_dir / f"{sample_id}.csv"
        if not csv_path.exists():
            subject_csv_path = input_csv_dir / f"{sample_id.split('_')[0]}.csv"
            csv_path = subject_csv_path if subject_csv_path.exists() else csv_path
        if not csv_path.exists():
            print(f"Warning: missing flow CSV for {sample_id}")
            stats["failed_files"] += 1
            continue

        try:
            audio, _ = librosa.load(wav_path, sr=Config.SAMPLE_RATE)
            audio = unify_audio_length(audio, Config.SAMPLE_RATE, target_duration=3.0)

            volumes, raw_flow, raw_time = load_flow_csv(csv_path)
            if raw_time is None:
                raw_time = calculate_time_column(volumes, raw_flow)

            processed_flow, processed_time = process_flow_curve(
                raw_flow,
                raw_time,
                interpolation_method=interpolation_method,
                sequence_length=sequence_length,
            )
            mel_spec = generate_mel_spectrogram(audio, target_frames=sequence_length)

            np.save(mel_dir / f"{sample_id}.npy", mel_spec)
            pd.DataFrame({"time": processed_time, "flow": processed_flow}).to_csv(
                csv_dir / f"{sample_id}.csv",
                index=False,
                header=False,
            )

            metadata = {
                "sample_id": sample_id,
                "source_wav": str(wav_path),
                "source_csv": str(csv_path),
                "raw_total_time": float(raw_time[-1]) if len(raw_time) else 0.0,
                "raw_terminal_volume": float(volumes[-1]) if len(volumes) else 0.0,
            }
            with open(meta_dir / f"{sample_id}.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            stats["processed_files"] += 1
        except Exception as exc:
            print(f"Failed to process sample {sample_id}: {exc}")
            stats["failed_files"] += 1

    stats["label_csv_saved"] = copy_label_file(input_csv_dir, output_dir)
    with open(output_dir / "processing_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Processed {stats['processed_files']} of {stats['total_files']} files.")
    print(f"Processed data saved to {output_dir}")
    return stats


def main() -> None:
    """Run the command-line preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess smartphone audio and flow curves for CFBMNet.")
    parser.add_argument("--wav_dir", type=str, default="./data_temporary/wav", help="Input WAV directory.")
    parser.add_argument("--csv_dir", type=str, default="./data_temporary/csv", help="Input raw flow CSV directory.")
    parser.add_argument("--output_dir", type=str, default="./data_pre", help="Output processed data directory.")
    parser.add_argument(
        "--method",
        type=str,
        default="cubic_spline_clamped",
        choices=["linear", "pchip", "akima", "cubic_spline", "cubic_spline_clamped", "bspline"],
        help="Flow-curve interpolation method.",
    )
    parser.add_argument("--sequence_length", type=int, default=None, help="Number of points in each processed curve.")
    args = parser.parse_args()

    process_dataset(
        input_wav_dir=args.wav_dir,
        input_csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        interpolation_method=args.method,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
