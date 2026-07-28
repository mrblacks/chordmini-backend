"""
Wrapper for BTC chord recognition, called by the Flask backend detectors.

Usage:
    from btc_chord_recognition import btc_chord_recognition
    ok = btc_chord_recognition(audio_path, output_lab_path, model_variant='sl')
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR / "src"

_CHECKPOINTS = {
    "sl": str(_THIS_DIR / "checkpoints" / "SL" / "btc_model_large_voca.pt"),
    "pl": str(_THIS_DIR / "checkpoints" / "btc" / "btc_combined_best.pth"),
}

_CONFIG_PATH = str(_THIS_DIR / "config" / "ChordMini.yaml")


def _setup_paths():
    """Ensure ChordMini directories are at the front of sys.path."""
    for p in [str(_SRC_DIR), str(_THIS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _enter_chordmini_context():
    """
    Temporarily shadow the Flask backend's 'utils' package with ChordMini's src/utils.
    Returns a dict of modules that were displaced; pass to _exit_chordmini_context.
    """
    _setup_paths()
    displaced = {}
    # ChordMini's internal code uses bare 'utils', 'src', etc.
    # The backend already imported its own 'utils'. We remove it from the cache
    # so Python will re-resolve it from sys.path (which now has _SRC_DIR first).
    for key in list(sys.modules.keys()):
        if key == "utils" or key.startswith("utils."):
            displaced[key] = sys.modules.pop(key)
    return displaced


def _exit_chordmini_context(displaced: dict):
    """Restore previously displaced modules."""
    # Remove any ChordMini utils that were loaded during inference
    for key in list(sys.modules.keys()):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    # Restore the backend's utils
    sys.modules.update(displaced)


def btc_chord_recognition(audio_path: str, output_lab_path: str, model_variant: str = "sl") -> bool:
    """
    Run BTC chord recognition on a single audio file and write a .lab file.

    Args:
        audio_path: Path to the input audio file.
        output_lab_path: Path where the output .lab file will be written.
        model_variant: 'sl' (Self-Label) or 'pl' (Pseudo-Label).

    Returns:
        True on success, False on failure.
    """
    variant = model_variant.lower()
    if variant not in _CHECKPOINTS:
        raise ValueError(f"Unknown model_variant '{model_variant}'. Use 'sl' or 'pl'.")

    checkpoint_path = _CHECKPOINTS[variant]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")

    displaced = _enter_chordmini_context()
    original_dir = os.getcwd()
    try:
        os.chdir(str(_THIS_DIR))

        import numpy as np
        import torch

        from src.evaluation.utils.common import (
            extract_norm_stats as _extract_norm_stats,
            extract_song_features as _extract_song_features,
        )
        from src.evaluation.utils.inference import predict_sliding_windows
        from src.models import load_model
        from src.utils import HParams, get_config_value as _cfg, get_device, idx2voca_chord, set_random_seed

        set_random_seed(42, include_python_random=True)
        config = HParams.load(_CONFIG_PATH)
        device = get_device()

        predict_args = argparse.Namespace(
            model_type="BTC",
            use_overlap=None,
            overlap_ratio=None,
            vote_aggregation="hard",
            kernel_size=9,
            use_gaussian=False,
            smooth_logits=False,
            smooth_predictions=False,
        )

        model, _, _ = load_model(checkpoint_path, "BTC", config, device, predict_args)
        mean, std = _extract_norm_stats(checkpoint_path)
        idx_to_chord = idx2voca_chord()
        model.idx_to_chord = idx_to_chord
        model.eval()

        # Resolve seq_len
        seq_len = None
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                seq_len = ckpt.get("timestep")
        except Exception:
            pass
        if seq_len is None and hasattr(model, "timestep"):
            seq_len = getattr(model, "timestep", None)
        if seq_len is None:
            seq_len = _cfg(config, "model", "seq_len", _cfg(config, "model", "timestep", 108))
        try:
            seq_len = int(seq_len)
        except Exception:
            seq_len = 108
        seq_len = max(seq_len, 1)

        # Extract features
        extracted = _extract_song_features(audio_path, config)
        if isinstance(extracted, tuple):
            features = extracted[0]
            frame_duration = float(extracted[1]) if len(extracted) > 1 else float(
                _cfg(config, "feature", "hop_duration",
                     _cfg(config, "feature", "hop_length", 2048) / _cfg(config, "mp3", "song_hz", 22050))
            )
        else:
            features = extracted
            frame_duration = float(
                _cfg(config, "feature", "hop_duration",
                     _cfg(config, "feature", "hop_length", 2048) / _cfg(config, "mp3", "song_hz", 22050))
            )
        features = np.asarray(features, dtype=np.float32)

        import librosa
        sample_rate = _cfg(config, "mp3", "song_hz", 22050)
        try:
            song_duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            audio_data, sr = librosa.load(audio_path, sr=sample_rate)
            song_duration = float(len(audio_data) / float(sr))

        # Run inference
        preds = predict_sliding_windows(
            model=model,
            feature_matrix=features,
            mean=mean,
            std=std,
            seq_len=seq_len,
            batch_size=16,
            model_type="BTC",
            n_classes=170,
            vote_aggregation=predict_args.vote_aggregation,
            use_overlap=predict_args.use_overlap,
            overlap_ratio=predict_args.overlap_ratio,
            smooth_logits=predict_args.smooth_logits,
            smooth_predictions=predict_args.smooth_predictions,
            kernel_size=predict_args.kernel_size,
            use_gaussian=predict_args.use_gaussian,
        )

        max_frames = int(np.floor(song_duration / frame_duration)) if frame_duration > 0 else len(preds)
        if max_frames > 0:
            preds = preds[:max_frames]

        # Build .lab segments
        lines = []
        if preds.size > 0:
            def _label(idx):
                return str(idx_to_chord.get(int(idx), idx_to_chord.get(169, "N")))

            prev = int(preds[0])
            start_t = 0.0
            for i in range(1, len(preds)):
                cur = int(preds[i])
                if cur != prev:
                    end_t = i * frame_duration
                    lines.append(f"{start_t:.6f} {end_t:.6f} {_label(prev)}\n")
                    start_t = end_t
                    prev = cur
            lines.append(f"{start_t:.6f} {len(preds) * frame_duration:.6f} {_label(prev)}\n")

        with open(output_lab_path, "w") as f:
            f.writelines(lines)

        return True

    finally:
        os.chdir(original_dir)
        _exit_chordmini_context(displaced)
