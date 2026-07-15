"""Locate manually downloaded antelopev2 face models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaceModelFile:
    relative_path: Path
    minimum_bytes: int


FACE_MODEL_FILES = (
    FaceModelFile(
        Path("recognition/model.onnx"),
        200_000_000,
    ),
    FaceModelFile(
        Path("detection/model.onnx"),
        10_000_000,
    ),
)


def _valid_face_model(path: Path, minimum_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def face_models_complete(model_dir: Path) -> bool:
    return all(
        _valid_face_model(model_dir / spec.relative_path, spec.minimum_bytes)
        for spec in FACE_MODEL_FILES
    )


def locate_face_model_dir(explicit: Path | None = None) -> Path:
    """Resolve the antelopev2 directory.

    Priority: explicit arg > ANTELOPEV2_DIR env var >
    <toolkit root>/models/antelopev2.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ANTELOPEV2_DIR")
    if env:
        return Path(env).expanduser().resolve()
    toolkit_root = Path(__file__).resolve().parents[2]
    return (toolkit_root / "models" / "antelopev2").resolve()


def ensure_face_models(
    model_dir: Path,
) -> Path:
    model_dir = Path(model_dir).expanduser().resolve()
    missing = []
    for spec in FACE_MODEL_FILES:
        destination = model_dir / spec.relative_path
        if not _valid_face_model(destination, spec.minimum_bytes):
            missing.append(str(destination))
    if missing:
        raise FileNotFoundError(
            "Missing or incomplete antelopev2 files (automatic downloads are "
            "disabled): " + ", ".join(missing)
        )
    return model_dir
