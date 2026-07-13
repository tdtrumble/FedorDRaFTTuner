#!/usr/bin/env python3
"""Export SAM 3D Body scans from DRaFT reference images.

Uses the same detector-free, full-image, body-only inference path as
``toolkit.rewards.body.BodyGeometryReward`` (``inference_type="body"``).

Outputs per reference image:
  - overlay.jpg   skeleton + mesh overlay (requires pyrender)
  - params.npz    MHR shape/scale params and metadata
  - mesh.obj      posed mesh in camera space (optional, default on)

Also writes the averaged reference prototype used during DRaFT training.

Example:
  python scripts/export_sam_body_scans.py datasets/subject
  python scripts/export_sam_body_scans.py datasets/subject -o output/sam_body_scans/subject --soma
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from toolkit.rewards.body import (  # noqa: E402
    SAM3D_REPO_DIR,
    TOOLKIT_ROOT as BODY_TOOLKIT_ROOT,
    _ensure_sam3d_importable,
    _reference_paths,
    _resolve_checkpoint,
)

assert TOOLKIT_ROOT == BODY_TOOLKIT_ROOT


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _canonical_vertices(
    shape_params: np.ndarray,
    scale_params: np.ndarray,
    native_scales_fn,
    device: torch.device,
    soma_lod: str,
) -> np.ndarray | None:
    try:
        from soma import SOMALayer
    except ImportError:
        return None

    shape = torch.as_tensor(shape_params, dtype=torch.float32, device=device).reshape(1, -1)
    scale = torch.as_tensor(scale_params, dtype=torch.float32, device=device).reshape(1, -1)
    native_scale = native_scales_fn(scale)
    layer = SOMALayer(
        device=device,
        identity_model_type="mhr",
        lod=soma_lod,
        enable_procedural_transforms=False,
    )
    poses = torch.zeros(1, 77, 3, device=device, dtype=torch.float32)
    out = layer(
        poses=poses,
        identity_coeffs=shape,
        scale_params=native_scale,
        apply_correctives=False,
    )
    verts = getattr(out, "vertices", None)
    if verts is None and isinstance(out, dict):
        verts = out.get("vertices")
    if verts is None:
        return None
    return verts.detach().cpu().numpy()[0]


def _render_overlay(bgr: np.ndarray, outputs: list, faces: np.ndarray) -> np.ndarray | None:
    if str(SAM3D_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(SAM3D_REPO_DIR))
    try:
        from tools.vis_utils import visualize_sample_together
    except ImportError as exc:
        print(f"overlay skipped (import failed): {exc}")
        return None
    try:
        return visualize_sample_together(bgr, outputs, faces)
    except Exception as exc:  # noqa: BLE001
        print(f"overlay skipped (render failed): {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export SAM 3D Body reference scans (same path as DRaFT body reward)."
    )
    parser.add_argument(
        "reference_images",
        type=str,
        help="Reference image file or folder (same as draft.reward.reference_images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output folder (default: output/sam_body_scans/<dataset_name>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model.ckpt (default: models/sam-3d-body-vith or HF hub)",
    )
    parser.add_argument(
        "--sam3d-hf-repo",
        type=str,
        default="facebook/sam-3d-body-vith",
        help="Hugging Face repo when checkpoint is not local",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip overlay.jpg (still writes params.npz)",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip per-image mesh.obj export",
    )
    parser.add_argument(
        "--soma",
        action="store_true",
        help="Also export canonical neutral-pose prototype mesh via SOMA-X",
    )
    parser.add_argument(
        "--soma-lod",
        type=str,
        default="low",
        help="SOMA-X LOD for canonical mesh export",
    )
    args = parser.parse_args()

    ref_root = Path(args.reference_images)
    if not args.output:
        out_dir = TOOLKIT_ROOT / "output" / "sam_body_scans" / ref_root.name
    else:
        out_dir = Path(args.output)
    per_dir = out_dir / "per_image"
    proto_dir = out_dir / "prototype"
    per_dir.mkdir(parents=True, exist_ok=True)
    proto_dir.mkdir(parents=True, exist_ok=True)

    paths = _reference_paths(ref_root)
    device = torch.device(args.device)

    _ensure_sam3d_importable()
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    ckpt_path, mhr_path = _resolve_checkpoint(
        args.checkpoint or None, args.sam3d_hf_repo
    )
    print(f"checkpoint: {ckpt_path}")
    model, cfg = load_sam_3d_body(
        checkpoint_path=ckpt_path, device=str(device), mhr_path=mhr_path
    )
    model.eval()
    estimator = SAM3DBodyEstimator(model, cfg)
    faces = estimator.faces

    def native_scales(scale_params: torch.Tensor) -> torch.Tensor:
        head = model.head_pose
        return head.scale_mean[None, :].to(scale_params) + scale_params @ head.scale_comps.to(
            scale_params
        )

    shapes: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    valid: list[str] = []
    skipped: list[dict[str, str]] = []
    records: list[dict] = []

    for path in paths:
        stem = path.stem
        item_dir = per_dir / stem
        item_dir.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append({"path": str(path), "reason": "unreadable image"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            outs = estimator.process_one_image(rgb, inference_type="body")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        if not outs:
            skipped.append({"path": str(path), "reason": "no person/body detected"})
            continue

        out = outs[0]
        shape = np.asarray(out["shape_params"], dtype=np.float32)
        scale = np.asarray(out["scale_params"], dtype=np.float32)
        shapes.append(shape)
        scales.append(scale)
        valid.append(str(path))

        np.savez_compressed(
            item_dir / "params.npz",
            shape_params=shape,
            scale_params=scale,
            bbox=np.asarray(out["bbox"], dtype=np.float32),
            focal_length=np.float32(out["focal_length"]),
            pred_cam_t=np.asarray(out["pred_cam_t"], dtype=np.float32),
            source_path=str(path),
        )

        if not args.no_mesh:
            _write_obj(
                item_dir / "mesh.obj",
                np.asarray(out["pred_vertices"], dtype=np.float32),
                faces,
            )

        if not args.no_overlays:
            overlay = _render_overlay(bgr, outs, faces)
            if overlay is not None:
                cv2.imwrite(str(item_dir / "overlay.jpg"), overlay.astype(np.uint8))

        records.append(
            {
                "source": str(path),
                "output_dir": str(item_dir),
                "shape_dim": int(shape.shape[0]),
                "scale_dim": int(scale.shape[0]),
            }
        )
        print(f"ok: {path.name}")

    if not shapes:
        print("no valid body scans; nothing exported")
        for item in skipped:
            print(f"  skipped {item['path']}: {item['reason']}")
        return 1

    proto_shape = np.stack(shapes).mean(axis=0)
    proto_scale = np.stack(scales).mean(axis=0)
    np.savez_compressed(
        proto_dir / "prototype_shape_scale.npz",
        shape_params=proto_shape,
        scale_params=proto_scale,
        num_references=len(shapes),
    )

    soma_note = None
    if args.soma:
        canon = _canonical_vertices(
            proto_shape,
            proto_scale,
            native_scales,
            device,
            args.soma_lod,
        )
        if canon is None:
            soma_note = "SOMA-X unavailable; install py-soma-x warp-lang for canonical mesh"
            print(soma_note)
        else:
            _write_obj(proto_dir / "prototype_canonical_neutral.obj", canon, faces)
            print(f"wrote canonical prototype mesh ({canon.shape[0]} verts)")

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(ref_root.resolve()),
        "inference_type": "body",
        "checkpoint": ckpt_path,
        "output_dir": str(out_dir.resolve()),
        "valid_count": len(valid),
        "skipped_count": len(skipped),
        "valid_reference_images": valid,
        "skipped_reference_images": skipped,
        "prototype_shape_dim": int(proto_shape.shape[0]),
        "prototype_scale_dim": int(proto_scale.shape[0]),
        "per_image": records,
        "soma": soma_note,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nexported {len(valid)} scans -> {out_dir}")
    if skipped:
        print(f"skipped {len(skipped)} images (see summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
