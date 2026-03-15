from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import typer

from tunnelid_bio.config import load_config
from tunnelid_bio.dataset.local import VALID_LABELS, store_session
from tunnelid_bio.evaluation.pad_eval import run_pad_eval
from tunnelid_bio.evaluation.runner import evaluate_dataset
from tunnelid_bio.pipeline import BiometricPipeline
from tunnelid_bio.types import FramePacket
from tunnelid_bio.utils.io import (
    load_embedding_binary,
    load_json,
    load_template_set,
    resolve_template_paths,
    save_embedding_binary,
    save_json,
    save_template_set,
)
from tunnelid_bio.utils.math_utils import cosine_similarity

app = typer.Typer(help="TunnelID standalone biometric module CLI")


def _pose_instruction(pose: str) -> str:
    mapping = {
        "center": "Look straight at camera",
        "yaw_left": "Turn head LEFT (20-30 deg)",
        "yaw_right": "Turn head RIGHT (20-30 deg)",
        "pitch_up": "Tilt head UP (15-20 deg)",
        "pitch_down": "Tilt head DOWN (15-20 deg)",
    }
    return mapping.get(pose, pose)


def _capture_from_camera(
    duration_s: float,
    fps: float,
    camera_id: int,
    guided_poses: Optional[List[str]] = None,
    show_window: bool = True,
    liveness_prompt: bool = True,
    liveness_grace_s: float = 1.5,
) -> List[FramePacket]:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_id}")

    frames: List[FramePacket] = []
    interval = 1.0 / max(1.0, fps)
    start = time.time()
    next_t = start

    try:
        if not guided_poses:
            while time.time() - start < duration_s:
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.time()
                if now >= next_t:
                    if show_window:
                        overlay = frame.copy()
                        cv2.putText(
                            overlay,
                            "Capturing...",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        if liveness_prompt:
                            cv2.putText(
                                overlay,
                                "Liveness: blink or turn head now",
                                (20, 78),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        cv2.imshow("TunnelID Capture", overlay)
                        cv2.waitKey(1)
                    frames.append(
                        FramePacket(
                            frame_id=len(frames),
                            image_bgr=frame,
                            timestamp_ms=now * 1000.0,
                            pose_hint=None,
                        )
                    )
                    next_t += interval
        else:
            per_pose_duration = max(1.2, duration_s / max(1, len(guided_poses)))
            for i, pose in enumerate(guided_poses, start=1):
                msg = f"[{i}/{len(guided_poses)}] {pose}: {_pose_instruction(pose)}"
                print(msg, flush=True)
                pose_start = time.time()
                while time.time() - pose_start < per_pose_duration:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    now = time.time()
                    if now < next_t:
                        continue
                    next_t = now + interval
                    if show_window:
                        overlay = frame.copy()
                        cv2.putText(
                            overlay,
                            f"Pose {i}/{len(guided_poses)}: {pose}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            overlay,
                            _pose_instruction(pose),
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if liveness_prompt:
                            cv2.putText(
                                overlay,
                                "Liveness: blink or turn head now",
                                (20, 116),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        cv2.imshow("TunnelID Capture", overlay)
                        cv2.waitKey(1)
                    frames.append(
                        FramePacket(
                            frame_id=len(frames),
                            image_bgr=frame,
                            timestamp_ms=now * 1000.0,
                            pose_hint=pose,
                        )
                    )

            if liveness_prompt and liveness_grace_s > 0.0:
                print("Liveness window: please blink or smile now", flush=True)
                grace_start = time.time()
                while time.time() - grace_start < liveness_grace_s:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    now = time.time()
                    if now < next_t:
                        continue
                    next_t = now + interval
                    if show_window:
                        overlay = frame.copy()
                        cv2.putText(
                            overlay,
                            "Liveness window: blink or turn head now",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("TunnelID Capture", overlay)
                        cv2.waitKey(1)
                    frames.append(
                        FramePacket(
                            frame_id=len(frames),
                            image_bgr=frame,
                            timestamp_ms=now * 1000.0,
                            pose_hint=None,
                        )
                    )
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    return frames


def _frames_from_video(path: Path, max_frames: int = 120) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    out = []
    try:
        while len(out) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame)
    finally:
        cap.release()
    return out


def _load_frames(
    input_source: str,
    duration_s: float,
    fps: float,
    camera_id: int,
    guided_poses: Optional[List[str]] = None,
    show_window: bool = True,
    liveness_prompt: bool = True,
    liveness_grace_s: float = 1.5,
) -> List[np.ndarray | FramePacket]:
    if input_source == "camera":
        return _capture_from_camera(
            duration_s=duration_s,
            fps=fps,
            camera_id=camera_id,
            guided_poses=guided_poses,
            show_window=show_window,
            liveness_prompt=liveness_prompt,
            liveness_grace_s=liveness_grace_s,
        )

    p = Path(input_source)
    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return _frames_from_video(p)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Unable to read image: {p}")
        return [img.copy() for _ in range(90)]

    raise RuntimeError("Unsupported input source. Use camera, video path, or image path.")


def _load_thresholds(thresholds_path: Optional[str]) -> dict:
    if not thresholds_path:
        return {}
    p = Path(thresholds_path)
    if not p.exists():
        raise RuntimeError(f"Threshold file not found: {p}")
    return load_json(p)


def _load_templates_and_meta(user_id: str, template_dir: str) -> Tuple[np.ndarray, dict]:
    npz_path, npy_path, meta_path = resolve_template_paths(template_dir, user_id)
    meta = load_json(meta_path) if meta_path.exists() else {}

    if npz_path.exists():
        return load_template_set(npz_path), meta
    if npy_path.exists():
        return load_embedding_binary(npy_path).reshape(1, -1), meta
    raise RuntimeError(f"Template not found for user: {user_id}")


def _merge_template_set(existing: np.ndarray, new_embedding: np.ndarray, max_k: int, quality_history: List[float]) -> np.ndarray:
    new_row = np.asarray(new_embedding, dtype=np.float32).reshape(1, -1)
    old = np.asarray(existing, dtype=np.float32)
    if old.ndim == 1:
        old = old.reshape(1, -1)
    merged = np.vstack([old, new_row])

    if merged.shape[0] <= max_k:
        return merged

    if len(quality_history) == merged.shape[0]:
        idx = np.argsort(np.asarray(quality_history, dtype=np.float64))[::-1][:max_k]
        return merged[idx]

    return merged[-max_k:]


@app.command("enroll")
def enroll(
    user_id: str = typer.Option(..., "--user-id"),
    input_source: str = typer.Option("camera", "--input", help="camera|video_path|image_path"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    duration_s: float = typer.Option(4.0, "--duration-s"),
    fps: float = typer.Option(15.0, "--fps"),
    camera_id: int = typer.Option(0, "--camera-id"),
    template_dir: str = typer.Option("data/templates", "--template-dir"),
    guided: bool = typer.Option(True, "--guided/--no-guided"),
    show_window: bool = typer.Option(True, "--show-window/--no-window"),
    skip_liveness: bool = typer.Option(False, "--skip-liveness"),
    liveness_grace_s: float = typer.Option(1.5, "--liveness-grace-s", min=0.0, max=10.0),
    thresholds: Optional[str] = typer.Option(None, "--thresholds", help="thresholds_dev.json artifact"),
    max_templates: int = typer.Option(3, "--max-templates", min=1, max=10),
) -> None:
    cfg = load_config(config)
    pipeline = BiometricPipeline(cfg)
    poses = cfg.capture.required_poses if guided and input_source == "camera" else None
    frames = _load_frames(
        input_source,
        duration_s=duration_s,
        fps=fps,
        camera_id=camera_id,
        guided_poses=poses,
        show_window=show_window,
        liveness_prompt=True,
        liveness_grace_s=liveness_grace_s,
    )

    thr = _load_thresholds(thresholds)
    npz_path, npy_path, meta_path = resolve_template_paths(template_dir, user_id)

    existing = None
    quality_history: List[float] = []
    if npz_path.exists() or npy_path.exists():
        existing, meta = _load_templates_and_meta(user_id, template_dir)
        quality_history = [float(x) for x in meta.get("template_qualities", [])] if meta else []

    result = pipeline.enroll_session(
        frames,
        enforce_liveness=not skip_liveness,
        thresholds=thr,
        existing_templates=existing,
    )

    if result["status"] == "ACCEPTED" and result["embedding"] is not None:
        emb = np.asarray(result["embedding"], dtype=np.float32)
        if existing is None:
            merged = emb.reshape(1, -1)
            quality_history = [float(result.get("quality_metrics", {}).get("quality_score", 0.0))]
        else:
            quality_history.append(float(result.get("quality_metrics", {}).get("quality_score", 0.0)))
            merged = _merge_template_set(existing, emb, max_k=max_templates, quality_history=quality_history)
            if len(quality_history) > merged.shape[0]:
                quality_history = quality_history[-merged.shape[0] :]

        save_template_set(npz_path, merged)
        save_embedding_binary(npy_path, np.mean(merged, axis=0).astype(np.float32))
        save_json(
            meta_path,
            {
                "user_id": user_id,
                "created_at_epoch_s": time.time(),
                "model_versions": result["model_versions"],
                "quality_metrics": result["quality_metrics"],
                "template_count": int(merged.shape[0]),
                "template_qualities": quality_history,
            },
        )
        result["embedding"] = f"BINARY_SAVED:{npz_path}"
    else:
        result["embedding"] = None

    print(json.dumps(result, indent=2, default=float))


@app.command("verify")
def verify(
    user_id: str = typer.Option(..., "--user-id"),
    input_source: str = typer.Option("camera", "--input"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    duration_s: float = typer.Option(4.0, "--duration-s"),
    fps: float = typer.Option(15.0, "--fps"),
    camera_id: int = typer.Option(0, "--camera-id"),
    template_dir: str = typer.Option("data/templates", "--template-dir"),
    guided: bool = typer.Option(True, "--guided/--no-guided"),
    show_window: bool = typer.Option(True, "--show-window/--no-window"),
    skip_liveness: bool = typer.Option(False, "--skip-liveness"),
    liveness_grace_s: float = typer.Option(1.5, "--liveness-grace-s", min=0.0, max=10.0),
    thresholds: Optional[str] = typer.Option(None, "--thresholds", help="thresholds_dev.json artifact"),
    allow_model_mismatch: bool = typer.Option(False, "--allow-model-mismatch/--block-model-mismatch"),
) -> None:
    stored, meta = _load_templates_and_meta(user_id, template_dir)

    cfg = load_config(config)
    pipeline = BiometricPipeline(cfg)
    poses = cfg.capture.required_poses if guided and input_source == "camera" else None
    frames = _load_frames(
        input_source,
        duration_s=duration_s,
        fps=fps,
        camera_id=camera_id,
        guided_poses=poses,
        show_window=show_window,
        liveness_prompt=True,
        liveness_grace_s=liveness_grace_s,
    )

    thr = _load_thresholds(thresholds)
    result = pipeline.verify_session(
        frames,
        stored,
        enforce_liveness=not skip_liveness,
        thresholds=thr,
        enrolled_model_versions=meta.get("model_versions"),
        allow_model_mismatch=allow_model_mismatch,
    )
    print(json.dumps(result, indent=2, default=float))


@app.command("collect")
def collect(
    user_id: str = typer.Option(..., "--user-id"),
    label: str = typer.Option(
        "bona_fide",
        "--label",
        help="bona_fide|attack_print|attack_screen|attack_video|attack_deepfake_screen",
    ),
    input_source: str = typer.Option("camera", "--input", help="camera|video_path|image_path"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    duration_s: float = typer.Option(4.0, "--duration-s"),
    fps: float = typer.Option(15.0, "--fps"),
    camera_id: int = typer.Option(0, "--camera-id"),
    dataset_root: str = typer.Option("data/dataset", "--dataset"),
    template_dir: str = typer.Option("data/templates", "--template-dir"),
    thresholds: Optional[str] = typer.Option(None, "--thresholds"),
    device_id: str = typer.Option("cam0", "--device-id"),
    lighting: str = typer.Option("unknown", "--lighting"),
    distance: str = typer.Option("unknown", "--distance"),
    glasses: str = typer.Option("unknown", "--glasses"),
    guided: bool = typer.Option(True, "--guided/--no-guided"),
    show_window: bool = typer.Option(True, "--show-window/--no-window"),
    store_frames: bool = typer.Option(False, "--store-frames/--no-store-frames"),
    liveness_grace_s: float = typer.Option(1.5, "--liveness-grace-s", min=0.0, max=10.0),
) -> None:
    if label not in VALID_LABELS:
        raise RuntimeError(f"Unsupported label: {label}")

    cfg = load_config(config)
    pipeline = BiometricPipeline(cfg)
    poses = cfg.capture.required_poses if guided and input_source == "camera" and label == "bona_fide" else None
    frames = _load_frames(
        input_source,
        duration_s=duration_s,
        fps=fps,
        camera_id=camera_id,
        guided_poses=poses,
        show_window=show_window,
        liveness_prompt=True,
        liveness_grace_s=liveness_grace_s,
    )

    thr = _load_thresholds(thresholds)
    result = pipeline.enroll_session(frames, enforce_liveness=False, thresholds=thr)

    if result["status"] != "ACCEPTED" or result["embedding"] is None:
        print(json.dumps({"status": "REJECTED", "reason": result["reason"], "reason_codes": result.get("reason_codes", [])}, indent=2))
        return

    try:
        stored, _ = _load_templates_and_meta(user_id, template_dir)
        similarity_to_enroll = float(
            np.max(
                [
                    cosine_similarity(np.asarray(result["embedding"], dtype=np.float32), e)
                    for e in stored
                ]
            )
        )
    except RuntimeError:
        similarity_to_enroll = 0.0

    frame_images = [f.image_bgr if isinstance(f, FramePacket) else f for f in frames]
    session_dir = store_session(
        dataset_root=dataset_root,
        user_id=user_id,
        label=label,
        device_id=device_id,
        embedding=np.asarray(result["embedding"], dtype=np.float32),
        liveness_pass=bool(result["liveness_pass"]),
        liveness_confidence=float(result["liveness_confidence"]),
        quality_score=float(result.get("quality_metrics", {}).get("quality_score", 0.0)),
        similarity_to_enroll=similarity_to_enroll,
        metadata={
            "reason": result.get("reason", "OK"),
            "reason_codes": result.get("reason_codes", []),
            "model_versions": result.get("model_versions", {}),
            "quality_metrics": result.get("quality_metrics", {}),
            "output_metadata": result.get("metadata", {}),
        },
        lighting=lighting,
        distance=distance,
        glasses=glasses,
        store_frames=store_frames,
        frames_bgr=frame_images if store_frames else None,
    )

    print(json.dumps({"status": "COLLECTED", "session_dir": str(session_dir)}, indent=2))


@app.command("pad-eval")
def pad_eval(
    dataset: str = typer.Option("data/pad_dataset", "--dataset"),
    out: str = typer.Option("artifacts/pad", "--out"),
    target_bpcer: float = typer.Option(0.03, "--target-bpcer"),
) -> None:
    artifact = run_pad_eval(dataset_root=dataset, out_dir=out, target_bpcer=target_bpcer)
    print(json.dumps(artifact, indent=2, default=float))


@app.command("eval")
def eval_cmd(
    dataset: str = typer.Option("data/dataset", "--dataset"),
    out: str = typer.Option("artifacts", "--out"),
    policy: str = typer.Option("security-first", "--policy", help="security-first|ux-first|balanced"),
    calibration_method: str = typer.Option("none", "--calibration", help="none|platt|isotonic"),
    max_frr: float = typer.Option(0.03, "--max-frr"),
    max_far: float = typer.Option(1e-4, "--max-far"),
) -> None:
    if policy not in {"security-first", "ux-first", "balanced"}:
        raise RuntimeError(f"Invalid policy: {policy}")
    if calibration_method not in {"none", "platt", "isotonic"}:
        raise RuntimeError(f"Invalid calibration mode: {calibration_method}")

    artifact = evaluate_dataset(
        dataset_root=dataset,
        out_dir=out,
        policy=policy,
        calibration_method=calibration_method,
        max_frr=max_frr,
        max_far=max_far,
    )
    print(json.dumps(artifact["recommended"], indent=2, default=float))


@app.command("calibrate")
def calibrate(
    dataset: str = typer.Option("data/dataset", "--dataset"),
    out: str = typer.Option("artifacts", "--out"),
    policy: str = typer.Option("security-first", "--policy", help="security-first|ux-first|balanced"),
    calibration_method: str = typer.Option("none", "--calibration", help="none|platt|isotonic"),
    max_frr: float = typer.Option(0.03, "--max-frr"),
    max_far: float = typer.Option(1e-4, "--max-far"),
    output_file: str = typer.Option("thresholds_dev.json", "--output-file"),
) -> None:
    artifact = evaluate_dataset(
        dataset_root=dataset,
        out_dir=out,
        policy=policy,
        calibration_method=calibration_method,
        max_frr=max_frr,
        max_far=max_far,
    )

    out_path = Path(out) / output_file
    out_path.write_text(json.dumps(artifact, indent=2, default=float))
    print(json.dumps({"thresholds_artifact": str(out_path)}, indent=2))


@app.command("simulate-spoof")
def simulate_spoof(
    spoof_source: str = typer.Option(..., "--spoof-source", help="image/video replay source"),
    user_id: str = typer.Option(..., "--user-id"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    template_dir: str = typer.Option("data/templates", "--template-dir"),
    thresholds: Optional[str] = typer.Option(None, "--thresholds"),
) -> None:
    stored, meta = _load_templates_and_meta(user_id, template_dir)
    cfg = load_config(config)
    pipeline = BiometricPipeline(cfg)

    frames = _load_frames(spoof_source, duration_s=4.0, fps=15.0, camera_id=0)
    result = pipeline.verify_session(
        frames,
        stored,
        thresholds=_load_thresholds(thresholds),
        enrolled_model_versions=meta.get("model_versions"),
    )
    result["spoof_simulation"] = True

    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    app()
