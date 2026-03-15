# TunnelID Biometric Module (Standalone)

Deterministic, standalone face-authentication module focused on secure employee login in controlled environments.

## What This Implements

- Guided multi-view capture with required pose protocol:
  - `center`, `yaw_left`, `yaw_right`, `pitch_up`, `pitch_down`
- Per-frame quality gating:
  - detection confidence, sharpness, exposure, occlusion, pose error, motion blur
- Identity embedding extraction:
  - ArcFace-style 512-D embeddings via pretrained `insightface`
  - L2 normalized per frame
- Multi-frame template aggregation:
  - quality-weighted mean
  - outlier rejection by cosine distance
  - intra-session variance checks
- Liveness detection:
  - passive (texture, moire, frequency)
  - temporal (face-ROI optical flow, consistency, blink, micro-motion, parallax, optional rPPG)
  - active challenge (blink twice or turn head, deterministic prompt selection)
  - security-first fusion: `overall=min(active, max(passive, temporal))` with replay-suspect gating
- API outputs include:
  - embedding `E` (`np.ndarray`, shape `(512,)`)
  - liveness pass/fail + confidence
  - quality and diagnostics metadata
- Evaluation harness:
  - genuine/impostor score distributions
  - ROC, EER, FAR at fixed FRR
  - APCER/BPCER
  - median auth time, retry rate, failure reasons

No raw images are persisted by default. Embeddings are saved as binary `.npy`/`.npz` (not plaintext logs).

## Folder Structure

```
facerec/
  configs/
    default.yaml
  scripts/
    calibrate_thresholds.py
    evaluate_sessions.py
  tunnelid_bio/
    aggregation/
      aggregator.py
    capture/
      face_analyzer.py
      pose.py
      protocol.py
      quality.py
    embedding/
      alignment.py
      encoder.py
    evaluation/
      harness.py
      metrics.py
    liveness/
      active.py
      fusion.py
      passive.py
      temporal.py
    pipeline/
      api.py
      service.py
    utils/
      io.py
      math_utils.py
    config.py
    types.py
    versioning.py
  data/
    templates/
```

## Run Locally

1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Enroll user

```bash
python -m tunnelid_bio.cli enroll --user-id alice --input camera --config configs/default.yaml
```

3. Verify user

```bash
python -m tunnelid_bio.cli verify --user-id alice --input camera --config configs/default.yaml

Use guided capture prompts (default on) with window:

```bash
python -m tunnelid_bio.cli enroll --user-id alice --input camera --guided --show-window
```
```

4. Simulate spoof (photo/screen replay style input)

```bash
python -m tunnelid_bio.cli simulate-spoof --user-id alice --spoof-source path/to/replay.jpg --config configs/default.yaml
```

5. PAD evaluation over labeled dataset

```bash
python -m tunnelid_bio.cli pad-eval --dataset data/pad_dataset --out artifacts/pad
```

## Python API

```python
from tunnelid_bio.pipeline import enroll_session, verify_session

enroll_out = enroll_session(frames_sequence, config_path="configs/default.yaml")
verify_out = verify_session(frames_sequence, stored_embedding, config_path="configs/default.yaml")
```

`enroll_session` output:

- `embedding`: `(512,)` float vector on success, else `None`
- `liveness_pass`: `bool`
- `quality_metrics`: dict
- `status`: `ACCEPTED` or `REJECTED`
- `reason`: reject reason

`verify_session` output:

- `similarity_score`: cosine similarity
- `liveness_pass`: `bool`
- `decision`: `MATCH` or `NO_MATCH`
- `reason`: reject reason

## Evaluation Harness

Use `tunnelid_bio.evaluation.EvaluationHarness` with multiple sessions/identities.

Metrics returned:

- Genuine vs impostor score distributions
- ROC arrays (`fpr`, `tpr`, `fnr`, `thresholds`)
- EER + EER threshold
- FAR at configured FRR target
- APCER, BPCER
- Median authentication time
- Retry rate and failure reasons

Run summary over saved `.npz` sessions:

```bash
python scripts/evaluate_sessions.py --sessions data/eval_sessions.npz --config configs/default.yaml
```

## Threshold Calibration

Security-first threshold calibration script:

```bash
python scripts/calibrate_thresholds.py \
  --sessions data/eval_sessions.npz \
  --config configs/default.yaml \
  --target-far 0.001 \
  --output configs/calibrated.yaml
```

Writes a config override with calibrated `embedding.min_cosine_similarity_match`.

## PAD Dataset Labels

- `bona_fide`
- `attack_print`
- `attack_screen`
- `attack_video`
- `attack_deepfake_screen`

## Design Notes

- Determinism:
  - fixed seed (`2026`), fixed OpenCV thread count
  - deterministic active prompt selection
- Modularity:
  - capture, embedding, aggregation, liveness, evaluation separated
- Version tags:
  - encoder, preprocessing, liveness, aggregation are emitted with results
- Security controls:
  - no long-term raw image storage
  - no plaintext embedding logs

## Extending to Fuzzy Extractor Later

Use the returned stable template `E` as the fuzzy extractor input:

1. Optional quantization/feature binarization stage over `E`
2. Helper-data generation (secure sketch / code-offset)
3. Key reconstruction in verify path from fresh `E'`
4. Keep matcher thresholding and liveness independent from key derivation logic

This module is intentionally independent so fuzzy-extractor integration can be attached without changing capture/liveness internals.
