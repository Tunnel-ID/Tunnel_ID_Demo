from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from tunnelid_bio.config import PipelineConfig
from tunnelid_bio.evaluation.metrics import compute_eer, compute_roc, far_at_frr
from tunnelid_bio.types import SessionBundle
from tunnelid_bio.utils.math_utils import cosine_similarity


@dataclass
class EvaluationHarness:
    cfg: PipelineConfig
    sessions: List[SessionBundle] = field(default_factory=list)

    def add_session(self, session: SessionBundle) -> None:
        self.sessions.append(session)

    def extend(self, sessions: List[SessionBundle]) -> None:
        self.sessions.extend(sessions)

    def _bona_fide(self) -> List[SessionBundle]:
        return [s for s in self.sessions if not s.is_spoof and s.accepted]

    def _genuine_impostor_scores(self) -> tuple[np.ndarray, np.ndarray]:
        genuine = []
        impostor = []
        bona = self._bona_fide()

        for a, b in combinations(bona, 2):
            score = cosine_similarity(a.embedding, b.embedding)
            if a.identity_id == b.identity_id:
                genuine.append(score)
            else:
                impostor.append(score)

        return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)

    def _spoof_metrics(self) -> Dict[str, float]:
        spoof = [s for s in self.sessions if s.is_spoof]
        bona = [s for s in self.sessions if not s.is_spoof]

        if spoof:
            apcer = float(sum(1 for s in spoof if s.liveness_pass) / len(spoof))
        else:
            apcer = 0.0

        if bona:
            bpcer = float(sum(1 for s in bona if not s.liveness_pass) / len(bona))
        else:
            bpcer = 0.0

        return {"APCER": apcer, "BPCER": bpcer}

    def run(self) -> Dict[str, object]:
        genuine, impostor = self._genuine_impostor_scores()

        if genuine.size == 0 or impostor.size == 0:
            raise ValueError("Not enough accepted bona fide sessions to compute ROC")

        labels = np.concatenate(
            [np.ones_like(genuine, dtype=np.int32), np.zeros_like(impostor, dtype=np.int32)]
        )
        scores = np.concatenate([genuine, impostor])

        roc = compute_roc(labels, scores)
        eer, eer_thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])
        far05, far_thr = far_at_frr(
            roc["fpr"],
            roc["fnr"],
            roc["thresholds"],
            self.cfg.evaluation.default_frr_target,
        )

        auth_times = [s.auth_time_ms for s in self.sessions if s.auth_time_ms > 0]
        retry_rate = float(sum(1 for s in self.sessions if not s.accepted) / max(1, len(self.sessions)))

        reasons: Dict[str, int] = {}
        for s in self.sessions:
            if s.failure_reason:
                reasons[s.failure_reason] = reasons.get(s.failure_reason, 0) + 1

        out = {
            "n_sessions": len(self.sessions),
            "n_genuine_scores": int(genuine.size),
            "n_impostor_scores": int(impostor.size),
            "genuine_scores": genuine,
            "impostor_scores": impostor,
            "roc": roc,
            "EER": eer,
            "EER_threshold": eer_thr,
            "FAR_at_FRR_target": far05,
            "FAR_threshold": far_thr,
            "FRR_target": self.cfg.evaluation.default_frr_target,
            "median_auth_time_ms": float(np.median(auth_times)) if auth_times else 0.0,
            "retry_rate": retry_rate,
            "failure_reasons": reasons,
        }
        out.update(self._spoof_metrics())
        return out

    def save_binary(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        ids = np.array([s.identity_id for s in self.sessions], dtype=object)
        sids = np.array([s.session_id for s in self.sessions], dtype=object)
        embeds = np.stack([s.embedding for s in self.sessions]).astype(np.float32)
        spoof = np.array([s.is_spoof for s in self.sessions], dtype=np.bool_)
        accepted = np.array([s.accepted for s in self.sessions], dtype=np.bool_)
        liveness = np.array([s.liveness_pass for s in self.sessions], dtype=np.bool_)
        atime = np.array([s.auth_time_ms for s in self.sessions], dtype=np.float32)
        freason = np.array([s.failure_reason or "" for s in self.sessions], dtype=object)

        np.savez_compressed(
            p,
            identity_id=ids,
            session_id=sids,
            embeddings=embeds,
            is_spoof=spoof,
            accepted=accepted,
            liveness_pass=liveness,
            auth_time_ms=atime,
            failure_reason=freason,
        )

    @classmethod
    def load_binary(cls, cfg: PipelineConfig, path: str | Path) -> "EvaluationHarness":
        data = np.load(Path(path), allow_pickle=True)
        harness = cls(cfg=cfg)

        n = data["embeddings"].shape[0]
        for i in range(n):
            harness.add_session(
                SessionBundle(
                    session_id=str(data["session_id"][i]),
                    identity_id=str(data["identity_id"][i]),
                    embedding=np.asarray(data["embeddings"][i], dtype=np.float32),
                    liveness_pass=bool(data["liveness_pass"][i]),
                    accepted=bool(data["accepted"][i]),
                    is_spoof=bool(data["is_spoof"][i]),
                    auth_time_ms=float(data["auth_time_ms"][i]),
                    failure_reason=(str(data["failure_reason"][i]) or None),
                )
            )
        return harness
