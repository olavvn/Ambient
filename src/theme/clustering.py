"""
임베딩 공간에서 반복 패턴을 클러스터링하여 테마 후보를 선별
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from dataclasses import dataclass, field
from typing import List


@dataclass
class ThemeCandidate:
    """테마 후보"""
    centroid: np.ndarray
    member_indices: List[int]
    representative_idx: int
    score: float


def cluster_segments(
    embeddings: np.ndarray,
    method: str = "dbscan",
    eps: float = 0.3,
    min_samples: int = 3,
    n_clusters: int = 5,
) -> List[ThemeCandidate]:
    """
    임베딩 클러스터링 → 테마 후보 목록 반환
    결과는 score 내림차순 정렬
    """
    embeddings = normalize(embeddings)

    if method == "dbscan":
        labels = DBSCAN(
            eps=eps, min_samples=min_samples,
            metric="cosine", algorithm="brute",
        ).fit_predict(embeddings)
    else:
        k = min(n_clusters, len(embeddings))
        labels = KMeans(n_clusters=k, random_state=42,
                        n_init="auto").fit_predict(embeddings)

    candidates: List[ThemeCandidate] = []
    unique_labels = set(labels) - {-1}

    for lbl in unique_labels:
        member_idx = np.where(labels == lbl)[0].tolist()
        if len(member_idx) < 2:
            continue
        cluster_embs = embeddings[member_idx]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        dists = 1 - cluster_embs @ centroid
        rep_idx = member_idx[int(np.argmin(dists))]

        intra_dist = float(dists.mean())
        score = len(member_idx) / (intra_dist + 1e-6)

        candidates.append(ThemeCandidate(
            centroid=centroid,
            member_indices=member_idx,
            representative_idx=rep_idx,
            score=score,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_theme(candidates: List[ThemeCandidate],
                 strategy: str = "top1") -> ThemeCandidate:
    """
    테마 후보 중 최종 테마 선택.

    strategy:
        "top1"    - 가장 높은 score
        "diverse" - 상위 3개 중 가장 초반에 등장하는 것
    """
    if not candidates:
        raise ValueError("No theme candidates found")

    if strategy == "top1" or len(candidates) == 1:
        return candidates[0]
    elif strategy == "diverse":
        top3 = candidates[:3]
        return min(top3, key=lambda c: min(c.member_indices))
    return candidates[0]
