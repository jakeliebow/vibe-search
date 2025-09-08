from typing import Dict
from src.models.detection import YoloObjectTrack, MarAtIndex
import numpy as np


def process_and_inject_identity_heuristics(yolo_objects: Dict[int, YoloObjectTrack]):
    for yolo_id in yolo_objects:
        yolo_object_track = yolo_objects[yolo_id]

        last_mar = MarAtIndex(frame_index=0, mar=0.0)

        for index, detection in enumerate(yolo_object_track.detections):
            face_embeddings = []
            face = detection.face
            # print(face)
            if face:
                mar = face.mar
                if mar:
                    mar_derivative = (mar - last_mar.mar) / (
                        max(1, index - last_mar.frame_index)
                    )
                    face.mar_derivative = mar_derivative
                    last_mar = MarAtIndex(frame_index=index, mar=mar)
                    print("set")

                if face.embedding is not None:
                    if yolo_object_track.face_embeddings is None:
                        yolo_object_track.face_embeddings = []
                    face_embeddings.append(face.embedding)
            if len(face_embeddings) > 0:

                face_embeddings = np.array(face_embeddings)
                yolo_object_track.face_embeddings = select_diverse_fast(
                    face_embeddings,
                    k=5,
                    coreset=200,
                    simhash_bits=16,
                )


def select_diverse_fast(
    E: np.ndarray,
    k: int,
    *,
    coreset: int = 0,  # e.g., 2000 or 20*k to speed up on huge N
    simhash_bits: int = 0,  # e.g., 16 or 24 to collapse near-dupes; 0 = off
    rng: np.random.Generator | None = None,
):
    """
    Returns:
      dict(indices=[...], assign: (N,), assign_dist: (N,), coverage: float)
      indices are into ORIGINAL E.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = E.shape[0]
    if N == 0 or k <= 0:
        return {
            "indices": [],
            "assign": np.array([]),
            "assign_dist": np.array([]),
            "coverage": 0.0,
        }

    # 1) L2-normalize (cosine-ready)
    E = E.astype(np.float32, copy=False)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    # 2) Optional SimHash dedup to avoid near-identicals
    if simhash_bits and simhash_bits > 0:
        R = rng.standard_normal((simhash_bits, E.shape[1])).astype(np.float32)
        sig = (E @ R.T) > 0  # (N, B) booleans
        # pack bits to bytes for hashing
        packed = np.packbits(sig, axis=1)
        # unique rows
        _, unique_idx = np.unique(
            packed.view([("", packed.dtype)] * packed.shape[1]), return_index=True
        )
        unique_idx.sort()
        E_small = E[unique_idx]
        map_back = unique_idx
    else:
        E_small = E
        map_back = np.arange(N, dtype=int)

    M = E_small.shape[0]
    if M == 0:
        return {
            "indices": [],
            "assign": np.array([]),
            "assign_dist": np.array([]),
            "coverage": 0.0,
        }

    # 3) Optional coreset to bound M
    if coreset and M > coreset:
        sub = rng.choice(M, size=coreset, replace=False)
        sub.sort()
        E_core = E_small[sub]
        map_back = map_back[sub]
    else:
        E_core = E_small

    M = E_core.shape[0]
    k = min(k, M)

    # 4) k-means++ seeding (cosine distance = 1 - dot)
    # seed with point closest to mean direction
    mean = E_core.mean(axis=0)
    mean /= np.linalg.norm(mean) + 1e-9
    first = int(np.argmax(E_core @ mean))
    chosen = [first]

    # min distance to current chosen set
    min_dist = 1.0 - (E_core @ E_core[first])  # (M,)
    for _ in range(1, k):
        j = int(np.argmax(min_dist))
        chosen.append(j)
        # update single pass
        d_new = 1.0 - (E_core @ E_core[j])
        np.minimum(min_dist, d_new, out=min_dist)

    # 5) Assign all originals to nearest chosen (compute on core, then broadcast)
    C = np.stack([E_core @ E_core[c] for c in chosen], axis=1)  # (M, k) cosine sims
    assign_core = np.argmax(C, axis=1)
    assign_dist_core = 1.0 - C[np.arange(M), assign_core]
    coverage = float(assign_dist_core.max())

    # map chosen back to original indices
    chosen_orig = map_back[np.array(chosen, dtype=int)]

    # assign FULL N via nearest chosen reps (one batched matmul)
    reps = E[chosen_orig]  # (k, D)
    sims_full = E @ reps.T  # (N, k)
    assign_full = np.argmax(sims_full, axis=1)
    assign_dist_full = 1.0 - sims_full[np.arange(N), assign_full]

    return chosen_orig.tolist()
