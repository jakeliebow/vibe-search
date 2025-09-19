from src.models.frame import Frame
from typing import Dict, List, Tuple, Set
import numpy as np
from dataclasses import dataclass
from src.models.relate import HeuristicRelation, HeuristicRelations, Pair


def activation_function(value, median, std):
    return np.tanh((value - median) / std) if std > 0 else 0.0


@dataclass
class Edge:
    weight: float
    vertexes: Tuple[str, str]


def calculate_entity_relationships(
    frames: List[Frame],
    *,
    confidence_threshold: float = 0.01,
) -> List[Edge]:
    heuristic_relations = HeuristicRelations()
    for frame in frames:
        if not frame.diarized_audio_segments or not frame.detections:
            continue
        for det in frame.detections:
            if det.face is None or det.face.mar_derivative is None:
                continue
            w = abs(det.face.mar_derivative)
            heuristic_relations.mar_derivatives.items.append(w)

            for diar_seg in frame.diarized_audio_segments:
                object_voice_embedding_key_pair = (
                    diar_seg.speaker_label,
                    det.yolo_uuid,
                )
                if (
                    object_voice_embedding_key_pair
                    not in heuristic_relations.mar_derivatives.pairs
                ):
                    heuristic_relations.mar_derivatives.pairs[
                        object_voice_embedding_key_pair
                    ] = Pair()

                heuristic_relations.mar_derivatives.pairs[
                    object_voice_embedding_key_pair
                ].score += w
                heuristic_relations.mar_derivatives.pairs[
                    object_voice_embedding_key_pair
                ].total += 1
                for trans_seg in frame.transcribed_audio_segments:
                    voice_embedding_transcription_key_pair = (
                        trans_seg.uuid,
                        diar_seg.speaker_label,
                    )
                    heuristic_relations.speaker_transcription_relation.items.append(
                        trans_seg.probability
                    )

                    if (
                        voice_embedding_transcription_key_pair
                        not in heuristic_relations.speaker_transcription_relation.pairs
                    ):
                        heuristic_relations.speaker_transcription_relation.pairs[
                            voice_embedding_transcription_key_pair
                        ] = Pair()

                    heuristic_relations.speaker_transcription_relation.pairs[
                        voice_embedding_transcription_key_pair
                    ].score += trans_seg.probability
                    heuristic_relations.speaker_transcription_relation.pairs[
                        voice_embedding_transcription_key_pair
                    ].total += 1

    pairings: List[Edge] = []
    for heuristic_relation in [
        heuristic_relations.mar_derivatives,
        heuristic_relations.speaker_transcription_relation,
    ]:
        heuristic_relation.items.sort()
        std_deviation = (
            np.std(heuristic_relation.items) if heuristic_relation.items else 0.0
        )
        median_value = (
            heuristic_relation.items[len(heuristic_relation.items) // 2]
            if heuristic_relation.items
            else 0.0
        )
        for (v1, v2), pair in heuristic_relation.pairs.items():
            if pair.total == 0:
                continue
            average_mar_delta_during_cocurrence = pair.score / pair.total
            relation_weight = activation_function(
                average_mar_delta_during_cocurrence, median_value, std_deviation
            )
            if relation_weight >= confidence_threshold:
                pairings.append(Edge(relation_weight, (v1, v2)))

    return pairings
