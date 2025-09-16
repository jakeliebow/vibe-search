from typing import Optional, List, Set, Dict
from dataclasses import dataclass
from src.models.detection import YoloObjectTrack
from src.models.audio import SpeakerTrack
from src.relations.relate import Edge


@dataclass
class Identity:
    id: str
    yolo_track: Optional[YoloObjectTrack]
    speaker_track: Optional[SpeakerTrack]
    pairing_confidence: Optional[float]  # MAR derivative confidence if paired


def build_individual_identities(
    pairings: List[Edge],
    paired: Set[str],
    identified_objects: Dict[str, YoloObjectTrack],
    identified_speakers: Dict[str, SpeakerTrack],
) -> List[Identity]:
    """
    Build individual identities from paired and unpaired speakers and objects.

    Args:
        pairings: List of speaker-object pairings
        paired: Set of IDs that are already paired
        identified_objects: Dict of object tracks by ID
        identified_speakers: Dict of speaker tracks by ID

    Returns:
        List of individual Identity objects
    """
    identities: List[Identity] = []

    # Create identities for paired speakers and objects
    for pairing in pairings:
        identity_id = f"identity_{len(identities)}"
        yolo_track = identified_objects[pairing.object_id]
        speaker_track = identified_speakers[pairing.speaker_id]

        identities.append(
            Identity(
                id=identity_id,
                yolo_track=yolo_track,
                speaker_track=speaker_track,
                pairing_confidence=pairing.avg_abs_mar_derivative,
            )
        )

    # Create identities for unpaired speakers (speaker-only identities)
    for speaker_id, speaker_track in identified_speakers.items():
        if speaker_id not in paired:
            identity_id = f"identity_{len(identities)}"
            identities.append(
                Identity(
                    id=identity_id,
                    yolo_track=None,
                    speaker_track=speaker_track,
                    pairing_confidence=None,
                )
            )

    # Create identities for unpaired objects (object-only identities)
    for object_id, yolo_track in identified_objects.items():
        if object_id not in paired:
            identity_id = f"identity_{len(identities)}"
            identities.append(
                Identity(
                    id=identity_id,
                    yolo_track=yolo_track,
                    speaker_track=None,
                    pairing_confidence=None,
                )
            )

    return identities
