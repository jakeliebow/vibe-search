from pydantic import BaseModel, Field
from typing import Dict, Tuple, List


class Pair(BaseModel):
    score: float = Field(default_factory=lambda _: 0.0)
    total: int = Field(default_factory=lambda _: 0)


class HeuristicRelation(BaseModel):
    pairs: Dict[Tuple[str, str], Pair] = Field(default_factory=dict)
    items: List[float] = Field(default_factory=list)


class HeuristicRelations(BaseModel):
    mar_derivatives: HeuristicRelation = Field(default_factory=HeuristicRelation)
    speaker_transcription_relation: HeuristicRelation = Field(
        default_factory=HeuristicRelation
    )
