import numpy as np

from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4


class Embedding(BaseModel):
    """
    Represents a generic embedding vector.
    """

    embedding: np.ndarray = Field(..., description="The embedding vector as a list of floats.")
    uuid: str = Field(default_factory=lambda _: str(uuid4()), description="A unique identifier for the embedding.")
    image_data:np.ndarray = Field(..., description="2d nd array of pixel data")

    class Config:
        arbitrary_types_allowed = True