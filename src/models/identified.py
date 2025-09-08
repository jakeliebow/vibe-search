import numpy as np

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class IdentityType(str, Enum):
    """Enum for asset types"""
    PERSON = "person"
    OBJECT = "object"

class Identity(BaseModel):
    id: str = Field(..., description="resolved identity id")
    type:IdentityType = Field(..., description="type of identity, e.g. person")

