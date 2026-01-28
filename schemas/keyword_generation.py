from pydantic import BaseModel, Field
from typing import List

class KeywordGeneration_Result(BaseModel):
    text_queries: List[str] = Field(
        default_factory=list,
        description="The keyword of text queries."
    )
    image_queries: List[str] = Field(
        default_factory=list,
        description="The keyword of image queries."
    )