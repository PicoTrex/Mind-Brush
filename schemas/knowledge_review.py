from pydantic import BaseModel, Field
from typing import List

class KnowledgeReview_Result(BaseModel):
    final_prompt: str = Field(..., description="The final prompt after knowledge review.")
    reference_image: List[str] = Field(
        default_factory=list,
        description="List of image paths."
    )