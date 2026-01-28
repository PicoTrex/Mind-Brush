from pydantic import BaseModel, Field
from typing import List

class TextRAGInjection_Result(BaseModel):
    prompt: str = Field(..., description="The final, fact-enriched prompt with specific proper nouns.")
    final_image_queries: List[str] = Field(
        default_factory=list,
        description="The final image queries generated from the text."
    )