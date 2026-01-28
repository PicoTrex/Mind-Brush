from pydantic import BaseModel, Field
from typing import List

class KnowledgeReasoning_Result(BaseModel):
    reasoning_knowledge: List[str] = Field(
        default_factory=list,
        description="The knowledge extracted by reasoning."
    )