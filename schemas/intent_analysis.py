from pydantic import BaseModel, Field
from typing import List, Literal

class IntentAnalysis_Result(BaseModel):
    need_process_problem: List[str] = Field(
        default_factory=list, 
        description="The specific problems that need further processing."
    )
    intent_category: Literal[
        "Direct_Generation",
        "Reasoning_Generation",
        "Search_Generation",
        "Reasoning_Search_Generation",
        "Search_Reasoning_Generation"
    ] = Field(..., description="The category of intent for the analysis.")