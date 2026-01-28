from pydantic import BaseModel, Field
from typing import List, Literal

# Intent Analysis Result
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

# Keyword Generation Result
class KeywordGeneration_Result(BaseModel):
    text_queries: List[str] = Field(
        default_factory=list,
        description="The keyword of text queries."
    )
    image_queries: List[str] = Field(
        default_factory=list,
        description="The keyword of image queries."
    )

# Text RAG Injection Result
class TextRAGInjection_Result(BaseModel):
    prompt: str = Field(..., description="The final, fact-enriched prompt with specific proper nouns.")
    final_image_queries: List[str] = Field(
        default_factory=list,
        description="The final image queries generated from the text."
    )

# Knowledge Reasoning Result
class KnowledgeReasoning_Result(BaseModel):
    reasoning_knowledge: List[str] = Field(
        default_factory=list,
        description="The knowledge extracted by reasoning."
    )

# Knowledge Review Result
class KnowledgeReview_Result(BaseModel):
    final_prompt: str = Field(..., description="The final prompt after knowledge review.")
    reference_image: List[str] = Field(
        default_factory=list,
        description="List of image paths."
    )