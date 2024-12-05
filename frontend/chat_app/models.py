from pydantic import BaseModel, Field


class KnowledgeBase(BaseModel):
    name: str = Field(
        default="unknown", description="Name of the user, first name and last name"
    )
    age: str = Field(default="unknown", description="Age of the user")
    gender: str = Field(default="unknown", description="Gender of the user")
    query: str = Field(
        default="unknown",
        description="Detailed User's Query to answer. It should be framed in Question format.",
    )
    summary: str = Field(
        "unknown",
        description="Running summary of conversation. Update the summary by incorporating the new input by retaining the most useful information.",
    )
    response: str = Field(
        "unknown",
        description="An ideal response to the user based on their new message",
    )


class GradeDocuments(BaseModel):
    """Score for relevance check on retrieved documents."""
    relevancy_score: float = Field(
        description="Your score of the relevance of the document to the question on a scale of 0 to 1"
    )
