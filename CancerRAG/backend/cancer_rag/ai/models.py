from pydantic import BaseModel, Field


class KnowledgeBase(BaseModel):
    name: str = Field(
        default="unknown", description="Name of the user, first name and last name"
    )
    age: str = Field(default="unknown", description="Age of the user")
    gender: str = Field(default="unknown", description="Gender of the user")
    education_level : str = Field(default='12th grade', description="Highest education level of the user.")
    disease_site : str = Field(default='unknown', description="Description of location of Cancer. If specified as general, then it is general cancer")
    query: str = Field(
        default="unknown",
        description="Frame user input into a question format using disease_site and current summary",
    )
    summary: str = Field(
        "unknown",
        description="Running summary of conversation upto 500-600 words. Create a summary of the conversion using previous chatbot responses and user inputs",
    )


class GradeDocuments(BaseModel):
    """Score for relevance check on retrieved documents."""
    relevancy_score: float = Field(
        description="Your score of the relevance of the document to the question on a scale of 0 to 1"
    )
