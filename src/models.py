"""Pydantic models for Federal Register API data."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RegulationDocument(BaseModel):
    """Schema for Federal Register regulation documents."""
    
    document_number: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    type: str = Field(..., description="Document type (e.g., Notice, Rule, Proposed Rule)")
    abstract: Optional[str] = Field(None, description="Document abstract or summary")
    html_url: str = Field(..., description="URL to HTML version of the document")
    publication_date: datetime = Field(..., description="Publication date and time")
    agency_names: Optional[str] = Field(None, description="Comma-separated list of agency names")
    
    @field_validator('document_number')
    @classmethod
    def validate_document_number(cls, v: str) -> str:
        """Ensure document_number is not empty."""
        if not v or not v.strip():
            raise ValueError("document_number cannot be empty")
        return v.strip()
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "document_number": "2024-01234",
                "title": "Artificial Intelligence Regulatory Framework",
                "type": "Notice",
                "abstract": "This notice proposes new regulations...",
                "html_url": "https://www.federalregister.gov/documents/2024/01/15/2024-01234",
                "publication_date": "2024-01-15T10:30:00Z"
            }
        }
