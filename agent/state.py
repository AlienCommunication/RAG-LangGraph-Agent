from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RAGState:
    query: str
    retrieved_docs: List[str] = None
    ranked_docs: List[str] = None
    answer: Optional[str] = None
    confidence: float = 0.0
    retries: int = 0
    max_retries: int = 2
