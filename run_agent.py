import os
import argparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.graph import build_graph
from agent.state import RAGState
from retrievers.vector import VectorRetriever
from retrievers.keyword import KeywordRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.reranker import ReRanker

load_dotenv()

VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

def build_agent():
    """
    - Hybrid retrieval
    - Re-ranking
    - LangGraph agent
    """

    answer_llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    judge_llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    #Load corpus for keyword retriever
    with open("./data/technical_manual.txt", "r", encoding="utf-8") as f:
        corpus_docs = [f.read()]

    # Retrievers
    vector_retriever = VectorRetriever(
        persist_dir=VECTOR_DB_DIR,
        model_name=EMBEDDING_MODEL
    )

    keyword_retriever = KeywordRetriever(corpus_docs)

    hybrid_retriever = HybridRetriever(
        keyword=keyword_retriever,
        vector=vector_retriever,
        alpha=0.5
    )

    reranker = ReRanker()

    # Build LangGraph agent
    agent = build_graph(
        retriever=hybrid_retriever,
        reranker=reranker,
        llm=answer_llm,
        judge_llm=judge_llm
    )

    return agent

def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG System")
    parser.add_argument("query", help="User question")
    args = parser.parse_args()

    agent = build_agent()

    # Initialize agent state
    state = RAGState(query=args.query)

    # Run agent
    final_state = agent.invoke(state)

    answer = final_state.get("answer")
    confidence = final_state.get("confidence", 0.0)

    if answer:
        print(answer)
        print("\nConfidence:", round(confidence, 2))
    else:
        print("Insufficient Context.")


    print("=" * 60)


if __name__ == "__main__":
    main()
