import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent.graph import build_graph
from agent.state import RAGState
from retrievers.vector import VectorRetriever
from retrievers.keyword import KeywordRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.reranker import ReRanker
from evaluation.metrics import recall_at_k

load_dotenv()

VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

class RAGEvaluator:
    """
    1. Retrieval quality
    2. Faithfulness (groundedness)
    3. Answer relevance
    """

    def __init__(self, corpus_docs: List[str]):
        self.answer_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.judge_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        vector = VectorRetriever(VECTOR_DB_DIR, EMBEDDING_MODEL)
        keyword = KeywordRetriever(corpus_docs)
        hybrid = HybridRetriever(keyword, vector)
        reranker = ReRanker()

        self.agent = build_graph(
            retriever=hybrid,
            reranker=reranker,
            llm=self.answer_llm,
            judge_llm=self.judge_llm
        )

    # LLM JUDGE HELPER

    def _judge_binary(self, prompt: str, inputs: Dict) -> int:
        chain = (
            ChatPromptTemplate.from_template(prompt)
            | self.judge_llm
            | StrOutputParser()
        )
        result = chain.invoke(inputs).strip().lower()
        return 1 if "pass" in result or "yes" in result else 0

    # METRIC 1: CONTEXT / RETRIEVAL RELEVANCE

    def eval_context_relevance(self, query: str, context: str) -> int:
        prompt = """
        You are a retrieval relevance judge.

        Query:
        {query}

        Retrieved Context:
        {context}

        Question:
        Does the retrieved context contain information useful to answer the query?

        Answer PASS or FAIL.
        """
        return self._judge_binary(prompt, {
            "query": query,
            "context": context
        })

    # METRIC 2: FAITHFULNESS (NO HALLUCINATION)

    def eval_faithfulness(self, context: str, answer: str) -> int:
        prompt = """
        You are a strict fact checker.

        Context:
        {context}

        Answer:
        {answer}

        Question:
        Is every factual claim in the answer fully supported by the context?

        Answer PASS or FAIL.
        """
        return self._judge_binary(prompt, {
            "context": context,
            "answer": answer
        })

    # METRIC 3: ANSWER RELEVANCE

    def eval_answer_relevance(self, query: str, answer: str) -> int:
        prompt = """
        You are an answer quality evaluator.

        User Query:
        {query}

        Answer:
        {answer}

        Question:
        Does the answer directly address the user's question?

        Answer PASS or FAIL.
        """
        return self._judge_binary(prompt, {
            "query": query,
            "answer": answer
        })

    # MAIN EVALUATION LOOP

    def run(self, test_set: List[Dict]) -> pd.DataFrame:
        """
        test_set format:
        [
          {
            "question": "...",
            "ground_truth_docs": ["expected chunk text", ...]
          }
        ]
        """

        results = []

        print(f" Running evaluation on {len(test_set)} test cases\n")

        for idx, item in enumerate(test_set, start=1):
            query = item["question"]
            gt_docs = item.get("ground_truth_docs", [])

            print(f"[{idx}] Question: {query}")

            # --- Run agent ---
            state = RAGState(query=query)
            final_state = self.agent.invoke(state)
            retrieved_docs = final_state.get("ranked_docs", [])
            answer = final_state.get("answer", "")
            confidence = final_state.get("confidence", 0.0)

            context_text = "\n".join(retrieved_docs)

            # --- Metrics ---
            retrieval_recall = (
                recall_at_k(retrieved_docs, gt_docs)
                if gt_docs else None
            )

            context_score = self.eval_context_relevance(query, context_text)
            faithfulness_score = self.eval_faithfulness(context_text, answer)
            answer_score = self.eval_answer_relevance(query, answer)

            results.append({
                "Question": query,
                "Answer": answer,
                "Confidence": confidence,
                "Retrieval_Recall@K": retrieval_recall,
                "Context_Relevance": context_score,
                "Faithfulness": faithfulness_score,
                "Answer_Relevance": answer_score,
                "Total_Score": context_score + faithfulness_score + answer_score
            })

        df = pd.DataFrame(results)

        print("\n" + "=" * 60)
        print(" EVALUATION SUMMARY")
        print("=" * 60)
        print(df[[
            "Context_Relevance",
            "Faithfulness",
            "Answer_Relevance",
            "Confidence"
        ]].mean())

        return df

if __name__ == "__main__":
    # Example test set
    test_data = [
        {
            "question": "What is the voltage requirement?",
            "ground_truth_docs": ["voltage"]
        },
        {
            "question": "Does safety protocol require gloves?",
            "ground_truth_docs": ["safety", "gloves"]
        },
        {
            "question": "Who is the CEO of the company?",
            "ground_truth_docs": []
        }
    ]

    # Load raw corpus for keyword retriever
    with open("./data/technical_manual.txt", "r", encoding="utf-8") as f:
        corpus = [f.read()]

    evaluator = RAGEvaluator(corpus)
    df = evaluator.run(test_data)

    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/evaluation_results.csv", index=False)

    print(" Results saved to artifacts/evaluation_results.csv")
