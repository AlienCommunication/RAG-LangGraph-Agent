from langgraph.graph import StateGraph, END
from agent.state import RAGState
from agent.nodes import retrieve_node, rerank_node, generate_node, judge_node
from langchain_chroma import Chroma

def build_graph(retriever, reranker, llm, judge_llm):
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("rerank", lambda s: rerank_node(s, reranker))
    graph.add_node("generate", lambda s: generate_node(s, llm))
    graph.add_node("judge", lambda s: judge_node(s, judge_llm))

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "judge")
    graph.add_conditional_edges(
        "judge",
        lambda s: "retry" if s.confidence < 0.7 and s.retries < s.max_retries else "end",
        {
            "retry": "retrieve",
            "end": END
        }
    )

    return graph.compile()

