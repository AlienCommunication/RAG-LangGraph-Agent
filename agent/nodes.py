from agent.state import RAGState

def retrieve_node(state: RAGState, retriever):
    state.retrieved_docs = retriever.search(state.query)
    return state

def rerank_node(state: RAGState, reranker):
    state.ranked_docs = reranker.rerank(state.query, state.retrieved_docs)
    return state

def generate_node(state: RAGState, llm):
    prompt = f"""
    Context:
    {state.ranked_docs}

    Question:
    {state.query}

    Answer strictly from context.
    """
    state.answer = llm.invoke(prompt).content
    return state

def judge_node(state: RAGState, judge_llm):
    prompt = f"""
    Context: {state.ranked_docs}
    Answer: {state.answer}

    Is the answer fully supported? Return a confidence score 0-1.
    """
    state.confidence = float(judge_llm.invoke(prompt).content.strip())
    return state
