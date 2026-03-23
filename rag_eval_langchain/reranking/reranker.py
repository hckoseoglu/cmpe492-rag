from langsmith import traceable
from sentence_transformers import CrossEncoder

_model = None


def load_reranker(model_name: str = "BAAI/bge-reranker-base") -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(model_name, device="cpu")
    return _model


@traceable(name="rerank_docs")
def rerank_docs(query: str, docs: list, top_k: int = 10, model: CrossEncoder = None) -> list:
    if len(docs) <= top_k:
        return docs

    if model is None:
        model = load_reranker()

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
