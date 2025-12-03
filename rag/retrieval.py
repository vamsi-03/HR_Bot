from typing import Dict, Iterator, List, Tuple

from llm.client import LLMRouter, build_policy_prompt, build_conversational_prompt # Updated imports
from rag.vector_store import VectorStore

SCORE_THRESHOLD = 0.55


def retrieve(question: str, store: VectorStore, top_k: int = 3) -> List[Tuple[Dict, float]]:
    return store.search(question, top_k=top_k)


def answer_question(question: str, store: VectorStore, llm: LLMRouter) -> Dict:
    intent = classify_intent(question, llm)
    
    # 1. Handle Conversational Intents (Let model generate response)
    if intent in ("chitchat", "non_hr"):
        prompt = build_conversational_prompt(question, intent)
        answer = llm.generate(prompt)
        return {
            "answer": answer,
            "citations": [],
            "grounded": False,
        }
        
    # 2. Handle HR Policy (RAG)
    hits = retrieve(question, store)
    filtered_hits = [(meta, score) for meta, score in hits if score >= SCORE_THRESHOLD]
    contexts = [meta["text"] for meta, _ in filtered_hits]

    # If nothing relevant, return explicit no-info
    if not contexts:
        return {"answer": "No information found.", "citations": [], "grounded": False}

    # Use the dedicated policy prompt
    prompt = build_policy_prompt(question, contexts)
    answer = llm.generate(prompt)

    citations = [
        {
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id"),
            "score": score,
            "text": meta.get("text"),
        }
        for meta, score in filtered_hits
    ]
    grounded = bool(contexts)
    return {"answer": answer, "citations": citations, "grounded": grounded}


def answer_question_stream(question: str, store: VectorStore, llm: LLMRouter) -> Tuple[Iterator[str], List[Dict], bool]:
    intent = classify_intent(question, llm)
    
    # 1. Handle Conversational Intents (Let model generate streamed response)
    if intent in ("chitchat", "non_hr"):
        prompt = build_conversational_prompt(question, intent)
        # Stream the dynamic conversational response
        return llm.stream(prompt), [], False
        
    # 2. Handle HR Policy (RAG)
    hits = retrieve(question, store)
    filtered_hits = [(meta, score) for meta, score in hits if score >= SCORE_THRESHOLD]
    contexts = [meta["text"] for meta, _ in filtered_hits]
    
    citations = [
        {
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id"),
            "score": score,
            "text": meta.get("text"),
        }
        for meta, score in filtered_hits
    ]
    grounded = bool(contexts)
    
    # If policy question but no context, send a hard no-info response
    if not contexts:
        def _no_context() -> Iterator[str]:
            yield "No information found."
        return _no_context(), [], False  # Citations empty when no context
    
    # Use the dedicated policy prompt for grounded answers
    prompt = build_policy_prompt(question, contexts)
    return llm.stream(prompt), citations, grounded


def is_hr_query(question: str) -> bool:
    # Legacy fallback when classifier fails
    return True


def classify_intent(question: str, llm: LLMRouter) -> str:
    prompt = (
        "Classify the user message into one of: hr_policy, non_hr, chitchat.\n"
        "hr_policy: HR policies/procedures (leave, PTO, benefits, conduct, onboarding, payroll, attendance, dress code, harassment, travel/expenses, parental/bereavement/sick leave).\n"
        "chitchat: greetings/small talk (hi, how are you, what's up) without HR content.\n"
        "non_hr: anything else not HR-related.\n\n"
        f"Message: {question}\n\n"
        "Respond with only one label: hr_policy, non_hr, or chitchat."
    )
    try:
        label = llm.generate(prompt).strip().lower()
        if "chitchat" in label:
            return "chitchat"
        if "non_hr" in label or "non-hr" in label:
            return "non_hr"
        return "hr_policy"
    except Exception:
        return "hr_policy"
