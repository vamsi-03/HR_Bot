import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import google.generativeai as genai
import ollama
from ollama import Client as OllamaClient

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str = ""


class BaseLLM:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def available(self) -> ProviderStatus:  # pragma: no cover - interface
        raise NotImplementedError

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - interface
        yield self.generate(prompt)


class OllamaLLM(BaseLLM):
    def __init__(self) -> None:
        self.client = OllamaClient(host=settings.ollama_host)
        self.model = settings.ollama_model

    def generate(self, prompt: str) -> str:
        logger.info("Using Ollama model %s", self.model)
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def stream(self, prompt: str) -> Iterator[str]:
        logger.info("Streaming with Ollama model %s", self.model)
        stream = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def available(self) -> ProviderStatus:
        try:
            _ = self.client.list()
            return ProviderStatus("ollama", True, "reachable")
        except Exception as exc:  # pylint: disable=broad-except
            return ProviderStatus("ollama", False, str(exc))


class GeminiLLM(BaseLLM):
    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini API key not configured")
        logger.info("Using Gemini model %s", self.model)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    def stream(self, prompt: str) -> Iterator[str]:
        if not self.api_key:
            raise RuntimeError("Gemini API key not configured")
        logger.info("Streaming with Gemini model %s", self.model)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def available(self) -> ProviderStatus:
        if not self.api_key:
            return ProviderStatus("gemini", False, "API key missing")
        try:
            model = genai.GenerativeModel(self.model)
            _ = model.count_tokens("ping")
            return ProviderStatus("gemini", True, "reachable")
        except Exception as exc:  # pylint: disable=broad-except
            return ProviderStatus("gemini", False, str(exc))


class LLMRouter:
    def __init__(self) -> None:
        self.providers: List[BaseLLM] = [GeminiLLM(),OllamaLLM()]

    def provider_statuses(self) -> List[ProviderStatus]:
        return [provider.available() for provider in self.providers]

    def generate(self, prompt: str) -> str:
        last_error: Optional[str] = None
        for provider in self.providers:
            status = provider.available()
            if not status.available:
                last_error = status.detail
                logger.warning("%s unavailable: %s", status.name, status.detail)
                continue
            try:
                return provider.generate(prompt)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                logger.exception("Provider %s failed", status.name)
        raise RuntimeError(f"No LLM providers available. Last error: {last_error}")

    def stream(self, prompt: str) -> Iterator[str]:
        def _generator() -> Iterator[str]:
            last_error: Optional[str] = None
            for provider in self.providers:
                status = provider.available()
                if not status.available:
                    last_error = status.detail
                    logger.warning("%s unavailable: %s", status.name, status.detail)
                    continue
                try:
                    # Prefer true streaming if available
                    if hasattr(provider, "stream"):
                        yield from provider.stream(prompt)  # type: ignore
                    else:
                        yield provider.generate(prompt)
                    return
                except Exception as exc:  # pylint: disable=broad-except
                    last_error = str(exc)
                    logger.exception("Provider %s failed", status.name)
            raise RuntimeError(f"No LLM providers available. Last error: {last_error}")

        return _generator()


def build_policy_prompt(question: str, contexts: List[str]) -> str:
    """Standard prompt for HR policy questions that require RAG grounding and citations."""
    if not contexts:
        return (
            "You are an HR assistant. There is no matching source content for this question. "
            "If it is a generic questions, like the user is chatting with the bot, provide human like replies. But when he is asking any questions. understand it and answer if they are relevant to HR policy"
            "Understand the question you got and if the question is not related to HR policies, and at the same time, not generic - say that I don't have context and ask them to give questions related to HR policy"
            "Do not make anything up and do not add citations.\n\n"
            f"User question: {question}"
        )
    context_block = "\n\n".join(
        [f"[Source {idx + 1}]\n{snippet}" for idx, snippet in enumerate(contexts)]
    )
    return (
        "You are an HR assistant. First, understand the user’s intent; if it is HR-policy related, answer ONLY using the provided sources. "
        "Keep it concise (2-4 short sentences), warm, and human, and explicitly cite sources. "
        "When summarizing policies (e.g., leave policies), list the specific types and rules found, and avoid unrelated benefits. "
        "If the answer is not contained in the sources, reply exactly with 'No information found.'\n\n"
        f"{context_block}\n\nUser question: {question}\nAnswer with citations like [Source X] and avoid extra sources."
    )


def build_conversational_prompt(question: str, intent: str) -> str:
    """Dedicated prompt for non-policy questions (chitchat/non_hr) to generate dynamic, friendly responses."""
    if intent == "chitchat":
        return (
            "You are an HR assistant. There is no matching source content for this question. "
            "If it is a generic questions, like the user is chatting with the bot, provide human like replies. But when he is asking any questions. understand it and answer if they are relevant to HR policy"
            "Understand the question you got and if the question is not related to HR policies, and at the same time, not generic - say that I don't have context and ask them to give questions related to HR policy"
            "Do not make anything up and do not add citations.\n\n"
            f"\n\nUser message: {question}"
        )
    if intent == "non_hr":
        return (
            "You are an HR assistant focused solely on policy. The user is asking a non-HR question (e.g., 'What is the capital of France?'). "
            "Generate a polite, 1-2 sentence response reminding them that your function is limited to HR policies (leave, benefits, conduct, etc.) "
            "and invite them to ask an HR-related question. Do not answer the user's question or use citations."
            f"\n\nUser message: {question}"
        )
    return "No information found." # Should not happen
