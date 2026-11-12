"""LLM proxy settings and ready-made clients. All LLM traffic goes through LiteLLM."""
import os

DEFAULT_PROXY_URL = "http://litellm.experiments.svc.cluster.local:4000"
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
MAX_TOKENS = 4096


def proxy_url() -> str:
    return os.environ.get("LLM_PROXY_URL", DEFAULT_PROXY_URL).rstrip("/")


def proxy_key() -> str:
    key = os.environ.get("LLM_PROXY_KEY")
    if not key:
        raise RuntimeError("LLM_PROXY_KEY is not set (load the litellm-client secret)")
    return key


def openai_base_url() -> str:
    """OpenAI-format base URL (chat completions)."""
    return f"{proxy_url()}/v1"


def anthropic_base_url() -> str:
    """Anthropic-format base URL (/v1/messages is appended by the SDK)."""
    return proxy_url()


def export_provider_env() -> None:
    """Set the env vars that frameworks read implicitly, all pointing at the proxy."""
    key = proxy_key()
    # anthropic>=2.x traces itself; frameworks already emit the LLM span (tracing.py also filters this scope).
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")
    os.environ.setdefault("ANTHROPIC_API_KEY", key)
    os.environ.setdefault("ANTHROPIC_BASE_URL", anthropic_base_url())
    os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_BASE_URL", openai_base_url())
    os.environ.setdefault("OPENAI_API_BASE", openai_base_url())


def anthropic_client(**kw):
    import anthropic

    return anthropic.Anthropic(base_url=anthropic_base_url(), api_key=proxy_key(), max_retries=4, **kw)


def async_anthropic_client(**kw):
    import anthropic

    return anthropic.AsyncAnthropic(base_url=anthropic_base_url(), api_key=proxy_key(), max_retries=4, **kw)


def openai_client(**kw):
    import openai

    return openai.OpenAI(base_url=openai_base_url(), api_key=proxy_key(), max_retries=4, **kw)


def async_openai_client(**kw):
    import openai

    return openai.AsyncOpenAI(base_url=openai_base_url(), api_key=proxy_key(), max_retries=4, **kw)


def chat_anthropic(model: str = HAIKU, **kw):
    """LangChain ChatAnthropic bound to the proxy. Never pass temperature."""
    from langchain_anthropic import ChatAnthropic

    kw.setdefault("max_tokens", MAX_TOKENS)
    kw.setdefault("max_retries", 4)
    return ChatAnthropic(model=model, base_url=anthropic_base_url(), api_key=proxy_key(), **kw)


def litellm_model_name(model: str = HAIKU) -> str:
    """Model string for LiteLLM-SDK based frameworks (ADK LiteLlm, smolagents, CrewAI).

    Routes through the proxy's OpenAI-format endpoint; pair with api_base=openai_base_url().
    """
    return f"openai/{model}"
