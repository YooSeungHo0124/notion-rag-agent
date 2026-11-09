from notion_agent.llm.base import LLMClient, LLMError, LLMResponse


def get_llm() -> LLMClient:
    """Build the configured LLM client."""
    from notion_agent.config import get_settings

    s = get_settings()
    if s.llm_provider == "claude_cli":
        from notion_agent.llm.claude_cli import ClaudeCLIClient

        return ClaudeCLIClient(
            model=s.llm_model,
            timeout_s=s.file.llm.timeout_s,
            max_retries=s.file.llm.max_retries,
        )
    raise NotImplementedError(f"LLM provider {s.llm_provider!r} is not implemented yet")


__all__ = ["LLMClient", "LLMError", "LLMResponse", "get_llm"]
