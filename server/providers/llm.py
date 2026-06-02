from aiavatar.sts.llm.chatgpt import ChatGPTService

from config import Settings


def create_llm(settings: Settings):
    if settings.llm_provider != "openclaw":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    llm = ChatGPTService(
        openai_api_key=settings.openclaw_token,
        base_url=settings.openclaw_base_url,
        model=settings.openclaw_model,
    )

    @llm.request_filter
    def request_filter(text: str):
        if text is not None:
            return settings.openclaw_request_prefix + text
        return text

    @llm.edit_chat_completion_params
    def edit_chat_completion_params(chat_completion_params: dict, context_id: str, user_id: str):
        user_message = chat_completion_params["messages"][-1]
        if not any(isinstance(c, str) or c.get("type") == "text" for c in user_message["content"]):
            user_message["content"].append({"type": "text", "text": settings.openclaw_request_prefix})

        chat_completion_params["messages"] = [user_message]
        chat_completion_params["extra_headers"] = {
            "x-openclaw-session-key": user_id
        }

    return llm
