import logging
from langchain_community.chat_models import ChatOpenAI

class LLMManager:
    def __init__(self, api_key, model_name, temperature, max_tokens, top_p, freq_penalty, pres_penalty):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty
        )
        logging.info(f"LLM initialized: {model_name}")

    async def invoke(self, prompt):
        # Async stub for future streaming support
        return self.llm.invoke(prompt)
