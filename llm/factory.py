"""
One function. Any LLM. Zero changes to your agent code.
Just set LLM_PROVIDER in .env and you're done.
"""
import logging
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMFactory:

    @staticmethod
    def create(provider: str = None) -> BaseChatModel:
        """
        Factory method returning a LangChain-compatible chat model.
        Every provider returns the SAME BaseChatModel interface —
        your nodes call .invoke() and .stream() identically on all of them.
        """
        s = get_settings()
        provider = provider or s.LLM_PROVIDER
        logger.info(f"[LLMFactory] Creating LLM: provider={provider}")

        shared_kwargs = {
            "temperature": s.LLM_TEMPERATURE,
            "max_tokens":  s.LLM_MAX_TOKENS,
            "timeout":     s.LLM_TIMEOUT,
        }

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=s.OPENAI_MODEL,
                api_key=s.OPENAI_API_KEY,
                **shared_kwargs,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=s.ANTHROPIC_MODEL,
                api_key=s.ANTHROPIC_API_KEY,
                **shared_kwargs,
            )

        elif provider == "bedrock":
            from langchain_aws import ChatBedrockConverse
            return ChatBedrockConverse(
                model=s.BEDROCK_MODEL_ID,
                region_name=s.AWS_REGION,
                temperature=s.LLM_TEMPERATURE,
                max_tokens=s.LLM_MAX_TOKENS,
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=s.OLLAMA_MODEL,
                base_url=s.OLLAMA_BASE_URL,
                temperature=s.LLM_TEMPERATURE,
            )

        elif provider == "huggingface":
            if s.HUGGINGFACE_INFERENCE_URL:
                # Self-hosted TGI / vLLM endpoint
                from langchain_community.chat_models import ChatHuggingFace
                from langchain_community.llms import HuggingFaceTextGenInference
                llm = HuggingFaceTextGenInference(
                    inference_server_url=s.HUGGINGFACE_INFERENCE_URL,
                    max_new_tokens=s.LLM_MAX_TOKENS,
                    temperature=s.LLM_TEMPERATURE,
                )
                return ChatHuggingFace(llm=llm)
            else:
                # HuggingFace Inference API (hosted)
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                endpoint = HuggingFaceEndpoint(
                    repo_id=s.HUGGINGFACE_MODEL,
                    huggingfacehub_api_token=s.HUGGINGFACE_API_KEY,
                    max_new_tokens=s.LLM_MAX_TOKENS,
                    temperature=s.LLM_TEMPERATURE,
                )
                return ChatHuggingFace(llm=endpoint)

        elif provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=s.AZURE_OPENAI_DEPLOYMENT,
                api_version=s.AZURE_OPENAI_API_VERSION,
                azure_endpoint=s.AZURE_OPENAI_ENDPOINT,
                api_key=s.AZURE_OPENAI_API_KEY,
                **shared_kwargs,
            )
        # elif provider == "groq":
        #     from langchain_openai import ChatOpenAI
        #     return ChatOpenAI(
        #     model=s.GROQ_MODEL,
        #     api_key=s.GROQ_API_KEY,
        #     base_url=s.GROQ_BASE_URL,   # ← only difference from OpenAI
        #     **shared_kwargs,
        # )

        elif provider == "groq":
            # langchain-groq gives you slightly better error messages
            from langchain_groq import ChatGroq
            # These models generate XML tool calls instead of JSON — unusable with tools
            GROQ_TOOL_BROKEN_MODELS = {
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview", 
            }
            model = s.GROQ_MODEL

            if model in GROQ_TOOL_BROKEN_MODELS:
                fallback = "mixtral-8x7b-32768"
                print(f"[LLMFactory] WARNING: {model} has broken tool calling on Groq.")
                print(f"[LLMFactory] Auto-switching to: {fallback}")
                model = fallback

            return ChatGroq(
            model=s.GROQ_MODEL,
            api_key=s.GROQ_API_KEY,
            temperature=s.LLM_TEMPERATURE,
            max_tokens=s.LLM_MAX_TOKENS,
        )

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider}'. "
                f"Choose from: openai, anthropic, bedrock, ollama, huggingface, azure_openai"
            )


@lru_cache(maxsize=8)
def get_llm(provider: str = None) -> BaseChatModel:
    """Cached — one LLM instance per provider per process."""
    return LLMFactory.create(provider)