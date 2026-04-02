import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from agent.state import AgentState
from agent.nodes import input_validator, response_formatter
from agent.graph import should_use_tools, after_tools, after_validation


def make_state(**overrides) -> AgentState:
    base = AgentState(
        messages=[], user_input="Hello", intent=None, confidence=None,
        next_node=None, iteration_count=0, retry_count=0,
        max_iterations=10, error=None, tool_results=[],
        final_response=None, metadata={},
    )
    base.update(overrides)
    return base


class TestInputValidator:
    def test_valid_input(self):
        state = make_state(user_input="  What is 2+2?  ")
        result = input_validator(state)
        assert result["user_input"] == "What is 2+2?"
        assert result["iteration_count"] == 0
        assert result["error"] is None

    def test_empty_input(self):
        state = make_state(user_input="   ")
        result = input_validator(state)
        assert result["error"] == "Empty input"

    def test_long_input_truncated(self):
        state = make_state(user_input="x" * 15_000)
        result = input_validator(state)
        assert len(result["user_input"]) == 10_000


class TestConditionalEdges:
    def test_routes_to_tools_when_tool_calls_present(self):
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [{"name": "calculator", "args": {}, "id": "1"}]
        state = make_state(messages=[ai_msg])
        assert should_use_tools(state) == "tool_executor"

    def test_routes_to_formatter_when_no_tool_calls(self):
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = []
        state = make_state(messages=[ai_msg])
        assert should_use_tools(state) == "response_formatter"

    def test_error_state_routes_to_formatter(self):
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [{"name": "x", "args": {}, "id": "1"}]
        state = make_state(messages=[ai_msg], error="something broke")
        assert should_use_tools(state) == "response_formatter"

    def test_after_tools_loops_to_llm(self):
        state = make_state(error=None)
        assert after_tools(state) == "llm_reasoner"

    def test_after_tools_error_goes_to_handler(self):
        state = make_state(error="tool failed")
        assert after_tools(state) == "error_handler"


class TestResponseFormatter:
    def test_extracts_content_from_last_ai_message(self):
        msg = AIMessage(content="The answer is 42.")
        state = make_state(messages=[msg])
        result = response_formatter(state)
        assert result["final_response"] == "The answer is 42."

    def test_fallback_when_no_ai_message(self):
        state = make_state(messages=[])
        result = response_formatter(state)
        assert result["final_response"] == "No response generated."


class TestLLMFactory:
    def test_factory_raises_on_unknown_provider(self):
        from llm.factory import LLMFactory
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            LLMFactory.create("unknown_llm")

    @patch("langchain_openai.ChatOpenAI")
    def test_factory_creates_openai(self, mock_openai):
        from llm.factory import LLMFactory
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.OPENAI_MODEL = "gpt-4o-mini"
            mock_settings.return_value.OPENAI_API_KEY = "test-key"
            mock_settings.return_value.LLM_TEMPERATURE = 0.0
            mock_settings.return_value.LLM_MAX_TOKENS = 2048
            mock_settings.return_value.LLM_TIMEOUT = 60
            LLMFactory.create("openai")
            mock_openai.assert_called_once()