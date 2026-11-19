import pytest

from app.llm.openrouter import LLMResponseError, OpenRouterClient


def payload(content, finish_reason="stop"):
    return {
        "model": "test/model",
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_extracts_plain_string_content():
    assert OpenRouterClient._extract_text(payload("  Hello world  ")) == "Hello world"


def test_extracts_structured_text_blocks():
    response = payload([
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "world"},
        {"type": "image", "image_url": "ignored"},
    ])
    assert OpenRouterClient._extract_text(response) == "Hello\nworld"


def test_extracts_provider_output_text_blocks():
    assert OpenRouterClient._extract_text(
        payload([{"type": "output_text", "text": "Answer"}])
    ) == "Answer"


def test_removes_think_block():
    assert OpenRouterClient._extract_text(
        payload("<think>private reasoning</think>Final answer")
    ) == "Final answer"


@pytest.mark.parametrize("content", [None, "", "   ", [], [{"type": "image"}], 42])
def test_rejects_empty_or_non_text_content(content):
    with pytest.raises(LLMResponseError):
        OpenRouterClient._extract_text(payload(content))


def test_rejects_truncated_output():
    with pytest.raises(LLMResponseError, match="truncated"):
        OpenRouterClient._extract_text(payload("Partial answer", finish_reason="length"))


def test_rejects_refusal_payload():
    response = payload("")
    response["choices"][0]["message"]["refusal"] = "I cannot help with that request."
    with pytest.raises(LLMResponseError, match="refused"):
        OpenRouterClient._extract_text(response)


def test_rejects_unexpected_response_shape():
    with pytest.raises(LLMResponseError, match="unexpected"):
        OpenRouterClient._extract_text({"choices": []})
