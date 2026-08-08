from dataclasses import dataclass

import json
import jsonschema
from dataclasses_json import DataClassJsonMixin
import backoff
import logging
from typing import Callable

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


logger = logging.getLogger("aide")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
    max_tries=10,  # Limit retries to prevent infinite loops
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.warning(f"Backoff exception (will retry up to 10 times): {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif prompt is None:
        return "null\n"
    elif isinstance(prompt, (int, float, bool)):
        return f"{prompt}\n"
    elif isinstance(prompt, list):
        lines: list[str] = []
        for item in prompt:
            if item is None:
                lines.append("- null")
            elif isinstance(item, str):
                lines.append(f"- {item.strip()}")
            elif isinstance(item, (int, float, bool)):
                lines.append(f"- {item}")
            elif isinstance(item, (dict, list)):
                try:
                    blob = json.dumps(item, indent=2, ensure_ascii=False)
                except Exception:
                    blob = str(item)
                lines.extend(["- ```json", blob, "```"])
            else:
                lines.append(f"- {str(item).strip()}")
        return "\n".join(lines + ["\n"])
    elif not isinstance(prompt, dict):
        return str(prompt).strip() + "\n"

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }

    @property
    def as_openai_responses_tool_dict(self):
        """Convert to OpenAI Responses API tool format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_responses_tool_choice_dict(self):
        """Convert to OpenAI Responses API tool choice format."""
        return {
            "type": "function",
            "name": self.name,
        }
