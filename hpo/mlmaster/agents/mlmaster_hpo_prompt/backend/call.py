import logging

from utils.llm_caller import LLM
from backend.backend_utils import PromptType, compile_prompt_to_md
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

def r1_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    elif isinstance(prompt, dict):
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    elif isinstance(prompt, list) and len(prompt) >= 2:
        logger.info(f"prompt: {prompt[0]['content']}\n{prompt[1]['content']}", extra={"verbose": True})
    else:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Convert prompt to appropriate format for LLM calls
    if cfg.agent.steerable_reasoning == True:
        # stream_complete expects a string prompt
        if isinstance(prompt, dict):
            prompt_str = compile_prompt_to_md(prompt)
        elif isinstance(prompt, str):
            prompt_str = prompt
        elif isinstance(prompt, list):
            # Convert messages list to string
            prompt_str = "\n".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict) and "content" in msg])
        else:
            prompt_str = str(prompt)
        
        response = llm.stream_complete(
            prompt_str,
            **model_kwargs
        )
        
    else:
        # stream_generate expects messages list
        if isinstance(prompt, dict):
            prompt_str = compile_prompt_to_md(prompt)
            messages = [{"role": "user", "content": prompt_str}]
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        
        response = llm.stream_generate(
            messages,
            **model_kwargs
        )


    if "</think>" in response:
        res = response[response.find("</think>")+8:]
    else:
        res = response

    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"response without think:\n{res}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res

def gpt_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    if type(prompt) == str:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    elif isinstance(prompt, dict):
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    elif isinstance(prompt, list) and len(prompt) >= 2:
        logger.info(f"prompt: {prompt[0]['content']}\n{prompt[1]['content']}", extra={"verbose": True})
    else:
        logger.info(f"prompt: {prompt}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Convert dict prompt to messages format if needed
    if isinstance(prompt, dict):
        prompt_str = compile_prompt_to_md(prompt)
        messages = [{"role": "user", "content": prompt_str}]
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    response = llm.stream_generate(
        messages,
        **model_kwargs
    )

    res = response
    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res