import asyncio
import time
from typing import Any, Dict, List

from openai import AsyncOpenAI

from src.society import Society

DELIMETER = "\n---\n"


async def evaluate_single_example(
    society: Society,
    example: Dict[str, Any],
    rubric: Any,
    idx: int,
) -> Dict[str, Any]:
    """Evaluate a single example and return results."""
    prompt = example["prompt"]

    start_time = time.time()
    rollout = await society.solve(prompt)
    end_time = time.time()

    state = {"timing": {"total_ms": (end_time - start_time) * 1000}}
    reward = await rubric.score_rollout(
        prompt=prompt,
        completion=rollout,
        answer=example["answer"],
        state=state,
        info=example["info"],
    )

    return {
        "idx": idx,
        "prompt": prompt,
        "rollout": rollout,
        "answer": example["answer"],
        "reward": reward.reward,
        "time_ms": state["timing"]["total_ms"],
    }


async def process_batch(
    society: Society,
    examples: List[Dict[str, Any]],
    rubric: Any,
    start_idx: int,
) -> List[Dict[str, Any]]:
    """Process a batch of examples concurrently."""
    tasks = [
        evaluate_single_example(society, example, rubric, start_idx + i)
        for i, example in enumerate(examples)
    ]
    return await asyncio.gather(*tasks)


async def batch_chat_completions(
    client: AsyncOpenAI, messages_list: List[List[Dict[str, str]]], model: str, **kwargs
) -> List[str]:
    """
    Process a batch of chat completion requests concurrently.

    Args:
        client: The AsyncOpenAI client to use
        messages_list: A list of message lists (each inner list is a conversation)
        model: The model name to use
        **kwargs: Additional arguments to pass to chat.completions.create

    Returns:
        A list of completion strings, one for each input message list
    """

    async def get_completion(messages: List[Dict[str, str]]) -> str:
        response = await client.chat.completions.create(
            messages=messages, model=model, **kwargs
        )
        return response.choices[0].message.content

    tasks = [get_completion(messages) for messages in messages_list]
    return await asyncio.gather(*tasks)
