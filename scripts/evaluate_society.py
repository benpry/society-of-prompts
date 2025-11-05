"""
Evaluate the society of prompts on a given environment.
"""

import asyncio
import time
from argparse import ArgumentParser

import verifiers as vf
from openai import AsyncOpenAI

from src.agent import Agent
from src.society import Society
from src.utils import process_batch


async def main(args):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    env = vf.load_environment(args.env, use_think=True)
    dataset = env.get_eval_dataset()
    rubric = env.rubric

    # TODO: stop this from being hardcoded
    agent = Agent(
        name="Lone Genius",
        model_name=args.model,
        description="A lone genius thinks really hard and never delegates anything.",
        solution_prompt="You solve problems by thinking really hard about them.",
        handoff_prompt="You never hand anything off to anyone else.",
        neighbors=[],
        client=client,
    )

    society = Society(agents=[agent], model_name=args.model, client=client)

    # Select examples to evaluate
    examples = list(dataset.select(range(args.num_examples)))

    # Process examples in batches
    all_results = []
    total_batches = (len(examples) + args.batch_size - 1) // args.batch_size

    print(
        f"Processing {len(examples)} examples in {total_batches} batches of size {args.batch_size}"
    )
    print("=" * 80)

    for batch_idx in range(0, len(examples), args.batch_size):
        batch_num = batch_idx // args.batch_size + 1
        batch_examples = examples[batch_idx : batch_idx + args.batch_size]

        print(
            f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_examples)} examples)..."
        )
        batch_start = time.time()

        batch_results = await process_batch(society, batch_examples, rubric, batch_idx)

        batch_end = time.time()
        batch_time = batch_end - batch_start

        all_results.extend(batch_results)

        # Print batch summary
        batch_rewards = [r["reward"] for r in batch_results]
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        print(f"Batch {batch_num} completed in {batch_time:.2f}s")
        print(f"Batch {batch_num} average reward: {avg_reward:.3f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    # Print individual results
    for result in all_results:
        print(f"\n[Example {result['idx']}]")
        print(f"Prompt: {result['prompt']}")
        print(f"Rollout: {result['rollout']}")
        print(f"Answer: {result['answer']}")
        print(f"Reward: {result['reward']:.3f}")
        print(f"Time: {result['time_ms']:.0f}ms")

    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    rewards = [r["reward"] for r in all_results]
    times = [r["time_ms"] for r in all_results]

    print(f"Total examples: {len(all_results)}")
    print(f"Average reward: {sum(rewards) / len(rewards):.3f}")
    print(f"Min reward: {min(rewards):.3f}")
    print(f"Max reward: {max(rewards):.3f}")
    print(f"Average time: {sum(times) / len(times):.0f}ms")
    print(f"Total time: {sum(times):.0f}ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args))
