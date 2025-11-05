"""
Run the full algorithm that evolves a society of agents with different prompts.
"""

import asyncio
import random
from typing import Any, Dict, List

import instructor
import pandas as pd
import verifiers as vf
from openai import AsyncOpenAI
from pyprojroot import here
from verifiers import Environment

from src.agent import Agent
from src.constants import MUTATION_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from src.society import Society, SocietySchema
from src.utils import DELIMETER, batch_chat_completions, process_batch


class SocietyOfPromptsEvolution:
    def __init__(self, env: Environment, model_name: str, client: AsyncOpenAI):
        self.env = env
        self.dataset = env.get_eval_dataset()
        self.rubric = env.rubric
        self.model_name = model_name
        self.client = client
        self.instructor_client = instructor.from_openai(self.client)
        self.past_societies = []
        self.batch_size = 10
        self.log = []

    async def step(self, batch: List[Dict[str, Any]], batch_idx: int):
        step_log = {}
        # process the batch
        batch_results = await process_batch(self.society, batch, self.rubric, batch_idx)
        step_log["batch_results"] = batch_results
        mean_reward = sum(result["reward"] for result in batch_results) / len(
            batch_results
        )
        step_log["mean_reward"] = mean_reward

        # reflect on the results
        reflections = await self.reflect(batch_results)
        step_log["reflections"] = reflections

        # add the reflections to the batch results
        for result, reflection in zip(batch_results, reflections):
            result["reflection"] = reflection

        new_society_schema = await self.mutate(batch_results)
        step_log["new_society_schema"] = new_society_schema.model_dump_json()

        print(f"got mutation: {new_society_schema.model_dump_json()}")
        self.past_societies.append((self.society.save(), mean_reward))
        self.society = Society(client=self.society.client)
        self.society.load(new_society_schema)
        print("updated society")
        return step_log

    async def run(self, initial_society: Society, max_iterations: int = 10):
        """
        Run the full algorithm that evolves a society of agents with different prompts.
        """
        self.society = initial_society

        # total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        # One batch for now
        batch = list(self.dataset.select(range(self.batch_size)))

        for i in range(max_iterations):
            step_log = await self.step(batch, i)
            self.log.append(step_log)

            if i >= max_iterations:
                break

        return pd.DataFrame(self.log)

    async def reflect(self, batch: List[Dict[str, Any]]) -> str:
        """Given a batch of results, reflect on them and give reasons for success or failure."""
        system_prompt = {
            "role": "system",
            "content": REFLECTION_SYSTEM_PROMPT,
        }
        all_prompts = []
        for result in batch:
            correct = "Yes" if result["answer"] == result["rollout"] else "No"
            all_prompts.append(
                [
                    system_prompt,
                    {
                        "role": "user",
                        "content": f"Current society: {self.society.save().model_dump_json()}\nProblem: {result['prompt']}\nAttempt: {result['rollout']}\nAnswer correct: {correct}",
                    },
                ]
            )

        all_responses = await batch_chat_completions(
            client=self.client,
            messages_list=all_prompts,
            model=self.society.model_name,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
        )

        print(f"example reflection: {all_responses[0]}")

        return all_responses

    async def mutate(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Propose a mutation to the structure of the society, which could involve changing an agent's specialty, adding a new agent, or removing an agent.
        """
        system_prompt = [
            {
                "role": "system",
                "content": MUTATION_SYSTEM_PROMPT,
            }
        ]

        problems_and_reflections = DELIMETER.join(
            f"Problem: {x['prompt']}\nCorrect: {'Yes' if x['answer'] == x['rollout'] else 'No'}\nReflection: {x['reflection']}"
            for x in examples
        )
        mean_reward = sum(x["reward"] for x in examples) / len(examples)

        if len(self.past_societies) > 0:
            sample_societies = random.sample(
                self.past_societies, k=min(3, len(self.past_societies))
            )
            sample_societies_str = DELIMETER.join(
                f"Society: {x[0].model_dump_json()}\nMean reward: {x[1]}"
                for x in sample_societies
            )
            mutation_prompt = system_prompt + [
                {
                    "role": "user",
                    "content": f"Current society: {self.society.save().model_dump_json()}\nMean reward: {mean_reward}\n\nPast societies and their mean rewards:\n{sample_societies_str}\n\nExample problems and reflections:\n{problems_and_reflections}",
                }
            ]
        else:
            mutation_prompt = system_prompt + [
                {
                    "role": "user",
                    "content": f"Current society: {self.society.save().model_dump_json()}\nMean reward: {mean_reward}\n\nExample problems and reflections:\n{problems_and_reflections}",
                }
            ]

        new_society = await self.instructor_client.chat.completions.create(
            messages=mutation_prompt,
            response_model=SocietySchema,
            model=self.model_name,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
        )

        return new_society


if __name__ == "__main__":
    env = vf.load_environment("livecodebench", use_think=True)

    agent = Agent(
        name="Lone Genius",
        model_name="Qwen/Qwen3-8B",
        description="A lone genius thinks really hard and never delegates anything.",
        solution_prompt="You solve problems by thinking really hard about them.",
        handoff_prompt="You never hand anything off to anyone else.",
        neighbors=[],
        client=AsyncOpenAI(base_url="http://localhost:8000/v1"),
    )

    society = Society(
        agents=[agent],
        model_name="Qwen/Qwen3-8B",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1"),
    )

    algorithm = SocietyOfPromptsEvolution(
        env=env,
        model_name="Qwen/Qwen3-8B",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1"),
    )

    log = asyncio.run(algorithm.run(society, max_iterations=10))
    log.to_csv(here("data/log.csv"), index=False)
