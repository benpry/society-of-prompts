from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.agent import Agent, AgentSchema


class SocietySchema(BaseModel):
    agents: list[AgentSchema]
    initial_agent: str


class Society:
    """
    A society of agents.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        agents: Optional[list] = None,
        model_name: Optional[str] = None,
    ):
        self.client = client
        self.agents = agents
        self.initial_agent = agents[0] if agents else None
        self.model_name = model_name

    def __str__(self) -> str:
        res = f"Society of {len(self.agents)} agent{'' if len(self.agents) == 1 else 's'}."
        if self.initial_agent is not None:
            res += f" Initial agent: {self.initial_agent.name}."

        res += "\nAgents:\n"
        for agent in self.agents:
            res += f"- {agent.name}: {agent.description}\n\tsolution prompt: {agent.solution_prompt}\n\thandoff prompt: {agent.handoff_prompt}\n\tneighbors: {agent.neighbors}\n"
        return res

    async def solve(self, problem: list) -> str:
        """Async solve method."""
        return await self.initial_agent.solve(problem)

    def load(self, schema: SocietySchema) -> None:
        """Load a society from a schema."""
        self.agents = []
        for agent_schema in schema.agents:
            agent = Agent(client=self.client, model_name=self.model_name)
            agent.load(agent_schema)
            self.agents.append(agent)

        self.initial_agent = next(
            agent for agent in self.agents if agent.name == schema.initial_agent
        )

    def save(self) -> SocietySchema:
        return SocietySchema(
            agents=[agent.save() for agent in self.agents],
            initial_agent=self.initial_agent.name,
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        client = AsyncOpenAI(base_url="http://localhost:8000/v1")

        agent = Agent(
            name="Lone Genius",
            model_name="gpt-4",
            description="A lone genius who thinks really hard and never delegates anything.",
            solution_prompt="You solve problems by thinking really hard about them.",
            handoff_prompt="You never hand anything off to anyone else.",
            neighbors=[],
            client=client,
        )

        society_of_one = Society(agents=[agent], client=client)

        result = await society_of_one.solve(
            [{"role": "user", "content": "What is the 100th prime number?"}]
        )
        print(result)

    asyncio.run(main())
