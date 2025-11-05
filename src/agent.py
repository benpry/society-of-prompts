from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.constants import AGENT_PROMPT_TEMPLATE

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "handoff",
            "description": "Hand off a problem to a neighbor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the agent to hand off the problem to.",
                    },
                    "problem": {
                        "type": "string",
                        "description": "The problem to hand off.",
                    },
                },
                "required": ["agent_name", "problem"],
            },
        },
    }
]


class AgentSchema(BaseModel):
    name: str
    description: str
    solution_prompt: str
    handoff_prompt: str
    neighbors: list[str]


class Agent:
    """
    An individual agent in the society.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        solution_prompt: Optional[str] = None,
        handoff_prompt: Optional[str] = None,
        neighbors: Optional[list] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.description = description
        self.solution_prompt = solution_prompt
        self.handoff_prompt = handoff_prompt
        self.neighbors = neighbors
        self.client = client
        self.prompt = AGENT_PROMPT_TEMPLATE.format(
            solution_prompt=self.solution_prompt,
            handoff_prompt=self.handoff_prompt,
            neighbors="\n".join(str(x) for x in self.neighbors)
            if self.neighbors
            else "None",
        )

    async def solve(self, problem: list) -> str:
        """
        Solve a problem with this agent.
        """

        responses = await self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.prompt},
                problem[-1],
            ],
            model=self.model_name,
            max_completion_tokens=4096,
            tools=TOOLS,
            temperature=0.6,
            top_p=0.95,
        )

        return responses.choices[0].message.content

    async def handoff(self, agent_name: str, problem: str) -> str:
        """
        Hand off a problem to an agent with a given name.
        """
        for neighbor in self.neighbors:
            if neighbor.name == agent_name:
                return await neighbor.solve(problem)
        raise ValueError(f"Agent {agent_name} not found in neighbors")

    def save(self) -> AgentSchema:
        return AgentSchema(
            name=self.name,
            description=self.description,
            solution_prompt=self.solution_prompt,
            handoff_prompt=self.handoff_prompt,
            neighbors=self.neighbors,
        )

    def load(self, schema: AgentSchema) -> None:
        self.name = schema.name
        self.description = schema.description
        self.solution_prompt = schema.solution_prompt
        self.handoff_prompt = schema.handoff_prompt
        self.neighbors = schema.neighbors

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
