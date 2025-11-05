MUTATION_SYSTEM_PROMPT = """You will see a set of problems that a society of agents with different specialties tried to solve. Your job is to propose a new structure for the society which will help it solve problems more effectively. The new society should be represented in the same json format as the current society. You will also see a few examples of past societies and the mean rewards they achieved.
- Each agent should have a name and a description that describe what that agent's role is.
- The solution prompt should give the agent a strategy for solving problems.
- The handoff prompt should tell the agent how and when it should share problems or subproblems with other agents.
- An agent's neighbors are the agents that it can hand off problems to.
- The initial agent is the agent that will see the problem to begin with. They can delegate the problem to other agents via tool calls.
"""

REFLECTION_SYSTEM_PROMPT = """You will be presented with a problem, an attempt to solve it, a correctness label, and the correct answer. Your job is to reflect on the solution, try to understand why it was correct or not, and produce a brief (~1 paragraph) summary of the solution, what it got right, and/or where it went wrong."""

AGENT_PROMPT_TEMPLATE = """You are part of a team that tries to solve problems. This is the strategy you should use: {solution_prompt}
You should decide how to hand off problems or subproblems to others in the following way: {handoff_prompt}
You have the following teammates: {neighbors}"""
