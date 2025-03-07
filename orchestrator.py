"""
Multi-Agent Orchestrator for MineLand Environment

This script implements a budget-aware orchestrator system for the MineLand environment that intelligently manages multiple AI agents
(GPT-4o and GPT-4o-mini) with different costs to complete survival tasks in MineLand.
The orchestrator aims to efficiently use the budget while successfully completing tasks.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import mineland
from mineland import Action
from mineland.alex.alex_agent import Alex

# Add matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, bot_name: str):
        """
        Initialize the base agent.
        
        Args:
            bot_name: Name of the bot
        """
        self.bot_name = bot_name
        self.state = {
            "current_goal": None,
            "last_action": None,
            "inventory": {},
            "health": 0,
            "food": 0,
            "position": None,
        }
    
    def update_state(self, obs):
        """
        Update the agent's internal state based on observations.
        
        Args:
            obs: Observation from the environment
        """
        # Access attributes directly instead of using .get()
        self.state["health"] = getattr(obs, "health", 0)
        self.state["food"] = getattr(obs, "food", 0)
        self.state["position"] = getattr(obs, "position", None)
        self.state["inventory"] = getattr(obs, "inventory", {})
    
    def run(self, obs, code_info, task_info):
        """
        Process observations and generate an action.
        
        Args:
            obs: Observation from the environment
            code_info: Information about previously executed code
            task_info: Information about the current task
            
        Returns:
            An Action object with the code to execute
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def get_cost(self) -> float:
        """
        Get the total cost incurred by this agent.
        
        Returns:
            Total cost in dollars
        """
        raise NotImplementedError("Subclasses must implement get_cost()")

class LLMOrchestrator:
    """Orchestrator that uses an LLM to make agent selection decisions."""
    
    def __init__(
        self,
        task_id: str = "survival_0.5_days",
        total_budget: float = 1.0,
        orchestrator_model: str = "gpt-4o-v2",
        azure_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version: str = "2024-08-01-preview",
        save_path: str = "./storage/llm_orchestrator",
        api_key: str = None,
    ):
        """
        Initialize the LLM-based orchestrator.
        
        Args:
            task_id: ID of the survival task to complete
            total_budget: Total budget in dollars (for agent costs only, orchestrator costs are tracked separately)
            orchestrator_model: Model to use for orchestration decisions
            azure_endpoint: Azure OpenAI endpoint
            api_version: API version for Azure OpenAI
            save_path: Path to save orchestrator logs and data
            api_key: API key for Azure OpenAI
        """
        # Store configuration
        self.task_id = task_id
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.save_path = save_path
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        
        # Initialize environment
        self.mland = mineland.make(
            task_id=task_id,
            agents_count=2,
            enable_auto_pause=True,
            image_size=(144, 256),
        )
        
        # Initialize orchestrator model configuration
        self.orchestrator_model = orchestrator_model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        
        # Cost per token for the orchestrator model (not counted against the budget)
        self.cost_per_token = 0.00001  # $0.01 per 1K tokens for GPT-4o-mini
        if orchestrator_model == "gpt-4o-v2":
            self.cost_per_token = 0.00005  # $0.05 per 1K tokens for GPT-4o
        
        # Initialize token usage tracking for the orchestrator
        self.orchestrator_tokens_used = 0
        self.orchestrator_cost = 0.0
        
        # Initialize Azure OpenAI client for the orchestrator
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )
        
        # Initialize Alex agents directly
        self.gpt4o_agent = Alex(
            deployment_name="gpt-4o-v2",
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            max_tokens=512,
            temperature=0,
            chat_api_key=self.api_key,
            save_path=f"{save_path}/gpt4o",
            bot_name="Bot0",
        )
        
        self.gpt4o_mini_agent = Alex(
            deployment_name="gpt-4o-mini",
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            max_tokens=512,
            temperature=0,
            chat_api_key=self.api_key,
            save_path=f"{save_path}/gpt4o-mini",
            bot_name="Bot1",
        )
        
        # List of all agents
        self.agents = [self.gpt4o_agent, self.gpt4o_mini_agent]
        
        # Initialize metrics tracking
        self.metrics = {
            "total_steps": 0,
            "gpt4o_steps": 0,
            "gpt4o_mini_steps": 0,
            "gpt4o_cost": 0.0,
            "gpt4o_mini_cost": 0.0,
            "orchestrator_cost": 0.0,
            "total_cost": 0.0,
        }
        
        # Initialize decision logging
        self.decisions = []
        
        print(f"Initialized LLM Orchestrator using {orchestrator_model}")
        print(f"Budget of ${total_budget:.2f} applies to agent costs only. Orchestrator costs are tracked separately.")
    
    def _track_orchestrator_token_usage(self, tokens_used: int) -> None:
        """
        Track token usage and update costs for the orchestrator.
        These costs are tracked separately and not counted against the main budget.
        
        Args:
            tokens_used: Number of tokens used
        """
        self.orchestrator_tokens_used += tokens_used
        cost = tokens_used * self.cost_per_token
        self.orchestrator_cost += cost
        self.metrics["orchestrator_cost"] = self.orchestrator_cost
        print(f"[ORCHESTRATOR] Used {tokens_used} tokens (${cost:.5f}) - not counted against budget")
        print(f"[ORCHESTRATOR] Total tokens: {self.orchestrator_tokens_used}, Total cost: ${self.orchestrator_cost:.5f}")
    
    def _get_agent_selection_prompt(self, obs: List[Any], code_info: List[Any], task_info: Any) -> str:
        """
        Generate a prompt for the orchestrator to assign subtasks to both agents.
        
        Args:
            obs: List of observations for each agent
            code_info: List of code execution info for each agent
            task_info: Information about the current task
            
        Returns:
            A prompt string for the orchestrator
        """
        # Create a dictionary to store agent states
        agent_states = [
            {"position": "unknown", "health": 0, "food": 0, "inventory": {}},
            {"position": "unknown", "health": 0, "food": 0, "inventory": {}}
        ]
        
        # Get agent costs (estimated)
        gpt4o_cost = self.metrics["gpt4o_cost"]
        gpt4o_mini_cost = self.metrics["gpt4o_mini_cost"]
        
        # Get remaining budget (based only on agent costs, not including orchestrator)
        agent_total_cost = gpt4o_cost + gpt4o_mini_cost
        remaining_budget = self.total_budget - agent_total_cost
        
        # Get code execution status from the observation data
        gpt4o_code_status = getattr(code_info[0], "status", "unknown") if code_info[0] else "none"
        gpt4o_code_error = getattr(code_info[0], "error", None) if code_info[0] else None
        gpt4o_mini_code_status = getattr(code_info[1], "status", "unknown") if code_info[1] else "none"
        gpt4o_mini_code_error = getattr(code_info[1], "error", None) if code_info[1] else None
        
        # Extract key environmental information
        time_of_day = getattr(obs[0], "time_of_day", "unknown") if obs[0] else "unknown"
        
        # Create a detailed prompt for the orchestrator
        prompt = f"""You are an orchestrator managing two Minecraft agents that can work in parallel on different subtasks:

1. GPT-4o Agent (${gpt4o_cost:.2f} spent so far):
   - Last code status: {gpt4o_code_status}
   - Last code error: {gpt4o_code_error}
   - Cost per token: $0.05/1K tokens
   - Capabilities: More sophisticated reasoning, better planning, higher success rate for complex tasks

2. GPT-4o-mini Agent (${gpt4o_mini_cost:.2f} spent so far):
   - Last code status: {gpt4o_mini_code_status}
   - Last code error: {gpt4o_mini_code_error}
   - Cost per token: $0.01/1K tokens
   - Capabilities: Good at simple, routine tasks, cost-effective for well-defined activities

Current Environment:
- Time of day: {time_of_day}

Agent Budget Information:
- Total budget: ${self.total_budget:.2f}
- Agent costs so far: ${agent_total_cost:.2f}
- Remaining budget: ${remaining_budget:.2f}
- Note: Your own costs as the orchestrator are not counted against this budget, so focus only on optimizing the agents' costs.

Task Information:
{task_info}

Your job is to decide what subtask each agent should work on in parallel. Each agent can either:
1. GET_NEW_TASK: Assign a new subtask to the agent (this will interrupt their current task)
2. CONTINUE: Let the agent continue its current task (if applicable)

Task Assignment Guidelines:
- Assign complex tasks (planning, resource management, combat) to GPT-4o
- Assign simple tasks (gathering, movement, crafting) to GPT-4o-mini
- If budget is getting low (< 30% remaining), prefer using GPT-4o-mini for more tasks
- If an agent is failing at a task (has errors), consider reassigning that task
- Make sure agents are working on complementary tasks, not duplicating efforts

Respond in the following format:
GPT4O_ASSIGNMENT: [GET_NEW_TASK or CONTINUE] - [Brief description of task]
GPT4O_MINI_ASSIGNMENT: [GET_NEW_TASK or CONTINUE] - [Brief description of task]
RATIONALE: [One or two sentences explaining your decision]"""

        return prompt
    
    def log_decision(self, agent_idx: int, reason: str) -> None:
        """
        Log a decision to use a particular agent.
        
        Args:
            agent_idx: Index of the chosen agent
            reason: Reason for the decision
        """
        agent_name = "GPT-4o" if agent_idx == 0 else "GPT-4o-mini"
        decision = {
            "step": self.metrics["total_steps"],
            "agent": agent_name,
            "reason": reason,
            "remaining_budget": self.remaining_budget,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.decisions.append(decision)
        
        # Log to file
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(f"{self.save_path}/decisions.log", "a") as f:
            f.write(f"{decision['timestamp']}: Used {agent_name} - {reason}\n")

    def _assign_subtasks(self, obs: List[Any], code_info: List[Any], task_info: Any) -> Tuple[bool, bool, str, str]:
        """
        Use the orchestrator model to assign subtasks to both agents.
        
        Args:
            obs: List of observations for each agent
            code_info: List of code execution info for each agent
            task_info: Information about the current task
            
        Returns:
            A tuple containing:
            - get_new_task_gpt4o: Whether GPT-4o should get a new task (True) or continue (False)
            - get_new_task_gpt4o_mini: Whether GPT-4o-mini should get a new task (True) or continue (False)
            - gpt4o_task_description: Description of the task assigned to GPT-4o
            - gpt4o_mini_task_description: Description of the task assigned to GPT-4o-mini
        """
        # Generate the selection prompt
        prompt = self._get_agent_selection_prompt(obs, code_info, task_info)
        
        try:
            # Get response from the orchestrator model
            response = self.client.chat.completions.create(
                model=self.orchestrator_model,
                messages=[
                    {"role": "system", "content": "You are an orchestrator that assigns subtasks to two Minecraft agents that can work in parallel. Follow the requested format exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,  # Increased to capture all assignments and rationale
            )
            
            # Extract the complete response
            full_response = response.choices[0].message.content.strip()
            
            # Track token usage using actual counts from the API response
            total_tokens = response.usage.total_tokens
            self._track_orchestrator_token_usage(total_tokens)
            
            # Parse the assignment for GPT-4o
            gpt4o_assignment = ""
            gpt4o_task_description = ""
            gpt4o_lines = [line for line in full_response.split('\n') if line.startswith("GPT4O_ASSIGNMENT:")]
            if gpt4o_lines:
                gpt4o_assignment = gpt4o_lines[0].replace("GPT4O_ASSIGNMENT:", "").strip()
                if "-" in gpt4o_assignment:
                    assignment_parts = gpt4o_assignment.split("-", 1)
                    gpt4o_assignment = assignment_parts[0].strip()
                    gpt4o_task_description = assignment_parts[1].strip()
            
            # Parse the assignment for GPT-4o-mini
            gpt4o_mini_assignment = ""
            gpt4o_mini_task_description = ""
            gpt4o_mini_lines = [line for line in full_response.split('\n') if line.startswith("GPT4O_MINI_ASSIGNMENT:")]
            if gpt4o_mini_lines:
                gpt4o_mini_assignment = gpt4o_mini_lines[0].replace("GPT4O_MINI_ASSIGNMENT:", "").strip()
                if "-" in gpt4o_mini_assignment:
                    assignment_parts = gpt4o_mini_assignment.split("-", 1)
                    gpt4o_mini_assignment = assignment_parts[0].strip()
                    gpt4o_mini_task_description = assignment_parts[1].strip()
            
            # Parse the rationale
            rationale = ""
            rationale_lines = [line for line in full_response.split('\n') if line.startswith("RATIONALE:")]
            if rationale_lines:
                rationale = rationale_lines[0].replace("RATIONALE:", "").strip()
            
            # Determine if agents need new tasks
            get_new_task_gpt4o = "GET_NEW_TASK" in gpt4o_assignment
            get_new_task_gpt4o_mini = "GET_NEW_TASK" in gpt4o_mini_assignment
            
            # Log the decision
            print(f"[ORCHESTRATOR] GPT-4o assignment: {'GET NEW TASK' if get_new_task_gpt4o else 'CONTINUE'} - {gpt4o_task_description}")
            print(f"[ORCHESTRATOR] GPT-4o-mini assignment: {'GET NEW TASK' if get_new_task_gpt4o_mini else 'CONTINUE'} - {gpt4o_mini_task_description}")
            print(f"[ORCHESTRATOR] Rationale: {rationale}")
            
            # Log decisions
            self.log_decision(0, gpt4o_task_description)
            self.log_decision(1, gpt4o_mini_task_description)
            
            return get_new_task_gpt4o, get_new_task_gpt4o_mini, gpt4o_task_description, gpt4o_mini_task_description
                
        except Exception as e:
            print(f"Error in task assignment: {str(e)}")
            # Default to continue for both agents in case of error
            return False, False, "Continue current task (default)", "Continue current task (default)"
    
    def run(self, max_steps: int = 1000) -> None:
        """
        Run the orchestrator for a specified number of steps.
        
        Args:
            max_steps: Maximum number of steps to run
        """
        print(f"\n{'='*80}")
        print(f"Starting LLM Orchestrator using {self.orchestrator_model} for task: {self.task_id}")
        print(f"Total budget: ${self.total_budget:.2f} (excluding orchestrator costs)")
        print(f"{'='*80}\n")
        
        # Create directory for logs
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        try:
            # Reset the environment
            print("Resetting environment and connecting agents...")
            obs = self.mland.reset()
            print("Environment reset complete. Agents connected.")
            
            # Store current subtasks for each agents
            current_subtasks = ["Initial exploration", "Initial exploration"]
            
            # Main loop
            for step in range(max_steps):
                print(f"\n{'-'*80}")
                print(f"Step {step+1}/{max_steps}")
                print(f"{'-'*80}")
                
                # Get code info and task info
                code_info = []
                for i in range(len(obs)):
                    # Check if obs[i] is a dictionary and extract code_info properly
                    if isinstance(obs[i], dict):
                        code_info.append(obs[i].get("code_info", {}))
                    else:
                        # For object-style obs, use getattr with a default empty dict
                        code_info.append(getattr(obs[i], "code_info", {}))
                
                # Make sure code_info always has the required keys
                for i in range(len(code_info)):
                    if code_info[i] is None:
                        code_info[i] = {}
                    if "code_tick" not in code_info[i]:
                        code_info[i]["code_tick"] = 0
                    if "is_running" not in code_info[i]:
                        code_info[i]["is_running"] = False
                    if "code_error" not in code_info[i]:
                        code_info[i]["code_error"] = False
                
                # Get task_info similarly
                task_info = getattr(obs[0], "task_info", None)
                
                if step > 0 and step % 10 == 0:
                    print(f"task_info: {task_info}")
                
                # Create a fixed-size list to hold exactly 2 actions
                actions = mineland.Action.no_op(2)
                
                if step == 0:
                    # Skip the first step which includes respawn events and other initializations
                    print("\n[SYSTEM] First step: Using no-op actions for initialization")
                    # actions is already initialized with no-ops
                else:
                    # Every 5 steps or if this is step 1, call the orchestrator to assign subtasks
                    if step == 1 or step % 5 == 0:
                        print("\n[ORCHESTRATOR] Assigning subtasks to agents...")
                        _, _, gpt4o_task, gpt4o_mini_task = self._assign_subtasks(obs, code_info, task_info)
                        current_subtasks = [gpt4o_task, gpt4o_mini_task]
                        print(f"[ORCHESTRATOR] GPT-4o task: {current_subtasks[0]}")
                        print(f"[ORCHESTRATOR] GPT-4o-mini task: {current_subtasks[1]}")
                    
                    # Run agents one by one and place their actions in the correct index of the actions list
                    for idx, agent in enumerate(self.agents):
                        agent_name = "GPT-4o" if idx == 0 else "GPT-4o-mini"
                        subtask = current_subtasks[idx]
                        print(f"\n[AGENT] {agent_name} working on: {subtask}")
                        
                        try:
                            # Create a user message to prepend to the observation to instruct the agent
                            # about their specific subtask
                            message = f"Your current task: {subtask}. "
                            message += f"Focus on completing this specific subtask while the other agent handles different responsibilities."
                            
                            # For Alex agents, we can create a "chat" event to communicate the subtask
                            # This ensures the agent sees the subtask as part of its observation
                            if hasattr(obs[idx], 'event') and isinstance(obs[idx].event, list):
                                subtask_event = {
                                    "type": "chat",
                                    "message": f"[ORCHESTRATOR] {message}"
                                }
                                # Add the subtask event to the beginning of the events list
                                obs[idx].event.insert(0, subtask_event)
                            
                            # Run the agent with the augmented observations
                            # Ensure code_info is a dictionary with necessary keys before passing to agent.run
                            agent_code_info = code_info[idx]
                            if agent_code_info is None:
                                agent_code_info = {"code_tick": 0, "is_running": False, "code_error": False}
                            elif not isinstance(agent_code_info, dict):
                                # Convert to dictionary if it's an object
                                agent_code_info = {
                                    "code_tick": getattr(agent_code_info, "code_tick", 0),
                                    "is_running": getattr(agent_code_info, "is_running", False),
                                    "code_error": getattr(agent_code_info, "code_error", False)
                                }
                            
                            # Make sure all required keys exist
                            if "code_tick" not in agent_code_info:
                                agent_code_info["code_tick"] = 0
                            if "is_running" not in agent_code_info:
                                agent_code_info["is_running"] = False
                            if "code_error" not in agent_code_info:
                                agent_code_info["code_error"] = False
                                
                            action = agent.run(obs[idx], agent_code_info, task_info)
                            
                            # The agent.run method might return a single action or a list with one action
                            # Make sure we're assigning a single Action object to actions[idx]
                            if isinstance(action, list) and len(action) > 0:
                                actions[idx] = action[0]  # Take the first action if it's a list
                            else:
                                actions[idx] = action     # Use the action directly if it's not a list
                            
                            # Update cost estimates
                            if idx == 0:
                                self.metrics["gpt4o_steps"] += 1
                                self.metrics["gpt4o_cost"] += 0.02  # Rough estimate
                            else:
                                self.metrics["gpt4o_mini_steps"] += 1
                                self.metrics["gpt4o_mini_cost"] += 0.004  # Rough estimate
                                
                            # Log action type
                            if hasattr(actions[idx], 'type') and actions[idx].type == Action.NEW:
                                print(f"[AGENT] {agent_name} generated new code")
                            else:
                                print(f"[AGENT] {agent_name} is resuming previous execution")
                        except Exception as e:
                            print(f"Error running {agent_name}: {str(e)}")
                            # The no-op action is already set as default
                
                # Step the environment
                print("\n[ENVIRONMENT] Taking step...")
                obs, code_info, event, done, task_info = self.mland.step(action=actions)
                
                # Calculate costs (for display only)
                agent_costs = self.metrics["gpt4o_cost"] + self.metrics["gpt4o_mini_cost"]
                self.remaining_budget = self.total_budget - agent_costs
                
                # Log step information
                with open(f"{self.save_path}/steps.log", "a") as f:
                    f.write(f"Step {step+1}: Tasks: {current_subtasks}, Budget: ${self.remaining_budget:.2f}\n")
                
                # Check if task is done
                if done:
                    print(f"\n{'-'*80}")
                    print(f"Task completed at step {step+1}!")
                    print(f"{'-'*80}")
                    break
                    
                # Check if we've exceeded our budget
                if self.remaining_budget <= 0:
                    print(f"\n{'-'*80}")
                    print(f"Budget exceeded at step {step+1}. Stopping.")
                    print(f"{'-'*80}")
                    break
            
            # Calculate total cost (including orchestrator)
            self.metrics["total_cost"] = self.metrics["gpt4o_cost"] + self.metrics["gpt4o_mini_cost"] + self.metrics["orchestrator_cost"]
            
            # Print final metrics
            print(f"\n{'='*80}")
            print("Run complete. Final metrics:")
            print(f"Total steps: {self.metrics['total_steps']}")
            print(f"GPT-4o actions generated: {self.metrics['gpt4o_steps']}")
            print(f"GPT-4o-mini actions generated: {self.metrics['gpt4o_mini_steps']}")
            
            # Calculate agent cost (excluding orchestrator)
            agents_cost = self.metrics["gpt4o_cost"] + self.metrics["gpt4o_mini_cost"]
            
            print(f"Agent cost (counted against budget): ${agents_cost:.2f}")
            print(f"- GPT-4o cost: ${self.metrics['gpt4o_cost']:.2f}")
            print(f"- GPT-4o-mini cost: ${self.metrics['gpt4o_mini_cost']:.2f}")
            print(f"Orchestrator cost (not counted in budget): ${self.metrics['orchestrator_cost']:.2f}")
            print(f"Total cost (including orchestrator): ${self.metrics['total_cost']:.2f}")
            print(f"{'='*80}\n")
            
            # Save final metrics to a JSON file
            metrics_path = Path(self.save_path) / "metrics.json"
            with open(metrics_path, "w") as f:
                import json
                json.dump(self.metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")
            
        finally:
            # Close the environment
            self.mland.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MineLand with LLM orchestration")
    parser.add_argument("--task", type=str, default="survival_0.5_days", help="Task ID to run")
    parser.add_argument("--budget", type=float, default=1.0, help="Total budget in dollars")
    parser.add_argument("--orchestrator-model", type=str, default="gpt-4o-v2", help="Model to use for orchestration")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of steps to run")
    parser.add_argument("--save-path", type=str, default="./storage/llm_orchestrator", help="Path to save results")
    parser.add_argument("--api-key", type=str, help="Azure OpenAI API key")
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = LLMOrchestrator(
        task_id=args.task,
        total_budget=args.budget,
        orchestrator_model=args.orchestrator_model,
        save_path=args.save_path,
        api_key=args.api_key,
    )
    
    # Run the orchestrator
    orchestrator.run(max_steps=args.max_steps) 