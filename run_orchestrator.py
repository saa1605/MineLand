#!/usr/bin/env python3
"""
Runner script for the MineLand budget-aware orchestrator

This script demonstrates how to use the LLM-based orchestrator with different configurations.
"""

import argparse
import os
import time
from pathlib import Path

from orchestrator import LLMOrchestrator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the MineLand budget-aware LLM orchestrator with different configurations"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        default="survival_0.5_days",
        help="Task ID from MineLand's survival tasks (e.g., survival_0.5_days)"
    )
    
    parser.add_argument(
        "--budget", 
        type=float, 
        default=1.0,
        help="Total budget for the task in dollars (e.g., 1.0 for $1.00)"
    )
    
    parser.add_argument(
        "--orchestrator-model",
        type=str,
        default="gpt-4o-v2",
        choices=["gpt-4o-v2", "gpt-4o-mini"],
        help="Model to use for the orchestrator (default: gpt-4o-v2)"
    )
    
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI endpoint"
    )
    
    parser.add_argument(
        "--api-version",
        type=str,
        default="2024-08-01-preview",
        help="Azure API version"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="Azure OpenAI API key (if not set in environment variable)",
        default=os.environ.get("AZURE_OPENAI_API_KEY")
    )
    
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=5000,
        help="Maximum number of steps to run"
    )
    
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="./results",
        help="Directory to save results"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the LLM orchestrator."""
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    run_dir = save_dir / f"run_{timestamp}"
    
    # Set API key in environment if provided
    if args.api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = args.api_key
    
    print(f"Running LLM orchestrator with task: {args.task}")
    print(f"Orchestrator model: {args.orchestrator_model}")
    print(f"Budget: ${args.budget:.2f} (for agent costs only)")
    print(f"Max steps: {args.max_steps}")
    print(f"Results will be saved to: {run_dir}")
    
    # Create and run the LLM orchestrator
    orchestrator = LLMOrchestrator(
        task_id=args.task,
        total_budget=args.budget,
        orchestrator_model=args.orchestrator_model,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        save_path=str(run_dir),
        api_key=args.api_key,
    )
    
    # Run the orchestrator
    orchestrator.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main() 