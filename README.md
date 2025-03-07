# MineLand Budget-Aware Multi-Agent Orchestrator

This project implements a budget-aware orchestrator system for the MineLand environment that intelligently manages multiple AI agents with different capabilities and costs to complete Minecraft survival tasks.

## Overview

The orchestrator system includes:

1. **Two LLM Agents**: 
   - **GPT-4o Agent**: A more capable but expensive agent
   - **GPT-4o-mini Agent**: A less capable but cheaper agent

2. **Budget-Aware Orchestrator**:
   - Dynamically decides which agent to use based on:
     - Current situation complexity
     - Remaining budget
     - Task progress

3. **Decision Metrics**:
   - Situation assessment (health, food, combat, time of day)
   - Budget allocation strategies
   - Cost tracking and logging

## Requirements

- Python 3.10+
- MineLand environment
- Azure OpenAI API access
- OpenAI Python SDK
- Other dependencies listed in `requirements.txt`

## Installation

1. Install MineLand environment following [MineLand installation instructions](https://github.com/cocacola-lab/MineLand).

2. Install additional dependencies:

```bash
pip install openai numpy opencv-python
```

3. Set up your Azure OpenAI API key:

```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Run the orchestrator with default settings:

```bash
python run_orchestrator.py
```

### Advanced Usage

Customize the orchestrator run with command-line arguments:

```bash
python run_orchestrator.py --task survival_1_days --budget 2.0 --max-steps 10000
```

Available arguments:

- `--task`: MineLand survival task ID (e.g., "survival_0.5_days")
- `--budget`: Total budget in dollars (e.g., 1.0 for $1.00)
- `--api-key`: Azure OpenAI API key (if not set in environment)
- `--max-steps`: Maximum number of steps to run
- `--save-dir`: Directory to save results

## Architecture

### BaseAgent

Abstract base class that defines the interface for all agent types:
- Observation processing
- Action generation
- Cost tracking

### GPT4oAgent

High-capability agent using GPT-4o model:
- Detailed observation processing
- Complex planning and reasoning
- Higher cost per token

### GPT4oMiniAgent

More economical agent using GPT-4o-mini model:
- Simplified observation processing
- Basic planning and reasoning
- Lower cost per token

### Orchestrator

Central controller that:
- Assesses the current situation
- Chooses appropriate agent based on complexity and budget
- Tracks costs and manages budget
- Logs decisions and metrics

## Results and Logging

The system generates comprehensive logs and metrics:

- **Frame Captures**: Screenshots from agent perspectives
- **Decision Logs**: Records of which agent was used and why
- **Budget Tracking**: Detailed cost accounting
- **Performance Metrics**: Steps, success rates, agent usage
- **Agent Prompts/Responses**: Input/output for each agent

## Examples

Example decision making process:

1. **Critical Health Situation**:
   - High complexity score (0.7)
   - Sufficient budget remaining (70%)
   - Decision: Use GPT-4o for sophisticated planning

2. **Routine Resource Gathering**:
   - Low complexity score (0.2)
   - Decision: Use cheaper GPT-4o-mini to conserve budget

3. **Low Budget Situation**:
   - Remaining budget below threshold
   - Decision: Force use of GPT-4o-mini regardless of complexity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MineLand Project](https://github.com/cocacola-lab/MineLand)
- Azure OpenAI API
