'''
This script demonstrates how to run the Alex agent in the mineland environment.

Please set your key in OPENAI_API_KEY environment variable before running this script.
Or, you can set the key in the script as follows (not recommended):
'''
# import os
# os.environ["OPENAI_API_KEY"] = "" # set your key here

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import mineland
from mineland.alex import Alex

# Create results directory
results_dir = Path('results')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = results_dir / f'alex_run_{timestamp}'
save_dir.mkdir(parents=True, exist_ok=True)

# Create separate directories for each agent
agent_dirs = []
for i in range(2):  # For both agents
    agent_dir = save_dir / f'agent_{i}'
    agent_dir.mkdir(exist_ok=True)
    agent_dirs.append(agent_dir)

mland = mineland.make(
    task_id="playground",
    agents_count = 2,
    image_size=(256, 256),  # Optional: customize view size
)

# initialize agents
agents = []
alex = Alex(personality='None',             # Alex configuration
            bot_name='MineflayerBot0',
            temperature=0.1)
blex = Alex(personality='None',             # Alex configuration
            bot_name='MineflayerBot1',
            temperature=0.1)

agents.append(alex)
agents.append(blex)

obs = mland.reset()

agents_count = len(obs)
agents_name = [obs[i]['name'] for i in range(agents_count)]

try:
    for i in range(5000):
        if i > 0 and i % 10 == 0:
            print("task_info: ", task_info)
        
        # Save frames for all agents
        for agent_idx in range(agents_count):
            frame = obs[agent_idx]['rgb']  # Get agent's view (format: CHW)
            frame = np.transpose(frame, (1, 2, 0))  # Convert to HWC format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = agent_dirs[agent_idx] / f'frame_{i:05d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            
        actions = []
        if i == 0:
            # skip the first step which includes respawn events and other initializations
            actions = mineland.Action.no_op(agents_count)
        else:
            # run agents
            for idx, agent in enumerate(agents):
                action = agent.run(obs[idx], code_info[idx], done, task_info, verbose=True)
                actions.append(action)

        obs, code_info, event, done, task_info = mland.step(action=actions)
        
        if done:
            break

finally:
    mland.close()
