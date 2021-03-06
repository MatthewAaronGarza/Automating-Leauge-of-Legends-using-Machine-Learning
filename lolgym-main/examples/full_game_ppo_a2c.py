# MIT License
# 
# Copyright (c) 2020 MiscellaneousStuff
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Example of basic full game environment implementing A2C PPO."""

import threading
from queue import Queue

from .full_game_ppo import PPOAgent, Controller

from absl import flags
from absl import app

from gym.spaces import Box, Discrete

import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer("count", 1, "Number of games to run at once")
# NOTE: Below flags are inherited from full_game_ppo module
#flags.DEFINE_string("config_path", "/mnt/c/Users/win8t/Desktop/pylol/config_dirs.txt", "Path to file containing GameServer and LoL Client directories")
#flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")

def train_thread(controller, run_client):
    epochs = 10
    batch_steps = 25
    episode_steps = batch_steps
    experiment_name = "run_away"
    run_client = run_client

    # Declare, train and run agent
    agent = PPOAgent(controller=controller, run_client=run_client)
    agent.train(epochs=epochs,
                batch_steps=batch_steps,
                episode_steps=episode_steps,
                experiment_name=experiment_name)
    agent.run(max_steps=episode_steps)

    agent.close()

def main(unused_argv):
    """Run an agent."""

    # Declare observation space, action space and model controller
    units = 1 # <= try changing this next...
    gamma = 0.99
    observation_space = Box(low=0, high=24000, shape=(1,), dtype=np.float32)
    action_space = Discrete(2)
    controller = Controller(units, gamma, observation_space, action_space)

    threads = []
    for _ in range(FLAGS.count-1):
        t = threading.Thread(target=train_thread,
                             args=(controller, False,))
        threads.append(t)
        t.start()
    
    train_thread(controller, True) # Training thread which renders the client

    for t in threads:
        t.join()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)