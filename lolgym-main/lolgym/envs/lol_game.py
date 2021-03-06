import logging

import gym
from pylol.env import lol_env
from pylol.env.environment import StepType
from pylol.lib import actions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id

def players_to_agents(players):
    player_list = players.split(",")
    player_list = [player.split(".") for player in player_list]
    return [lol_env.Agent(champion=c, team=t)
            for c, t in player_list]

class LoLGameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = {
        'agent_interface_format': \
            lol_env.parse_agent_interface_format(
            feature_map=16000,
            feature_move_range=8),
        'host': 'localhost',
        'players': "Ezreal.BLUE,Ezreal.PURPLE",
        'config_path': '/c/Users/matth/Desktop/pylol/config_dirs.txt'
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self.n_agents = 2

        self._episode = 0
        self._num_step = 0
        self._episode_reward = [0.0] * self.n_agents
        self._total_reward = [0.0] * self.n_agents

    def step(self, actions):
        return self._safe_step(actions)
    
    def broadcast_msg(self, msg):
        self._env.broadcast_msg(msg)

    def _safe_step(self, acts):
        self._num_step += 1
        try:
            cur_actions = [actions.FunctionCall(act[0], act[1:])
                           for act in acts]
            obs_n = self._env.step(cur_actions)
        except KeyboardInterrupt:
            logger.info(" Interrupted. Quitting...")
            return [None] * self.n_agents, [0] * self.n_agents, [True] * self.n_agents, \
                [{}] * self.n_agents
        except Exception:
            logger.exception(" An unexpected error occurred while applying action to environment.")
            return [None] * self.n_agents, [0] * self.n_agents, [True] * self.n_agents, \
                [{}] * self.n_agents
        reward_n = [obs.reward for obs in obs_n]
        self._episode_reward = [self._episode_reward[i] + reward_n[i] for i in range(self.n_agents)]
        self._total_reward = [self._total_reward[i] + reward_n[i] for i in range(self.n_agents)]
        done_n = [obs.step_type == StepType.LAST for obs in obs_n]
        return obs_n, reward_n, done_n, {}

    def teleport(self, playerId, point):
        """Teleports a player to a certain position on the map. this can only be done in a practice game so needs to be tuned for the live version"""
        if self._env:
            self._env._controllers[0].player_teleport(playerId, point[0], point[1])

    def reset(self):
        print("RESET")
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            mean_reward = [float(total_reward) / self._episode
                           for total_reward in self._total_reward]
            logger.info(" Episode {0} ended with reward {1} after {2} steps.".format(
                        self._episode, self._episode_reward, self._num_step))
            logger.info(" Got {0} total reward, with an average reward of {1} per episode".format(
                        self._total_reward, mean_reward))
        self._episode += 1
        self._num_step = 0
        self._episode_reward = [0.0] * self.n_agents
        logger.info(" Episode %d starting...", self._episode)
        obs = self._env.reset()
        # self.available_actions = obs.observation['available_actions']
        return obs

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        args["players"] = players_to_agents(args["players"])
        logger.debug(" Intialize LoLEnv with settings: %s", args)

        # Setup environment
        self._env = lol_env.LoLEnv(**args)

        # Connect to the game after setting up the environment
        controller = self._env._controllers[0]
        controller.connect()

    def close(self):
        print("CLOSE")
        if self._episode > 0:
            mean_reward = [float(total_reward) / self._episode
                           for total_reward in self._total_reward]
            logger.info(" Episode {0} ended with reward {1} after {2} steps.".format(
                        self._episode, self._episode_reward, self._num_step))
            logger.info(" Got {0} total reward, with an average reward of {1} per episode".format(
                        self._total_reward, mean_reward))
        if self._env is not None:
            self._env.close()
        super().close()
        
    @property
    def settings(self):
        return self._kwargs

    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward