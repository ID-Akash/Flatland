import logging
from typing import Any, NamedTuple, Dict
from collections import defaultdict

import gymnasium as gym
import numpy as np
# from flatland.envs.malfunction_generators import (
#     no_malfunction_generator,
#     malfunction_from_params,
#     MalfunctionParameters,
# )
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
# from flatland.envs.observations import GlobalObsForRailEnv
from new_env.global_obs import GlobalObservation


# class StepOutput(NamedTuple):
#     obs: Dict[int, Any]  # depends on observation builder
#     reward: Dict[int, float]
#     terminated: Dict[int, bool]
#     truncated: Dict[int, bool]
#     info: Dict[int, Dict[str, Any]]
    # done: Dict[int, bool]

class StepOutput(NamedTuple):
    obs: Any  # depends on observation builder
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    # done: bool


class FlatlandSingle(gym.Env):
    def render(self, mode="human"):
        pass

    def __init__(self, env_config):
        self._config = env_config
        self._config = {"number_of_agents": 1}
        rail_generator = sparse_rail_generator(max_num_cities=3)
        # malfunction_generator = no_malfunction_generator()
        line_generator = sparse_line_generator()

        self.env = None
        obs_config = {"max_width": 42, "max_height": 42}
        self._obs = GlobalObservation(obs_config)
        try:
            self.env = RailEnv(
                width=42,
                height=42,
                number_of_agents=1,
                rail_generator=rail_generator,
                line_generator=line_generator,
                obs_builder_object=self._obs.builder(),
                # malfunction_generator=malfunction_generator,
            )
        except ValueError as e:
            logging.error("=" * 50)
            logging.error(f"Error while creating env: {e}")
            logging.error("=" * 50)

        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        self._agents_done = []
        self.env.reset()

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = self._obs.observation_space()

        assert isinstance(observation_space, gym.spaces.Box)

        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                # self._config["number_of_agents"],
                *observation_space.shape,
            ),
        )

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([5] * self._config["number_of_agents"])

    def step(self, action_list: list) -> StepOutput:
        action_dict = {}
        for i, action in enumerate(action_list):
            action_dict[i] = action
        d, r, o = None, None, None
        terminated, truncated = None, None
        obs_or_done = False
        while not obs_or_done:
            # Perform env steps as long as there is no observation (for all agents) or all agents are done
            # The observation is `None` if an agent is done or malfunctioning.
            obs, rewards, dones, infos = self.env.step(action_dict)

            d, r, o, terminated, truncated = dict(), dict(), dict(), dict(), dict()
            
            for agent, done in dones.items():
                if agent != "__all__" and not agent in obs:
                    continue  # skip agent if there is no observation

                # Use this if using a single policy for multiple agents
                # if True or agent not in self._agents_done:
                if agent not in self._agents_done:
                    if agent != "__all__":
                        if done:
                            self._agents_done.append(agent)
                        o[agent] = obs[agent]
                        r[agent] = rewards[agent]
                        self._agent_scores[agent] += rewards[agent]
                        self._agent_steps[agent] += 1
                    d[agent] = dones[agent]
                    terminated[agent] = dones[agent]
                    truncated[agent] = dones[agent]

            action_dict = (
                {}
            )  # reset action dict for cases where we do multiple env steps
            obs_or_done = (
                len(o) > 0 or d["__all__"]
            )  # step through env as long as there are no obs/all agents done

        # return StepOutput(obs=obs[0])
        # assert all([x is not None for x in (d, r, o)])

        return StepOutput(
            obs=o[0],
            reward=r[0],
            # done=d,
            terminated=terminated[0],
            truncated=truncated[0],
            info={
                agent: {
                    "max_episode_steps": self.env._max_episode_steps,
                    "num_agents": self.env.get_num_agents(),
                    "agent_done": d[agent] and agent not in self.env.active_agents,
                    "agent_score": self._agent_scores[agent],
                    "agent_step": self._agent_steps[agent],
                }
                for agent in o.keys()
            },
        )
        # return StepOutput(
        #     obs=[step for step in step_r.obs.values()],
        #     reward=np.sum([r for r in step_r.reward.values()]),
        #     terminated=all(step_r.done.values()),
        #     truncated=all(step_r.done.values()),
        #     info=step_r.info[0],
        #     done=all(step_r.done.values()),
        # )

    def reset(self, seed=None, options=None):
        # foo, _ = self.env.reset()

        # print("="*50)
        # print(foo)
        # print("="*50)

        # return [step for step in foo.values()], {}
        self._agents_done = []
        self._agent_scores = defaultdict(float)
        self._agent_steps = defaultdict(int)
        obs, infos = self.env.reset(
            regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=seed,
        )
        # return {k: o for k, o in obs.items() if not k == "__all__"}, {}
        return obs[0], {}

        # return foo



# config = {}
# env = FlatlandSingle(config)
# env.reset()