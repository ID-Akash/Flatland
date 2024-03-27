from new_env.env import FlatlandSingle
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

if __name__ == "__main__":
    config = {}
    env = FlatlandSingle(config)
    # env.step([0]).obs.shape
    # print(env.step([0]))

    register_env("flatland", lambda config: FlatlandSingle(config))
    config['env'] = "flatland"
    algo = (
        PPOConfig()
        .environment("flatland", env_config=config)
        .build()
    )

    # algo.train()
    for i in range(10):
        result = algo.train()
        print(pretty_print(result))
        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")

    # ray.rllib.utils.check_env(env)

            
'''
TODOS:
1. Changing model specs
2. Changing Algo params
3. Running the model on GPU cluster
4. Setup of multi agent environment
5. Multi agent on distributed environment
'''