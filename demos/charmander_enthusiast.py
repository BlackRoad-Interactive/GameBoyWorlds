"""
To run this script, you will need to run:
"""
install_command = "uv pip install stable-baselines3[extra] wandb tensorboard"

from poke_worlds import get_environment
from poke_worlds.utils import show_frames
import gymnasium as gym
from gymnasium.spaces import Discrete, OneOf
import numpy as np
import click

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecTransposeImage
    from stable_baselines3.common.callbacks import CallbackList
    from wandb.integration.sb3 import WandbCallback
    import wandb    
except ImportError:
    print(f"Please install required packages to run this script:\n{install_command}`")
    exit(1)



class OneOfToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate total actions across all sub-spaces
        # Example: OneOf([Discrete(2), Discrete(3)]) -> total 5
        self.internal_env = env
        self.sub_spaces = env.action_space.spaces
        self.total_actions = sum(s.n for s in self.sub_spaces)
        self.action_space = Discrete(self.total_actions)

    def action(self, action):
        # Map the single integer back to (choice, sub_action)
        offset = 0
        for i, space in enumerate(self.sub_spaces):
            if action < offset + space.n:
                return (i, action - offset)
            offset += space.n
        print("Action mapping error!")
        return (0, 0)  # Fallback
    
    def get_high_level_action(self, action):
        # Map the single integer back to choice only
        action = self.action(action)
        high_level_action, kwargs = self.internal_env._controller._space_action_to_high_level_action(
            action
        )
        return high_level_action, kwargs
    



def make_env(rank, seed=0, save_video=False):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        original_env = get_environment(
            game="pokemon_red",
            controller_variant="low_level",
            environment_variant="charmander",
            max_steps=20,
            headless=True,
            save_video=save_video,
        )
        original_env.seed(seed + rank)  # Doesn't matter here, its deterministic
        ind_env = Monitor(OneOfToDiscreteWrapper(original_env))
        return ind_env

    return _init


@click.command()
@click.option(
    "--num_cpu", type=int, default=1, help="Number of CPU cores to use for training."
)
@click.option("--batch_size", type=int, default=64, help="Batch size for training.")
@click.option(
    "--exploration_fraction",
    type=float,
    default=0.1,
    help="Exploration fraction for training.",
)
@click.option(
    "--gamma", type=float, default=0.999, help="Discount factor for training."
)
@click.option(
    "--total_timesteps",
    type=int,
    default=int(2e8),
    help="Total timesteps for training.",
)
def train(num_cpu, batch_size, exploration_fraction, gamma, total_timesteps):
    callbacks = []
    #run = wandb.init(
    #    project="PokeWorlds",
    #    name="charmander_enthusiast",
    #    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #    monitor_gym=True,  # auto-upload the videos of agents playing the game
    #    save_code=False,  # optional
    #)
    #callbacks.append(WandbCallback())


    #env = VecNormalize(VecTransposeImage(SubprocVecEnv([make_env(i) for i in range(num_cpu)])), norm_obs=True, norm_reward=True)
    env = VecNormalize(VecTransposeImage(DummyVecEnv([make_env(0)])), norm_obs=True, norm_reward=True)  # Using single environment for simplicity

    # Instantiate the agent
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        #tensorboard_log=f"runs/{run.id}",
        batch_size=batch_size,
        policy_kwargs=dict(normalize_images=False)
    )
    # Train the agent and display a progress bar
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        #callback=CallbackList(callbacks),
    )
    # Save the agent
    model.save("charmander_enthusiast_agent")
    print("Training complete!")
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=1
    )
    print(f"Reward: {mean_reward} +/- {std_reward}")
    vec_env = model.get_env()
    obs = vec_env.reset()
    #obs_image = obs.reshape(144, 160, 1)
    #show_frames([obs_image], titles=[f"tmp_0"], save=True)    
    for i in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        high_level_action, kwargs = vec_env.envs[0].env.get_high_level_action(action[0])
        print(f"Step {i}: High-level action: {high_level_action}, kwargs: {kwargs}, reward: {rewards}")
        #obs_image = obs.reshape(144, 160, 1)
        #show_frames([obs_image], titles=[f"tmp_{i+1}"], save=True)
        #vec_env.render()


@click.command()
@click.option(
    "--render",
    type=bool,
    default=True,
    help="Whether to render the environment during evaluation.",
)
def evaluate(render):
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load("charmander_enthusiast_agent", env=VecNormalize(VecTransposeImage(DummyVecEnv([make_env(0, save_video=True)])), norm_obs=True, norm_reward=False))

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=1
    )
    print(f"Reward: {mean_reward} +/- {std_reward}")

    # Enjoy trained agent
    if render:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render()


@click.group()
def main():
    pass


main.add_command(train)
main.add_command(evaluate)

if __name__ == "__main__":
    main()
