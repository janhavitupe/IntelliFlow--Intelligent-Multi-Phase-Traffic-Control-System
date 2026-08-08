"""
train.py

Training loops for the Phase 4 RL agents.

Stage 1: train_tabular
    Runs TabularQAgent against TrafficRLEnv across many episodes, cycling
    traffic profiles so the agent does not overfit to a single pattern.
    Returns the per-episode cumulative reward curve (rising curve = the
    single most convincing piece of evidence the RL is working).

Stage 2: train_dqn
    Runs DQNAgent (pure-numpy MLP) against the SAME environment, but over
    the RAW 23-dim observation (no discretization). Uses experience replay,
    a target network, and epsilon-greedy exploration. Shares the same
    episode/profile/seed structure as train_tabular so the two are directly
    comparable.

Training loop shape (per the phase-4 spec):
    - Episode = one fixed-length window (EPISODE_LENGTH ticks) of a profile.
    - Profiles vary across episodes (LIGHT/NORMAL/RUSH/NIGHT/CUSTOM).
    - A fixed per-episode seed keeps evaluation reproducible.
"""
from config import rl as rl_config
from env.traffic_env import TrafficRLEnv
from .agents import TabularQAgent
from .dqn import DQNAgent


def train_tabular(
    agent=None,
    n_episodes=None,
    profiles=None,
    episode_length=None,
    seed=None,
    verbose=True,
):
    """
    Train a TabularQAgent and return the per-episode reward curve.

    Args:
        agent (TabularQAgent|None): agent to train (created if None).
        n_episodes (int|None): number of episodes.
        profiles (tuple|None): traffic profiles to cycle across episodes.
        episode_length (int|None): ticks per episode.
        seed (int|None): base seed per-episode reproducibility.
        verbose (bool): print progress.

    Returns:
        (TabularQAgent, list[float]): (trained agent, per-episode reward).
    """
    n_episodes = n_episodes if n_episodes is not None else rl_config.TABULAR_EPISODES
    profiles = profiles if profiles is not None else rl_config.PROFILES
    episode_length = (
        episode_length if episode_length is not None else rl_config.EPISODE_LENGTH
    )
    seed = seed if seed is not None else rl_config.SEED

    if agent is None:
        agent = TabularQAgent(seed=seed)

    rewards = []
    for ep in range(n_episodes):
        profile_key = profiles[ep % len(profiles)]
        ep_seed = seed + ep

        env = TrafficRLEnv(
            profile_key=profile_key,
            episode_length=episode_length,
            seed=ep_seed,
        )
        env.reset()

        # Initial discrete state: no active phase yet -> last_phase None.
        state = env.discretize_state(None)

        ep_reward = 0.0
        done = False
        while not done:
            action = agent.epsilon_greedy(state)
            _, reward, done, _ = env.step(action)

            next_state = env.discretize_state(env.scheduler.active_phase_type)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        rewards.append(ep_reward)

        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == n_episodes - 1):
            print(
                f"ep {ep + 1}/{n_episodes} profile={profile_key:<14} "
                f"reward={ep_reward:>10.1f} eps={agent.epsilon:.2f}"
            )

    return agent, rewards


def train_dqn(
    agent=None,
    n_episodes=None,
    profiles=None,
    episode_length=None,
    seed=None,
    verbose=True,
):
    """
    Train a DQNAgent over the raw 23-dim observation and return the curve.

    Args:
        agent (DQNAgent|None): agent to train (created if None).
        n_episodes (int|None): number of episodes.
        profiles (tuple|None): traffic profiles to cycle across episodes.
        episode_length (int|None): ticks per episode.
        seed (int|None): base seed per-episode reproducibility.
        verbose (bool): print progress.

    Returns:
        (DQNAgent, list[float]): (trained agent, per-episode reward).
    """
    n_episodes = n_episodes if n_episodes is not None else rl_config.DQN_EPISODES
    profiles = profiles if profiles is not None else rl_config.PROFILES
    episode_length = (
        episode_length if episode_length is not None else rl_config.EPISODE_LENGTH
    )
    seed = seed if seed is not None else rl_config.SEED

    if agent is None:
        agent = DQNAgent(
            obs_dim=rl_config.OBS_DIM,
            n_actions=rl_config.NUM_PHASES,
            seed=seed,
        )

    rewards = []
    for ep in range(n_episodes):
        profile_key = profiles[ep % len(profiles)]
        ep_seed = seed + ep

        env = TrafficRLEnv(
            profile_key=profile_key,
            episode_length=episode_length,
            seed=ep_seed,
        )
        obs = env.reset()

        ep_reward = 0.0
        done = False
        while not done:
            action = agent.epsilon_greedy(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        rewards.append(ep_reward)

        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == n_episodes - 1):
            print(
                f"ep {ep + 1}/{n_episodes} profile={profile_key:<14} "
                f"reward={ep_reward:>10.1f} eps={agent.epsilon:.2f}"
            )

    return agent, rewards
