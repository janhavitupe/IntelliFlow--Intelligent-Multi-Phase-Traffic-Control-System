"""
train.py

Training loops for the Phase 4 RL agents.

Stage 1: train_tabular
    Runs TabularQAgent against TrafficRLEnv, using the discretized state.

Stage 2: train_dqn
    Runs DQNAgent (pure-numpy MLP) against the SAME environment, but over
    the RAW 23-dim observation (no discretization).

Both share one loop (_train), so they see identical episodes:
    - Episode = one fixed-length window (EPISODE_LENGTH ticks) of a profile.
    - Profiles rotate across episodes (LIGHT/NORMAL/RUSH/NIGHT/CUSTOM) and
      each episode has its own seed (SEED + episode).
    - Epsilon decays LINEARLY to its floor over EPSILON_DECAY_FRACTION of
      the episodes, so exploration always finishes regardless of budget.
    - The episode's time limit is a truncation, not a terminal state, so
      the last transition still bootstraps.
    - Every VALIDATION_EVERY episodes the GREEDY policy is scored on
      held-out seeds through the real Simulation loop. That validation
      curve (avg wait vs episode) is the honest learning curve: raw episode
      rewards mix five profiles of very different difficulty.
    - The best-scoring checkpoint is restored at the end (early-stopping
      style model selection). Validation seeds are disjoint from both the
      training seeds and the final evaluation seeds, so this never peeks
      at test data.
"""
from config import rl as rl_config
from env.traffic_env import TrafficRLEnv
from .agents import TabularQAgent
from .dqn import DQNAgent


def epsilon_at(episode, n_episodes):
    """Linear decay from EPSILON_START to EPSILON_END, then flat."""
    decay_episodes = max(1, int(n_episodes * rl_config.EPSILON_DECAY_FRACTION))
    frac = min(1.0, episode / decay_episodes)
    return rl_config.EPSILON_START + frac * (
        rl_config.EPSILON_END - rl_config.EPSILON_START
    )


def validate(agent, profiles=None, seeds=None, ticks=None):
    """
    Score the greedy policy on held-out seeds (no exploration, no learning).

    Runs the standard Simulation with an RLStrategy, exactly as the final
    evaluation does.

    Returns:
        float: average waiting time across profiles x seeds (lower is better).
    """
    from strategies.rl_strategy import RLStrategy

    return score_policy(lambda: RLStrategy(agent=agent), profiles, seeds, ticks)


def score_policy(strategy_factory, profiles=None, seeds=None, ticks=None):
    """
    Average waiting time of any strategy on the validation set.

    Used for RL checkpoints and for the Fixed/Density reference lines, so
    all of them are scored identically.
    """
    from simulation import Simulation

    profiles = profiles if profiles is not None else rl_config.PROFILES
    seeds = seeds if seeds is not None else rl_config.VALIDATION_SEEDS
    ticks = ticks if ticks is not None else rl_config.VALIDATION_TICKS

    waits = []
    for profile_key in profiles:
        for seed in seeds:
            sim = Simulation(
                profile_key=profile_key, seed=seed, max_ticks=ticks,
                live=False, strategy=strategy_factory(),
            )
            for _ in range(ticks):
                sim.step()
            waits.append(sim.analytics.average_waiting_time)
    return sum(waits) / len(waits)


def _train(agent, use_discrete, n_episodes, profiles, episode_length, seed, verbose):
    """
    Shared episode loop.

    Returns:
        dict with
            "episode_rewards": list[float], one per episode
            "episode_profiles": list[str], profile of each episode
            "validation": list[(episode, avg_wait)], greedy-policy checkpoints
            "best_episode": int, checkpoint restored into the agent
    """
    history = {"episode_rewards": [], "episode_profiles": [], "validation": []}
    best = {"wait": float("inf"), "episode": 0, "weights": None}

    def checkpoint(ep):
        wait = validate(agent)
        history["validation"].append((ep, wait))
        if wait < best["wait"]:
            best.update(wait=wait, episode=ep, weights=agent.get_weights())
        if verbose:
            print(f"  [validation] after {ep:>4} episodes: greedy avg wait = {wait:.1f}")

    checkpoint(0)
    for ep in range(n_episodes):
        profile_key = profiles[ep % len(profiles)]
        agent.epsilon = epsilon_at(ep, n_episodes)

        env = TrafficRLEnv(
            profile_key=profile_key,
            episode_length=episode_length,
            seed=seed + ep,
        )
        obs = env.reset()
        state = env.discrete_state if use_discrete else obs

        ep_reward = 0.0
        done = False
        while not done:
            action = agent.epsilon_greedy(state)
            next_obs, reward, done, info = env.step(action)
            next_state = env.discrete_state if use_discrete else next_obs
            terminal = done and not info.get("truncated", False)
            # Semi-MDP discount: longer steps discount the future more.
            discount = rl_config.GAMMA_PER_SECOND ** info["duration"]

            if use_discrete:
                agent.update(state, action, reward, next_state, terminal, discount)
            else:
                agent.store(state, action, reward, next_state, terminal, discount)

            state = next_state
            ep_reward += reward

        history["episode_rewards"].append(ep_reward)
        history["episode_profiles"].append(profile_key)

        if (ep + 1) % rl_config.VALIDATION_EVERY == 0 or ep == n_episodes - 1:
            checkpoint(ep + 1)

    agent.set_weights(best["weights"])
    history["best_episode"] = best["episode"]
    if verbose:
        print(f"  restored best checkpoint: episode {best['episode']} "
              f"(validation avg wait {best['wait']:.1f})")
    return history


def train_tabular(
    agent=None,
    n_episodes=None,
    profiles=None,
    episode_length=None,
    seed=None,
    verbose=True,
):
    """
    Train a TabularQAgent.

    Returns:
        (TabularQAgent, dict): trained agent and training history
        (see _train).
    """
    seed = seed if seed is not None else rl_config.SEED
    if agent is None:
        agent = TabularQAgent(seed=seed)
    history = _train(
        agent,
        use_discrete=True,
        n_episodes=n_episodes if n_episodes is not None else rl_config.TABULAR_EPISODES,
        profiles=profiles if profiles is not None else rl_config.PROFILES,
        episode_length=episode_length if episode_length is not None else rl_config.EPISODE_LENGTH,
        seed=seed,
        verbose=verbose,
    )
    return agent, history


def train_dqn(
    agent=None,
    n_episodes=None,
    profiles=None,
    episode_length=None,
    seed=None,
    verbose=True,
):
    """
    Train a DQNAgent over the raw 23-dim observation.

    Returns:
        (DQNAgent, dict): trained agent and training history (see _train).
    """
    seed = seed if seed is not None else rl_config.SEED
    if agent is None:
        agent = DQNAgent(
            obs_dim=rl_config.OBS_DIM,
            n_actions=rl_config.NUM_PHASES,
            seed=seed,
        )
    history = _train(
        agent,
        use_discrete=False,
        n_episodes=n_episodes if n_episodes is not None else rl_config.DQN_EPISODES,
        profiles=profiles if profiles is not None else rl_config.PROFILES,
        episode_length=episode_length if episode_length is not None else rl_config.EPISODE_LENGTH,
        seed=seed,
        verbose=verbose,
    )
    return agent, history
