"""
plot_rewards.py

Quick look at tabular Q-learning: trains for n episodes and plots the
per-episode reward (saved to reward_curve.png). Episodes rotate through
profiles of very different difficulty, so this curve is noisy; the honest
learning curve is the held-out validation curve produced by
run_experiments.py (images/G4_rl_training_curves.png).

Usage:
    python plot_rewards.py [n_episodes]

If matplotlib is not installed, falls back to printing a simple ASCII
summary of the curve instead of crashing.
"""
import sys

from config import rl as rl_config
from rl.train import train_tabular


def _ascii_curve(rewards, width=60):
    """Print an ASCII approximation of the reward curve."""
    if not rewards:
        return
    lo, hi = min(rewards), max(rewards)
    span = (hi - lo) or 1.0
    print("\nASCII reward curve (higher = better):")
    for i, r in enumerate(rewards):
        bar = int((r - lo) / span * (width - 10))
        tick = "*" * bar
        window = rewards[max(0, i - 4): i + 1]
        avg = sum(window) / len(window)
        print(f"ep {i + 1:>4} | {tick:<{width - 10}} ({r:>8.0f}) avg5={avg:>7.0f}")


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else rl_config.TABULAR_EPISODES

    agent, history = train_tabular(n_episodes=n_episodes, verbose=True)
    rewards = history["episode_rewards"]

    # Report the first/early/last reward to show the trend.
    print("\n=== Reward curve summary ===")
    print(f"episodes: {len(rewards)}")
    if rewards:
        print(f"first reward: {rewards[0]:.0f}")
        mid = len(rewards) // 2
        print(f"mid reward  : {rewards[mid]:.0f}")
        print(f"last reward : {rewards[-1]:.0f}")
        print(f"best reward : {max(rewards):.0f}")

    # Try to plot with matplotlib; fall back to ASCII otherwise.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        plt.plot(rewards, label="episode reward")
        # Simple moving average overlay.
        window = max(1, len(rewards) // 10)
        if len(rewards) >= window:
            import numpy as np

            kernel = np.ones(window) / window
            smooth = np.convolve(rewards, kernel, mode="valid")
            plt.plot(
                range(window - 1, len(rewards)),
                smooth,
                label=f"SMA-{window}",
                linewidth=2,
            )
        plt.xlabel("Episode")
        plt.ylabel("Episode reward (-queueing delay / 1000)")
        plt.title("Tabular Q-learning reward curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = "reward_curve.png"
        plt.tight_layout()
        plt.savefig(out)
        print(f"\nSaved plot to {out}")
    except ImportError:
        _ascii_curve(rewards)


if __name__ == "__main__":
    main()