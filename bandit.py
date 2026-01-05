"""
Multi-armed bandit algorithm implementations.

This module provides implementations of several classic bandit algorithms:
- Epsilon-Greedy: Explores with probability epsilon, exploits otherwise
- Softmax (Boltzmann): Selects arms based on temperature-weighted probabilities
- UCB (Upper Confidence Bound): Balances exploration and exploitation using confidence bounds
- Thompson Sampling: Uses Bayesian posterior sampling for arm selection

Usage:
    uv run python bandit.py --policy epsilon-greedy --arms 10 --T 1000
"""

import argparse
import random
from typing import List

import numpy as np
from numpy import argmax, exp, log
from scipy.stats import beta
from tqdm import tqdm

from arm import Arm


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the bandit simulation.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Armed Bandit Algorithm Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--arms",
        type=int,
        default=100,
        help="Number of arms in the bandit",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000,
        help="Number of time steps per episode",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Exploration rate for epsilon-greedy policy",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature parameter for softmax policy",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["epsilon-greedy", "softmax", "ucb", "thompson"],
        default="epsilon-greedy",
        help="Bandit policy to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=100,
        help="Number of episodes to run",
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def reset_all_arms(arms: List[Arm]) -> None:
    """
    Reset all arms to their initial state.

    Args:
        arms: List of arms to reset.
    """
    for arm in arms:
        arm.reset()


class BanditAlgorithm:
    """
    Multi-armed bandit algorithm implementation.

    This class provides various bandit policies for selecting arms
    in a multi-armed bandit problem.

    Attributes:
        arms: List of Arm objects representing the bandit's arms.
        num_arms: Number of arms (K).
    """

    def __init__(self, arms: List[Arm]) -> None:
        """
        Initialize the bandit algorithm with a list of arms.

        Args:
            arms: List of Arm objects.
        """
        self.arms = arms
        self.num_arms = len(arms)

    def _get_empirical_means(self) -> List[float]:
        """
        Get empirical means for all arms.

        Returns:
            List of empirical mean values for each arm.
        """
        return [arm.empirical_mean for arm in self.arms]

    def _select_best_arm(self) -> int:
        """
        Select the arm with the highest empirical mean.

        Returns:
            Index of the arm with the highest empirical mean.
        """
        means = self._get_empirical_means()
        return int(argmax(means))

    def epsilon_greedy_policy(self, epsilon: float, time_steps: int = 100) -> int:
        """
        Run epsilon-greedy policy.

        This policy allocates a portion of time steps for exploration (playing
        each arm uniformly) and the remaining steps for exploitation (playing
        the arm with the highest empirical mean).

        Args:
            epsilon: Fraction of time steps to allocate to exploration.
            time_steps: Total number of time steps.

        Returns:
            Total reward accumulated over all time steps.
        """
        total_reward = 0
        explore_steps_per_arm = int(round(epsilon * time_steps / self.num_arms))
        exploit_steps = time_steps - explore_steps_per_arm * self.num_arms

        # Exploration phase: play each arm a fixed number of times
        for arm in self.arms:
            for _ in range(explore_steps_per_arm):
                total_reward += arm.play()

        # Exploitation phase: play the best arm
        best_arm_idx = self._select_best_arm()
        for _ in range(exploit_steps):
            total_reward += self.arms[best_arm_idx].play()

        return total_reward

    def softmax_policy(self, time_steps: int = 100, tau: float = 0.1) -> int:
        """
        Run softmax (Boltzmann exploration) policy.

        Arms are selected with probability proportional to exp(Q(a)/tau),
        where Q(a) is the empirical mean of arm a and tau is the temperature.

        Args:
            time_steps: Number of time steps to run.
            tau: Temperature parameter. Lower values favor exploitation,
                 higher values favor exploration.

        Returns:
            Total reward accumulated over all time steps.
        """
        total_reward = 0

        for _ in range(time_steps):
            means = self._get_empirical_means()

            # Calculate softmax probabilities
            exp_values = [exp(mean / tau) for mean in means]
            exp_sum = sum(exp_values)
            probabilities = [exp_val / exp_sum for exp_val in exp_values]

            # Sample arm according to probabilities
            cumulative_prob = np.cumsum(probabilities)
            random_value = random.random()

            selected_arm_idx = 0
            for idx, cum_prob in enumerate(cumulative_prob):
                if random_value < cum_prob:
                    selected_arm_idx = idx
                    break

            total_reward += self.arms[selected_arm_idx].play()

        return total_reward

    def ucb_policy(self, time_steps: int = 100) -> int:
        """
        Run Upper Confidence Bound (UCB1) policy.

        Each arm is selected based on its UCB value:
        UCB(a) = Q(a) + sqrt(2 * ln(t) / N(a))
        where Q(a) is the empirical mean and N(a) is the number of plays.

        Args:
            time_steps: Number of time steps to run.

        Returns:
            Total reward accumulated over all time steps.
        """
        total_reward = 0

        # Initial phase: play each arm once
        for arm in self.arms:
            total_reward += arm.play()

        # Main phase: use UCB values to select arms
        remaining_steps = time_steps - self.num_arms
        for t in range(self.num_arms, self.num_arms + remaining_steps):
            ucb_values = []

            for arm in self.arms:
                num_plays = arm.success + arm.fail
                if num_plays == 0:
                    # Should not happen after initial phase, but handle safely
                    ucb_values.append(float("inf"))
                else:
                    exploration_bonus = (2 * log(t) / num_plays) ** 0.5
                    ucb_values.append(arm.empirical_mean + exploration_bonus)

            best_arm_idx = int(argmax(ucb_values))
            total_reward += self.arms[best_arm_idx].play()

        return total_reward

    def thompson_sampling_policy(self, time_steps: int = 100) -> int:
        """
        Run Thompson Sampling policy.

        Each arm's reward probability is modeled as a Beta distribution.
        At each step, we sample from each arm's posterior and select
        the arm with the highest sampled value.

        Args:
            time_steps: Number of time steps to run.

        Returns:
            Total reward accumulated over all time steps.
        """
        total_reward = 0

        for _ in range(time_steps):
            # Sample from Beta posterior for each arm
            sampled_values = [
                beta.rvs(arm.success + 1, arm.fail + 1) for arm in self.arms
            ]

            # Select arm with highest sampled value
            selected_arm_idx = int(argmax(sampled_values))
            total_reward += self.arms[selected_arm_idx].play()

        return total_reward


def main() -> None:
    """Main entry point for the bandit simulation."""
    args = parse_arguments()

    # Set random seed for reproducibility
    if args.seed:
        set_random_seed(args.seed)

    # Create arms with random success probabilities
    arms = [Arm(probability=np.random.rand()) for _ in range(args.arms)]
    bandit = BanditAlgorithm(arms)

    # Run simulation
    total_rewards = 0
    num_episodes = args.num_plays

    for _ in tqdm(range(num_episodes), desc=f"Running {args.policy}"):
        reset_all_arms(arms)

        match args.policy:
            case "epsilon-greedy":
                reward = bandit.epsilon_greedy_policy(args.epsilon, args.T)
            case "softmax":
                reward = bandit.softmax_policy(args.T, args.tau)
            case "ucb":
                reward = bandit.ucb_policy(args.T)
            case "thompson":
                reward = bandit.thompson_sampling_policy(args.T)

        total_rewards += reward

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Policy: {args.policy}")
    print(f"Number of arms: {args.arms}")
    print(f"Time steps per episode: {args.T}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward per episode: {total_rewards / num_episodes:.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
