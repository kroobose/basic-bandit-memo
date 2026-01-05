"""
Arm module for multi-armed bandit problem.

This module provides the Arm class representing a single arm in a bandit problem.
Each arm has a fixed success probability and tracks its play history.
"""

from numpy.random import binomial


class Arm:
    """
    Represents a single arm in a multi-armed bandit problem.

    Each arm has a fixed probability of success (reward = 1) and tracks
    the number of successes, failures, and total plays.

    Attributes:
        success (int): Number of successful plays (reward = 1).
        fail (int): Number of failed plays (reward = 0).
        play_count (int): Total number of times this arm has been played.
    """

    def __init__(self, probability: float) -> None:
        """
        Initialize an arm with a given success probability.

        Args:
            probability: The probability of success (reward = 1) when this arm
                        is played. Must be between 0 and 1.
        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")

        self._probability = probability
        self.success = 0
        self.fail = 0
        self.play_count = 0

    @property
    def probability(self) -> float:
        """Return the true success probability of this arm."""
        return self._probability

    @property
    def empirical_mean(self) -> float:
        """
        Calculate the empirical success ratio based on play history.

        Returns:
            The ratio of successes to total plays, or 0.0 if never played.
        """
        total_plays = self.success + self.fail
        if total_plays == 0:
            return 0.0
        return self.success / total_plays

    def reset(self) -> None:
        """Reset all counters to their initial state."""
        self.success = 0
        self.fail = 0
        self.play_count = 0

    def play(self) -> int:
        """
        Play this arm once and return the reward.

        Simulates pulling the arm using a binomial distribution with n=1.
        Updates the internal counters based on the result.

        Returns:
            1 if the play was successful, 0 otherwise.
        """
        self.play_count += 1
        result = binomial(n=1, p=self._probability)

        if result == 1:
            self.success += 1
        else:
            self.fail += 1

        return result

    def __repr__(self) -> str:
        """Return a string representation of the arm."""
        return (
            f"Arm(p={self._probability:.3f}, "
            f"plays={self.play_count}, "
            f"success_rate={self.empirical_mean:.3f})"
        )
