# Basic Bandit

A Python implementation of classic multi-armed bandit algorithms for reinforcement learning.

## Overview

This project implements several well-known bandit algorithms that balance exploration and exploitation in sequential decision-making problems:

- **Epsilon-Greedy**: Explores with probability ε, exploits the best known arm otherwise
- **Softmax (Boltzmann)**: Selects arms with probability proportional to their estimated value
- **UCB (Upper Confidence Bound)**: Uses optimistic estimates based on confidence intervals
- **Thompson Sampling**: Uses Bayesian posterior sampling for arm selection

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd basic-bandit
```

2. Install dependencies using uv:

```bash
uv sync
```

## Usage

Run the bandit simulation with various policies:

```bash
# Epsilon-greedy policy (default)
uv run python bandit.py --policy epsilon-greedy --arms 100 --T 1000

# Softmax policy
uv run python bandit.py --policy softmax --tau 0.1

# UCB policy
uv run python bandit.py --policy ucb

# Thompson Sampling
uv run python bandit.py --policy thompson
```

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arms` | int | 100 | Number of arms in the bandit |
| `--T` | int | 1000 | Number of time steps per episode |
| `--epsilon` | float | 0.2 | Exploration rate for epsilon-greedy |
| `--tau` | float | 0.1 | Temperature for softmax policy |
| `--policy` | str | epsilon-greedy | Policy to use (epsilon-greedy, softmax, ucb, thompson) |
| `--num_plays` | int | 100 | Number of episodes to run |
| `--seed` | int | 1 | Random seed for reproducibility |
| `--verbose` | flag | False | Enable verbose output |

## Algorithm Details

### Epsilon-Greedy

The epsilon-greedy policy allocates a fraction ε of time steps for exploration (playing each arm uniformly) and the remaining (1-ε) fraction for exploitation (playing the best empirical arm).

### Softmax (Boltzmann Exploration)

Arms are selected with probability:

```
P(a) = exp(Q(a)/τ) / Σexp(Q(i)/τ)
```

where `Q(a)` is the empirical mean of arm `a` and `τ` (tau) is the temperature parameter.

### Upper Confidence Bound (UCB1)

Each arm is evaluated using:

```
UCB(a) = Q(a) + √(2 ln(t) / N(a))
```

where `t` is the current time step and `N(a)` is the number of times arm `a` has been played.

### Thompson Sampling

Models each arm's reward probability as a Beta distribution. At each step:
1. Sample from each arm's Beta(success+1, fail+1) posterior
2. Select the arm with the highest sampled value

## Project Structure

```
basic-bandit/
├── arm.py           # Arm class representing a single bandit arm
├── bandit.py        # Main bandit algorithm implementations
├── pyproject.toml   # Project configuration and dependencies
└── README.md        # This file
```

## License

MIT License
