# Assignment 19: Value Iteration in 4x4 GridWorld

This assignment implements value iteration for a 4x4 GridWorld problem where an agent needs to find optimal values for each state in a stochastic environment.

## Problem Description

- Environment: 4x4 GridWorld
- Start state: Top-left corner (state 0)
- Goal state: Bottom-right corner (state 15)
- Actions: Up, Down, Left, Right (equal probability 0.25)
- Rewards: -1 for each move, 0 for reaching the terminal state
- No obstacles in the grid
- Stochastic Environment: The agent moves in each direction with equal probability (0.25)

## Implementation Details

- Value iteration using the Bellman equation: V(s) = Σ P(s'|s,a)[R(s,a) + γV(s')]
- Discount factor (gamma) = 1.0
- Convergence threshold = 1e-4
- Values initialized to 0 for all states
- Iteration continues until maximum change in values < threshold
- Equal transition probabilities (P(s'|s,a) = 0.25) for all actions

## Running the Code

```bash
python value_iteration.py
```

##  Output

The final value function should look like this:

```
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]
```

## Dependencies

- NumPy

## Files

- `value_iteration.py`: Main implementation of the value iteration algorithm
- `README.md`: This file explaining the problem and solution # GridWorld-Value-Iteration
