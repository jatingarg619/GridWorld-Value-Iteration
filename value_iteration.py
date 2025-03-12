import numpy as np

class GridWorld:
    def __init__(self, size=4, gamma=1.0, theta=1e-4):
        self.size = size
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros((size, size))
        self.rewards = -np.ones((size, size))
        self.rewards[-1, -1] = 0  # Terminal state has 0 reward
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.action_prob = 0.25  # Equal probability for all actions
        
    def is_valid_state(self, state):
        """Check if state (row, col) is valid."""
        row, col = state
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_next_state(self, state, action):
        """Get next state given current state and action."""
        next_row = state[0] + action[0]
        next_col = state[1] + action[1]
        
        if self.is_valid_state((next_row, next_col)):
            return (next_row, next_col)
        return state
    
    def value_iteration(self):
        """Perform value iteration until convergence."""
        iteration = 0
        while True:
            delta = 0
            v_new = np.copy(self.values)
            
            # For each state
            for i in range(self.size):
                for j in range(self.size):
                    if i == self.size-1 and j == self.size-1:  # Skip terminal state
                        continue
                    
                    # Calculate expected value over all actions
                    expected_value = 0
                    for action in self.actions:
                        next_state = self.get_next_state((i, j), action)
                        # V(s) = Σ P(s'|s,a)[R(s,a) + γV(s')]
                        # Since P(s'|s,a) = 0.25 for all actions
                        expected_value += self.action_prob * (self.rewards[i, j] + self.gamma * self.values[next_state])
                    
                    # Update value with expected value over all actions
                    v_new[i, j] = expected_value
                    
                    # Track maximum change
                    delta = max(delta, abs(v_new[i, j] - self.values[i, j]))
            
            # Update values
            self.values = v_new
            iteration += 1
            
            # Check for convergence
            if delta < self.theta:
                break
        
        return iteration

def main():
    # Initialize GridWorld
    grid = GridWorld(size=4, gamma=1.0, theta=1e-4)
    
    # Run value iteration
    iterations = grid.value_iteration()
    
    # Print results with better formatting
    print(f"\nValue iteration converged after {iterations} iterations.\n")
    print("Final value function:")
    np.set_printoptions(precision=8, suppress=True)  # Format numpy output
    print(grid.values)

if __name__ == "__main__":
    main() 