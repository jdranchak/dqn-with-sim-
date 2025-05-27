import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import copy

class MonteCarloDQN(nn.Module):
    """
    DQN with Monte Carlo simulation for multi-step lookahead
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(MonteCarloDQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main Q-network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q


class EnvironmentModel(nn.Module):
    """
    Simple neural network to model environment dynamics for Monte Carlo simulation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EnvironmentModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input: current state + action (one-hot)
        input_dim = state_dim + action_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output: next state + reward + done probability
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """
        Predict next state, reward, and done probability
        """
        # One-hot encode action
        if action.dim() == 1:
            action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            action_onehot = action
            
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        next_state = self.next_state_head(x)
        reward = self.reward_head(x)
        done_prob = torch.sigmoid(self.done_head(x))
        
        return next_state, reward, done_prob
    
    def sample_transition(self, state, action):
        """
        Sample a transition for Monte Carlo simulation
        """
        with torch.no_grad():
            next_state, reward, done_prob = self.forward(state, action)
            
            # Sample done based on probability
            done = torch.bernoulli(done_prob).bool()
            
            return next_state, reward.squeeze(), done.squeeze()


class MonteCarloPlanner:
    """
    Monte Carlo planner that simulates multiple rollouts to evaluate actions
    """
    def __init__(self, env_model, q_network, num_simulations=50, max_depth=10, discount=0.99):
        self.env_model = env_model
        self.q_network = q_network
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.discount = discount
        
    def simulate_rollout(self, state, action, depth=0):
        """
        Simulate a single rollout from the given state and action
        """
        if depth >= self.max_depth:
            # Use Q-network to estimate value at max depth
            with torch.no_grad():
                return self.q_network(state.unsqueeze(0)).max().item()
        
        # Simulate one step
        next_state, reward, done = self.env_model.sample_transition(state.unsqueeze(0), torch.tensor([action]))
        next_state = next_state.squeeze(0)
        
        if done:
            return reward.item()
        
        # Choose next action using epsilon-greedy on Q-network
        if random.random() < 0.1:  # Small exploration during simulation
            next_action = random.randint(0, self.env_model.action_dim - 1)
        else:
            with torch.no_grad():
                next_action = self.q_network(next_state.unsqueeze(0)).argmax().item()
        
        # Recursive rollout
        future_value = self.simulate_rollout(next_state, next_action, depth + 1)
        
        return reward.item() + self.discount * future_value
    
    def evaluate_action(self, state, action):
        """
        Evaluate an action using Monte Carlo simulation
        """
        total_value = 0.0
        
        for _ in range(self.num_simulations):
            value = self.simulate_rollout(state, action)
            total_value += value
            
        return total_value / self.num_simulations
    
    def select_best_action(self, state, available_actions=None):
        """
        Select the best action using Monte Carlo evaluation
        """
        if available_actions is None:
            available_actions = list(range(self.env_model.action_dim))
        
        action_values = {}
        
        for action in available_actions:
            action_values[action] = self.evaluate_action(state, action)
        
        # Return action with highest expected value
        best_action = max(action_values.keys(), key=lambda a: action_values[a])
        return best_action, action_values


if __name__ == '__main__':
    # Test the networks
    state_dim = 12
    action_dim = 2
    batch_size = 10
    
    # Test MonteCarloDQN
    dqn = MonteCarloDQN(state_dim, action_dim)
    state = torch.randn(batch_size, state_dim)
    q_values = dqn(state)
    print(f"DQN output shape: {q_values.shape}")
    
    # Test EnvironmentModel
    env_model = EnvironmentModel(state_dim, action_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    next_state, reward, done_prob = env_model(state, actions)
    print(f"Environment model outputs - Next state: {next_state.shape}, Reward: {reward.shape}, Done prob: {done_prob.shape}")
    
    # Test MonteCarloPlanner
    planner = MonteCarloPlanner(env_model, dqn, num_simulations=10, max_depth=5)
    test_state = torch.randn(state_dim)
    best_action, action_values = planner.select_best_action(test_state)
    print(f"Best action: {best_action}, Action values: {action_values}") 