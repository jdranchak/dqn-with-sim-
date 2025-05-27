import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os

from experience_replay import ReplayMemory
from monte_carlo_dqn import MonteCarloDQN, EnvironmentModel, MonteCarloPlanner

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # force cpu for stability


class MonteCarloAgent:
    """
    DQN Agent with Monte Carlo simulation for multi-step lookahead
    """

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        
        # Monte Carlo specific parameters
        self.use_monte_carlo = hyperparameters.get('use_monte_carlo', True)
        self.mc_simulations = hyperparameters.get('mc_simulations', 20)
        self.mc_depth = hyperparameters.get('mc_depth', 5)
        self.mc_frequency = hyperparameters.get('mc_frequency', 1)  # Use MC every N steps
        self.env_model_lr = hyperparameters.get('env_model_lr', 0.001)

        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.env_model_optimizer = None

        # Path to Run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_mc.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_mc.pt')
        self.ENV_MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_env_model.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_mc.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Monte Carlo DQN Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0]

        # List to keep track of rewards collected per episode
        rewards_per_episode = []

        # Create policy and target network
        policy_dqn = MonteCarloDQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        
        # Environment model for Monte Carlo simulation
        env_model = EnvironmentModel(num_states, num_actions).to(device)
        
        # Monte Carlo planner
        planner = MonteCarloPlanner(env_model, policy_dqn, 
                                  num_simulations=self.mc_simulations, 
                                  max_depth=self.mc_depth,
                                  discount=self.discount_factor_g)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)
            env_memory = ReplayMemory(self.replay_memory_size)  # For environment model training

            # Create the target network and make it identical to the policy network
            target_dqn = MonteCarloDQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Optimizers
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            self.env_model_optimizer = torch.optim.Adam(env_model.parameters(), lr=self.env_model_lr)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken
            step_count = 0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy and environment model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            if os.path.exists(self.ENV_MODEL_FILE):
                env_model.load_state_dict(torch.load(self.ENV_MODEL_FILE))

            # Switch models to evaluation mode
            policy_dqn.eval()
            env_model.eval()

        # Train INDEFINITELY
        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:

                # Select action
                if is_training and random.random() < epsilon:
                    # Random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # Use Monte Carlo planning for better decisions
                    if (self.use_monte_carlo and 
                        step_count % self.mc_frequency == 0 and 
                        len(env_memory) > self.mini_batch_size):
                        
                        # Use Monte Carlo planning
                        try:
                            mc_action, action_values = planner.select_best_action(state)
                            action = torch.tensor(mc_action, dtype=torch.int64, device=device)
                            
                            if is_training and episode % 100 == 0:
                                print(f"MC Action values: {action_values}")
                        except:
                            # Fallback to Q-network if MC fails
                            with torch.no_grad():
                                action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                    else:
                        # Use Q-network directly
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))
                    env_memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count += 1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/abs(best_reward)*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    torch.save(env_model.state_dict(), self.ENV_MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                # Train environment model
                if len(env_memory) > self.mini_batch_size:
                    env_mini_batch = env_memory.sample(self.mini_batch_size)
                    self.train_environment_model(env_mini_batch, env_model)

    def train_environment_model(self, mini_batch, env_model):
        """Train the environment model to predict next states, rewards, and done flags"""
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # Predict next state, reward, and done probability
        pred_next_states, pred_rewards, pred_done_probs = env_model(states, actions)

        # Compute losses
        state_loss = nn.MSELoss()(pred_next_states, new_states)
        reward_loss = nn.MSELoss()(pred_rewards.squeeze(), rewards)
        done_loss = nn.BCELoss()(pred_done_probs.squeeze(), terminations)

        total_loss = state_loss + reward_loss + done_loss

        # Update environment model
        self.env_model_optimizer.zero_grad()
        total_loss.backward()
        self.env_model_optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """Optimize policy network"""
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                          target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test Monte Carlo DQN model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = MonteCarloAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True) 