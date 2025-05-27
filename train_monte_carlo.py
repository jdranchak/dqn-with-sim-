#!/usr/bin/env python3
"""
Training script for Monte Carlo DQN
"""

import argparse
import sys
from monte_carlo_agent import MonteCarloAgent

def main():
    parser = argparse.ArgumentParser(description='Train Monte Carlo DQN for Flappy Bird')
    parser.add_argument('--config', default='monte_carlo_flappybird', 
                       help='Hyperparameter configuration to use (default: monte_carlo_flappybird)')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode (load trained models and run with rendering)')
    parser.add_argument('--render', action='store_true', 
                       help='Render environment during training (slower)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 MONTE CARLO DQN TRAINING")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Mode: {'Testing' if args.test else 'Training'}")
    print("=" * 60)
    
    # Create the agent
    agent = MonteCarloAgent(args.config)
    
    try:
        if args.test:
            print("🎮 Running in test mode...")
            agent.run(is_training=False, render=True)
        else:
            print("🏋️ Starting training...")
            print("💡 The agent will:")
            print("   1. Learn a Q-network (like your original DQN)")
            print("   2. Learn an environment model")
            print("   3. Use Monte Carlo simulation to look ahead 5 steps")
            print("   4. Choose actions based on simulated outcomes")
            print("=" * 60)
            agent.run(is_training=True, render=args.render)
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("Models have been saved automatically during training")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 