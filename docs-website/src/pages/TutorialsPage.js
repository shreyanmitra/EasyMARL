import React, { useState } from 'react';
import { 
  Play, 
  Clock, 
  Zap, 
  BookOpen, 
  CheckCircle, 
  ArrowRight,
  Monitor,
  Settings,
  TrendingUp,
  Brain,
  Target
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';

const TutorialsPage = () => {
  const [selectedTutorial, setSelectedTutorial] = useState(null);

  const tutorials = [
    {
      id: 'quickstart',
      title: 'Quick Start Guide',
      level: 'Beginner',
      duration: '10 minutes',
      icon: <Play className="w-6 h-6" />,
      description: 'Get up and running with EasyMARL in minutes. Your first multi-agent training.',
      objectives: [
        'Install EasyMARL',
        'Run your first MARL algorithm',
        'Understand the basic workflow',
        'Visualize training results'
      ],
      content: {
        steps: [
          {
            title: 'Installation',
            code: `# Install EasyMARL
pip install easymarl

# Or install from source
pip install git+https://github.com/shreyanmitra/EasyMARL.git`,
            explanation: 'Install EasyMARL using pip. This includes all core algorithms and environments.'
          },
          {
            title: 'First Training Session',
            code: `from easymarl.controllers import UnifiedMultiAgentController

# Create controller with educational mode
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-6x6',
    algorithm='ippo',
    n_envs=1,  # Single environment for learning
    educational_mode=True  # Detailed explanations
)

# Train agents
results = controller.train(total_episodes=100)
print(f"Training complete! Final reward: {results['final_average_reward']:.3f}")`,
            explanation: 'This creates a simple multi-agent environment with 4 IPPO agents. Educational mode provides detailed explanations of what\'s happening.'
          },
          {
            title: 'Evaluate Performance',
            code: `# Evaluate trained agents
eval_results = controller.evaluate(num_episodes=10, save_videos=True)
print(f"Evaluation reward: {eval_results['average_reward']:.3f}")
print(f"Success rate: {eval_results['success_rate']:.1%}")`,
            explanation: 'Test your trained agents and optionally save videos to see how they behave.'
          }
        ]
      }
    },
    {
      id: 'educational-mode',
      title: 'Educational Mode Deep Dive',
      level: 'Beginner',
      duration: '20 minutes',
      icon: <BookOpen className="w-6 h-6" />,
      description: 'Learn MARL concepts through EasyMARL\'s educational features.',
      objectives: [
        'Understand educational mode features',
        'Learn MARL concepts step-by-step',
        'Interpret training metrics',
        'Debug common issues'
      ],
      content: {
        steps: [
          {
            title: 'Enabling Educational Mode',
            code: `controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8',
    algorithm='ippo',
    educational_mode=True,  # Enable learning features
    n_envs=1,              # Single env for clarity
    config={
        'max_steps': 500,     # Longer episodes
        'log_interval': 1     # Log every episode
    }
)`,
            explanation: 'Educational mode provides step-by-step explanations, longer episodes for observation, and frequent logging.'
          },
          {
            title: 'Understanding Training Output',
            code: `# Educational mode will print:
# 🎓 Episode 1 Walkthrough:
#    - Agents will observe the environment state
#    - Each agent will choose actions based on current policy
#    - Environment will update and provide rewards
#    - Algorithm will learn from this experience

results = controller.train(total_episodes=50)`,
            explanation: 'Educational mode explains each step of the learning process, helping you understand what agents are doing.'
          },
          {
            title: 'Interpreting Metrics',
            code: `# Understanding the output:
# 📊 Episode 10/50 Summary:
#    Recent Avg Reward: 2.340
#    Recent Avg Length: 87.5 steps
#    Episode Duration: 0.45s
#    Training Speed: 133.3 episodes/min

# 🎓 Understanding the Metrics:
#    - Reward shows how well agents are performing
#    - Length shows how long episodes last
#    - Improving performance trend`,
            explanation: 'Learn to read and interpret the training metrics to understand agent performance and learning progress.'
          }
        ]
      }
    },
    {
      id: 'vectorized-training',
      title: 'High-Performance Vectorized Training',
      level: 'Intermediate',
      duration: '25 minutes',
      icon: <Zap className="w-6 h-6" />,
      description: 'Scale up your training with vectorized environments for 8x speedup.',
      objectives: [
        'Understand vectorization benefits',
        'Configure parallel environments',
        'Optimize training performance',
        'Monitor system resources'
      ],
      content: {
        steps: [
          {
            title: 'Vectorized Setup',
            code: `# High-performance training setup
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=8,  # 8 parallel environments
    enable_performance_monitoring=True,
    config={
        'batch_size': 64,
        'update_frequency': 4
    }
)`,
            explanation: 'Vectorized training runs multiple environments in parallel, dramatically reducing training time.'
          },
          {
            title: 'Performance Comparison',
            code: `# Single environment training
single_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=1
)

# Vectorized training (8x faster data collection)
vectorized_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='qmix',
    n_envs=8  # 8x speedup
)

# Train both and compare times
import time
start = time.time()
vectorized_results = vectorized_controller.train(total_episodes=1000)
vectorized_time = time.time() - start

print(f"Vectorized training completed in {vectorized_time:.1f}s")`,
            explanation: 'Vectorized training provides significant speedup by collecting data from multiple environments simultaneously.'
          },
          {
            title: 'Advanced Configuration',
            code: `# Advanced vectorized setup
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Advanced-15x15',
    algorithm='maddpg',
    n_envs=16,  # Scale up further
    enable_advanced_tracking=True,
    enable_performance_monitoring=True,
    config={
        'normalize_observations': True,
        'domain_randomization': True,
        'batch_size': 128
    }
)`,
            explanation: 'Advanced configurations can include observation normalization, domain randomization, and larger batch sizes for better performance.'
          }
        ]
      }
    },
    {
      id: 'algorithm-selection',
      title: 'Choosing the Right Algorithm',
      level: 'Intermediate',
      duration: '30 minutes',
      icon: <Brain className="w-6 h-6" />,
      description: 'Learn when and why to use different MARL algorithms.',
      objectives: [
        'Understand algorithm categories',
        'Match algorithms to problems',
        'Compare performance characteristics',
        'Handle different action spaces'
      ],
      content: {
        steps: [
          {
            title: 'Algorithm Categories',
            code: `# Policy-Based: Good for continuous actions and exploration
ippo_controller = UnifiedMultiAgentController(
    env_name='Continuous-Control-Env',
    algorithm='ippo',  # Independent learning
    n_envs=8
)

# Value-Based: Good for discrete actions and coordination
qmix_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',  # Value factorization
    n_envs=8
)

# Actor-Critic: Good for mixed scenarios
maddpg_controller = UnifiedMultiAgentController(
    env_name='Mixed-Action-Env',
    algorithm='maddpg',  # Centralized training
    n_envs=4
)`,
            explanation: 'Different algorithm types excel in different scenarios. Choose based on your action space and coordination requirements.'
          },
          {
            title: 'Problem-Algorithm Matching',
            code: `# Simple coordination tasks
simple_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Simple-6x6',
    algorithm='vdn',  # Simple value decomposition
    educational_mode=True
)

# Complex coordination with credit assignment
complex_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-12x12',
    algorithm='coma',  # Counterfactual reasoning
    n_envs=4
)

# Large-scale systems
scale_controller = UnifiedMultiAgentController(
    env_name='LargeScale-Environment',
    algorithm='mfpo',  # Mean field approximation
    n_envs=8
)`,
            explanation: 'Match algorithm complexity to problem complexity. Start simple and scale up as needed.'
          },
          {
            title: 'Performance Comparison',
            code: `# Compare multiple algorithms
algorithms = ['ippo', 'qmix', 'maddpg']
results = {}

for algo in algorithms:
    controller = UnifiedMultiAgentController(
        env_name='MultiGrid-Benchmark-8x8',
        algorithm=algo,
        n_envs=8
    )
    
    result = controller.train(total_episodes=1000)
    eval_result = controller.evaluate(num_episodes=20)
    
    results[algo] = {
        'final_reward': eval_result['average_reward'],
        'training_time': result['total_training_time'],
        'convergence_episode': result.get('convergence_episode', 1000)
    }

# Print comparison
for algo, metrics in results.items():
    print(f"{algo}: Reward={metrics['final_reward']:.3f}, "
          f"Time={metrics['training_time']:.1f}s")`,
            explanation: 'Empirically compare algorithms on your specific problem to find the best fit.'
          }
        ]
      }
    },
    {
      id: 'custom-environments',
      title: 'Creating Custom Environments',
      level: 'Advanced',
      duration: '45 minutes',
      icon: <Settings className="w-6 h-6" />,
      description: 'Build custom multi-agent environments for your specific use case.',
      objectives: [
        'Understand environment structure',
        'Implement custom observations',
        'Design reward functions',
        'Integrate with EasyMARL'
      ],
      content: {
        steps: [
          {
            title: 'Basic Environment Structure',
            code: `import gym
import numpy as np
from easymarl.environments import MultiAgentEnv

class CustomMultiAgentEnv(MultiAgentEnv):
    def __init__(self, n_agents=4, grid_size=8):
        super().__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 3), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)  # 4 directions + stay
        
    def reset(self):
        # Initialize environment state
        self.state = np.zeros((self.grid_size, self.grid_size, 3))
        # Return initial observations for all agents
        return self.get_observations()
    
    def step(self, actions):
        # Execute actions and update environment
        rewards = self.compute_rewards(actions)
        observations = self.get_observations()
        done = self.is_done()
        info = self.get_info()
        
        return observations, rewards, done, info`,
            explanation: 'Custom environments inherit from MultiAgentEnv and implement the standard gym interface with multi-agent extensions.'
          },
          {
            title: 'Reward Function Design',
            code: `def compute_rewards(self, actions):
    """Design rewards to encourage desired behaviors"""
    rewards = np.zeros(self.n_agents)
    
    for i, action in enumerate(actions):
        # Individual task completion reward
        if self.agent_reached_goal(i):
            rewards[i] += 10.0
        
        # Cooperation bonus
        nearby_agents = self.count_nearby_agents(i)
        rewards[i] += nearby_agents * 0.5
        
        # Penalty for collisions
        if self.agent_collision(i):
            rewards[i] -= 2.0
            
        # Small step penalty to encourage efficiency
        rewards[i] -= 0.01
    
    # Team reward component
    team_bonus = self.compute_team_performance()
    rewards += team_bonus
    
    return rewards`,
            explanation: 'Design reward functions that balance individual and team objectives. Include shaping rewards to guide learning.'
          },
          {
            title: 'Integration with EasyMARL',
            code: `# Register your custom environment
from easymarl.environments import register_env

register_env('CustomMultiAgent-v0', CustomMultiAgentEnv)

# Use with UnifiedMultiAgentController
controller = UnifiedMultiAgentController(
    env_name='CustomMultiAgent-v0',
    algorithm='qmix',
    n_envs=8,
    config={
        'env_config': {
            'n_agents': 6,
            'grid_size': 10
        }
    }
)

# Train on your custom environment
results = controller.train(total_episodes=2000)`,
            explanation: 'Register your environment with EasyMARL to use it seamlessly with all algorithms and features.'
          }
        ]
      }
    },
    {
      id: 'algorithm-configuration',
      title: 'Algorithm Configuration Templates',
      level: 'Intermediate',
      duration: '35 minutes',
      icon: <Settings className="w-6 h-6" />,
      description: 'Customize pre-defined algorithms with advanced parameter templates.',
      objectives: [
        'Understand algorithm configuration system',
        'Customize pre-defined algorithm templates',
        'Configure cooperative vs competitive scenarios',
        'Optimize algorithm hyperparameters'
      ],
      content: {
        steps: [
          {
            title: 'Basic Algorithm Configuration',
            code: `# Customize QMIX algorithm parameters
from easymarl.controllers import UnifiedMultiAgentController

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Custom-8x8',
    algorithm='qmix',  # Use existing algorithm template
    config={
        'n_agents': 6,                    # Custom agent count
        'learning_rate': 3e-4,           # Custom learning rate
        'batch_size': 256,               # Larger batch for stability
        'exploration_epsilon': 0.1,      # Exploration rate
        'mixer_hidden_dim': 128,         # QMIX-specific parameter
        'centralized_training': True,     # Training paradigm
        'gamma': 0.99,                   # Discount factor
        'target_update_freq': 200        # Target network updates
    }
)

# Train with customized parameters
results = controller.train(total_episodes=2000)`,
            explanation: 'Algorithm templates allow extensive customization while maintaining proven architecture. Each algorithm has specific parameters you can tune.'
          },
          {
            title: 'Cooperative Scenario Configuration',
            code: `# Setup for cooperative multi-agent tasks
cooperative_config = {
    'n_agents': 4,
    'cooperative': True,              # Enable cooperation mode
    'shared_rewards': True,           # All agents get same reward
    'communication': True,            # Enable agent communication
    'coordination_required': True,     # Environment requires coordination
    'centralized_training': True,     # Centralized training approach
    'parameter_sharing': 'full',      # Share parameters between agents
    'reward_scaling': 1.0,           # Reward normalization
    'value_loss_coef': 0.5,          # Value function coefficient
    'entropy_coef': 0.01             # Exploration encouragement
}

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-Custom',
    algorithm='mappo',  # Good for cooperation
    config=cooperative_config
)

# Monitor cooperation metrics
results = controller.train(
    total_episodes=3000,
    log_cooperation_metrics=True
)`,
            explanation: 'Cooperative scenarios require shared objectives, communication, and coordinated learning. MAPPO and QMIX excel in these settings.'
          },
          {
            title: 'Competitive Scenario Configuration', 
            code: `# Setup for competitive multi-agent scenarios
competitive_config = {
    'n_agents': 3,
    'competitive': True,             # Enable competition mode
    'zero_sum': True,               # Zero-sum game dynamics
    'individual_rewards': True,      # Each agent has own reward
    'limited_resources': True,       # Scarcity creates competition
    'adversarial_training': True,    # Train against each other
    'self_play': True,              # Agents learn from self-play
    'nash_equilibrium': True,       # Seek Nash equilibrium
    'fictitious_play': True,        # Use fictitious self-play
    'exploitation_prevention': 0.1   # Prevent exploitation
}

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Competitive-Custom',
    algorithm='nfsp',  # Neural Fictitious Self-Play for competition
    config=competitive_config
)

# Monitor competitive balance
results = controller.train(
    total_episodes=5000,
    log_competitive_metrics=True
)`,
            explanation: 'Competitive scenarios require different algorithms like NFSP, Nash-Q, or Minimax-Q that handle adversarial learning and game theory.'
          },
          {
            title: 'Mixed Cooperative-Competitive Configuration',
            code: `# Mixed scenarios with both cooperation and competition
mixed_config = {
    'n_agents': 6,
    'team_based': True,              # Teams compete, teammates cooperate
    'teams': [[0, 1, 2], [3, 4, 5]], # Agent team assignments
    'intra_team_cooperation': True,   # Cooperation within teams
    'inter_team_competition': True,   # Competition between teams
    'team_reward_sharing': True,     # Teams share rewards internally
    'communication_within_team': True, # Team communication allowed
    'heterogeneous_agents': True,    # Different agent types per team
    'dynamic_teams': False,          # Fixed team composition
    'tournament_style': True         # Tournament-based evaluation
}

controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Teams-6v6',
    algorithm='maddpg',  # Handles mixed scenarios well
    config=mixed_config
)

results = controller.train(
    total_episodes=4000,
    evaluate_teams_separately=True
)`,
            explanation: 'Mixed scenarios combine cooperation and competition. MADDPG and COMA handle these complex dynamics effectively.'
          },
          {
            title: 'Advanced Hyperparameter Templates',
            code: `# Algorithm-specific advanced configurations

# QMIX with advanced options
qmix_advanced = {
    'algorithm': 'qmix',
    'mixer_hidden_dim': 256,         # Larger mixing network
    'hypernet_layers': 3,            # Hypernetwork complexity
    'monotonic_constraint': True,     # Enforce monotonicity
    'dueling_networks': True,        # Use dueling architecture
    'double_q_learning': True,       # Double Q-learning
    'prioritized_replay': True,      # Prioritized experience replay
    'multi_step_returns': 3,        # N-step returns
    'td_lambda': 0.8                # TD-lambda for eligibility traces
}

# MAPPO with advanced options  
mappo_advanced = {
    'algorithm': 'mappo',
    'ppo_epochs': 10,               # PPO update epochs
    'num_mini_batches': 8,          # Mini-batch divisions
    'clip_param': 0.2,              # PPO clipping parameter
    'value_clip_param': 0.2,        # Value function clipping
    'max_grad_norm': 10.0,          # Gradient clipping
    'gae_lambda': 0.95,             # GAE lambda parameter
    'use_linear_lr_decay': True,    # Learning rate scheduling
    'use_popart': True,             # PopArt value normalization
    'use_valuenorm': True,          # Value normalization
    'use_feature_normalization': True # Feature normalization
}

# Use advanced configurations
advanced_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Complex-15x15',
    algorithm='qmix',
    config=qmix_advanced,
    n_envs=16  # More parallel environments for advanced training
)`,
            explanation: 'Advanced hyperparameter templates give you fine-grained control over algorithm behavior, neural network architecture, and training dynamics.'
          }
        ]
      }
    },
    {
      id: 'custom-algorithms',
      title: 'Creating Custom Algorithms',
      level: 'Advanced',
      duration: '60 minutes',
      icon: <Brain className="w-6 h-6" />,
      description: 'Implement completely new MARL algorithms using EasyMARL base classes.',
      objectives: [
        'Understand algorithm architecture',
        'Implement custom agents and algorithms',
        'Integrate with AlgorithmFactory',
        'Add GUI support for custom algorithms'
      ],
      content: {
        steps: [
          {
            title: 'Custom Agent Implementation',
            code: `# Create a custom MARL agent
from easymarl.algorithms.base import MARLAgent
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomAgent(MARLAgent):
    def __init__(self, agent_id: int, obs_space: dict, action_space: int, config: dict):
        super().__init__(agent_id, obs_space, action_space, config)
        
        # Custom neural network architecture
        self.hidden_dim = config.get('hidden_dim', 128)
        self.learning_rate = config.get('learning_rate', 1e-3)
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_space['vector'], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_space)
        )
        
        # Value network (for actor-critic)
        self.value_net = nn.Sequential(
            nn.Linear(obs_space['vector'], self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=self.learning_rate
        )
        
        # Experience buffer
        self.memory = []
        
    def get_action(self, observation: dict, training: bool = True):
        """Select action using custom policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation['vector'])
            action_logits = self.policy_net(obs_tensor)
            
            if training:
                # Sample from policy distribution
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[action]).item()
            else:
                # Deterministic action selection
                action = torch.argmax(action_logits).item()
                log_prob = 0.0
                
        return action, log_prob
    
    def update(self, batch_data: dict) -> dict:
        """Custom learning update implementation"""
        observations = torch.FloatTensor(batch_data['observations'])
        actions = torch.LongTensor(batch_data['actions'])
        rewards = torch.FloatTensor(batch_data['rewards'])
        next_observations = torch.FloatTensor(batch_data['next_observations'])
        dones = torch.BoolTensor(batch_data['dones'])
        
        # Compute policy loss
        action_logits = self.policy_net(observations)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        
        # Compute value estimates
        values = self.value_net(observations)
        next_values = self.value_net(next_observations)
        
        # Compute advantages (simple TD error)
        targets = rewards + 0.99 * next_values.squeeze() * (~dones)
        advantages = targets - values.squeeze()
        
        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs.squeeze() * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), targets.detach())
        
        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def save_model(self, filepath: str):
        """Save agent model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])`,
            explanation: 'Custom agents inherit from MARLAgent and implement the core methods: get_action(), update(), save_model(), and load_model().'
          },
          {
            title: 'Custom Algorithm Implementation',
            code: `# Create a custom MARL algorithm
from easymarl.algorithms.base import MARLAlgorithm
import numpy as np

class MyCustomAlgorithm(MARLAlgorithm):
    def __init__(self, env, config: dict, device):
        super().__init__(env, config, device)
        self.n_agents = env.n_agents
        self.obs_space = env.observation_space
        self.action_space = env.action_space.n
        self.config = config
        
        # Algorithm-specific parameters
        self.rollout_length = config.get('rollout_length', 100)
        self.batch_size = config.get('batch_size', 32)
        self.update_frequency = config.get('update_frequency', 10)
        
        # Create agents
        self._create_agents()
        
        # Training state
        self.step_count = 0
        
    def _create_agents(self):
        """Create custom agents"""
        self.agents = []
        for i in range(self.n_agents):
            agent = MyCustomAgent(
                agent_id=i,
                obs_space=self.obs_space,
                action_space=self.action_space,
                config=self.config
            )
            self.agents.append(agent)
    
    def collect_rollout(self, env) -> dict:
        """Collect experience data from environment"""
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': []
        }
        
        obs = env.reset()
        for step in range(self.rollout_length):
            # Get actions from all agents
            actions = []
            log_probs = []
            for i, agent in enumerate(self.agents):
                action, log_prob = agent.get_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
            
            # Execute actions in environment
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Store experience
            rollout_data['observations'].append(obs)
            rollout_data['actions'].append(actions)
            rollout_data['rewards'].append(rewards)
            rollout_data['next_observations'].append(next_obs)
            rollout_data['dones'].append(dones)
            rollout_data['infos'].append(infos)
            
            obs = next_obs
            if any(dones):
                obs = env.reset()
        
        return rollout_data
    
    def train_step(self, rollout_data: dict) -> dict:
        """Perform training step using collected data"""
        metrics = {}
        
        # Process rollout data for each agent
        for i, agent in enumerate(self.agents):
            # Extract agent-specific data
            agent_data = {
                'observations': [obs[i] for obs in rollout_data['observations']],
                'actions': [actions[i] for actions in rollout_data['actions']],
                'rewards': [rewards[i] for rewards in rollout_data['rewards']],
                'next_observations': [obs[i] for obs in rollout_data['next_observations']],
                'dones': rollout_data['dones']
            }
            
            # Update agent
            if len(agent_data['observations']) >= self.batch_size:
                # Sample random batch
                indices = np.random.choice(
                    len(agent_data['observations']), 
                    self.batch_size, 
                    replace=False
                )
                
                batch_data = {
                    'observations': [agent_data['observations'][idx]['vector'] for idx in indices],
                    'actions': [agent_data['actions'][idx] for idx in indices],
                    'rewards': [agent_data['rewards'][idx] for idx in indices],
                    'next_observations': [agent_data['next_observations'][idx]['vector'] for idx in indices],
                    'dones': [agent_data['dones'][idx] for idx in indices]
                }
                
                agent_metrics = agent.update(batch_data)
                metrics[f'agent_{i}'] = agent_metrics
        
        self.step_count += 1
        return metrics
    
    def save_algorithm(self, filepath: str):
        """Save the entire algorithm"""
        for i, agent in enumerate(self.agents):
            agent_path = f"{filepath}_agent_{i}.pt"
            agent.save_model(agent_path)
    
    def load_algorithm(self, filepath: str):
        """Load the entire algorithm"""
        for i, agent in enumerate(self.agents):
            agent_path = f"{filepath}_agent_{i}.pt"
            agent.load_model(agent_path)`,
            explanation: 'Custom algorithms inherit from MARLAlgorithm and coordinate multiple agents through collect_rollout() and train_step() methods.'
          },
          {
            title: 'Register Custom Algorithm',
            code: `# Register your custom algorithm with EasyMARL
from easymarl.algorithms import AlgorithmFactory

# Add to algorithm factory
AlgorithmFactory.ALGORITHMS['my_custom'] = MyCustomAlgorithm

# Add configuration template
from easymarl.core.config_manager import ConfigManager

def _my_custom_template():
    return {
        'algorithm_type': 'policy_based',
        'learning_paradigm': 'independent_learning',
        'action_space': 'discrete',
        'default_params': {
            'learning_rate': 1e-3,
            'hidden_dim': 128,
            'batch_size': 32,
            'rollout_length': 100,
            'update_frequency': 10,
            'gamma': 0.99
        },
        'network_config': {
            'hidden_dims': [128, 128],
            'activation': 'relu',
            'parameter_sharing': 'none'
        }
    }

# Add template to config manager
ConfigManager._my_custom_template = _my_custom_template

# Create configuration file: config/mode/my_custom.yaml
config_yaml = '''
# My Custom Algorithm Configuration
algorithm: my_custom
learning_rate: 0.001
hidden_dim: 128
batch_size: 32
rollout_length: 100
update_frequency: 10
gamma: 0.99

# Network configuration
network:
  hidden_dims: [128, 128]
  activation: 'relu'
  parameter_sharing: 'none'

# Training configuration
training:
  max_episodes: 2000
  log_frequency: 100
  save_frequency: 500
'''

with open('config/mode/my_custom.yaml', 'w') as f:
    f.write(config_yaml)`,
            explanation: 'Register your algorithm with the factory and config system to integrate seamlessly with all EasyMARL features.'
          },
          {
            title: 'Use Your Custom Algorithm',
            code: `# Use your custom algorithm just like built-in ones
from easymarl.controllers import UnifiedMultiAgentController
import easymarl

# Method 1: Using UnifiedMultiAgentController
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Empty-8x8-v0',
    algorithm='my_custom',  # Your custom algorithm
    config={
        'learning_rate': 1e-3,
        'hidden_dim': 256,
        'batch_size': 64
    },
    n_envs=8
)

results = controller.train(total_episodes=2000)

# Method 2: Using QuickStart API
trainer = easymarl.QuickStart(
    algorithm='my_custom',
    environment='MultiGrid-Empty-8x8-v0',
    episodes=1000
)

results = trainer.train()

# Method 3: Using the GUI
# Your algorithm will automatically appear in the dropdown menu!
# Just select 'my_custom' from the algorithm list in the web interface

# Evaluate your custom algorithm
evaluation_results = controller.evaluate(
    episodes=100,
    render=True,
    save_video=True
)

print(f"Custom algorithm performance: {evaluation_results['mean_reward']}")`,
            explanation: 'Your custom algorithm becomes a first-class citizen in EasyMARL, usable through all interfaces: controller, API, and GUI.'
          },
          {
            title: 'Add GUI Support',
            code: `# Add your algorithm to the GUI interface
# File: gui/gradio_interface.py

# Add to available algorithms list
AVAILABLE_ALGORITHMS = [
    "ippo",
    "maddpg", 
    "qmix",
    # ... existing algorithms
    "my_custom",  # Add your algorithm here
]

# Add algorithm description for GUI
ALGORITHM_DESCRIPTIONS = {
    # ... existing descriptions
    "my_custom": {
        "name": "My Custom Algorithm",
        "type": "Policy-Based Independent Learning",
        "description": "Custom actor-critic algorithm with independent learning and custom neural network architecture.",
        "features": ["Actor-Critic", "Independent Learning", "Custom Architecture", "Experience Replay"],
        "best_for": "Custom scenarios requiring specialized learning dynamics",
        "pros": ["Flexible architecture", "Customizable learning", "Research-oriented"],
        "cons": ["Requires tuning", "Less tested than built-ins"]
    }
}

# Your algorithm will now appear in the GUI with full description!`,
            explanation: 'Adding GUI support makes your custom algorithm accessible to all users through the web interface with descriptions and usage guidance.'
          }
        ]
      }
    },
    {
      id: 'research-interface',
      title: 'Advanced Research Interface',
      level: 'Expert',
      duration: '75 minutes',
      icon: <Brain className="w-6 h-6" />,
      description: 'Master the research interface for algorithm discovery, experimentation, and optimization.',
      objectives: [
        'Use algorithm discovery and search tools',
        'Set up structured research experiments',
        'Perform algorithm comparisons and analysis',
        'Conduct hyperparameter optimization studies'
      ],
      content: {
        steps: [
          {
            title: 'Algorithm Discovery and Search',
            code: `# Advanced algorithm discovery for research
from easymarl.core.research_interface import AlgorithmDiscovery, ResearchInterface

# Initialize discovery system
discovery = AlgorithmDiscovery()

# Search algorithms by research criteria
print("🔍 Discovering algorithms for your research...")

# Find cooperative algorithms for your research
cooperative_algos = discovery.search_algorithms(
    category='Value-Based',
    type='Cooperative',
    action_space='Discrete'
)
print(f"Cooperative value-based algorithms: {cooperative_algos}")

# Find algorithms by complexity level
beginner_algos = discovery.search_algorithms(complexity='Beginner')
intermediate_algos = discovery.search_algorithms(complexity='Intermediate') 
advanced_algos = discovery.search_algorithms(complexity='Advanced')

print(f"\\nAlgorithms by complexity:")
print(f"  Beginner: {beginner_algos}")
print(f"  Intermediate: {intermediate_algos}")
print(f"  Advanced: {advanced_algos}")

# Get research recommendations based on focus area
research_recs = discovery.get_research_recommendations(
    research_focus='cooperative_ai',
    experience_level='intermediate'
)

print(f"\\n🎯 Research Recommendations:")
print(f"  Recommended: {research_recs['recommended']}")
print(f"  Alternatives: {research_recs['alternatives']}")

# Get detailed algorithm information
for algo in research_recs['recommended']:
    info = discovery.get_algorithm_info(algo)
    print(f"\\n📋 {algo.upper()} Details:")
    print(f"  Paper: {info.get('paper', 'N/A')}")
    print(f"  Key Concepts: {info.get('key_concepts', [])}")
    print(f"  Strengths: {info.get('strengths', [])}")
    print(f"  Research Applications: {info.get('research_applications', [])}")`,
            explanation: 'The algorithm discovery system helps researchers find the most suitable algorithms for their specific research objectives and experience level.'
          },
          {
            title: 'Structured Research Experiments',
            code: `# Create comprehensive research experiments
research = ResearchInterface()

# Create a detailed research experiment
experiment = research.create_experiment(
    name="cooperative_coordination_study",
    description="Investigating value decomposition methods for multi-agent coordination in complex environments",
    algorithm="qmix",
    environment="MultiGrid-Cooperative-12x12"
)

# Configure detailed experiment parameters
experiment.hyperparameters = {
    'learning_rate': 5e-4,
    'batch_size': 64,
    'mixer_hidden_dim': 256,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.05,
    'target_update_frequency': 200,
    'gamma': 0.99
}

# Advanced training configuration
experiment.training_config = {
    'max_episodes': 5000,
    'evaluation_frequency': 500,
    'save_frequency': 1000,
    'early_stopping': True,
    'patience': 10,
    'min_improvement': 0.01
}

# Research-specific tracking options
experiment.research_options = {
    'track_agent_states': True,
    'record_action_distributions': True,
    'analyze_coordination_patterns': True,
    'compare_with_baselines': ['vdn', 'ippo'],
    'ablation_studies': ['no_mixer', 'different_network_sizes'],
    'statistical_significance': True,
    'confidence_intervals': 0.95
}

# Validation configuration
validation_errors = research.validate_configuration(experiment)
if validation_errors:
    print("⚠️ Configuration warnings:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("✅ Experiment configuration validated successfully")

# Save experiment configuration for reproducibility
research.save_experiment_config(experiment, "cooperative_study_config.json")
print("💾 Experiment configuration saved for reproducibility")`,
            explanation: 'Structured experiments ensure reproducible research with comprehensive tracking of parameters, metrics, and experimental conditions.'
          },
          {
            title: 'Algorithm Comparison and Analysis',
            code: `# Comprehensive algorithm comparison for research
comparison_algorithms = ['qmix', 'vdn', 'mappo', 'coma']

print("🔬 Starting comprehensive algorithm comparison...")

# Get detailed comparison matrix
comparison = discovery.compare_algorithms(comparison_algorithms)

# Display comparison table
print("\\n📊 Algorithm Comparison Matrix:")
print("=" * 100)
print(f"{'Algorithm':<10} {'Category':<25} {'Complexity':<12} {'Sample Eff.':<12} {'Scalability':<12}")
print("=" * 100)

for algo, details in comparison.items():
    print(f"{algo.upper():<10} "
          f"{details['category']:<25} "
          f"{details['complexity']:<12} "
          f"{details['sample_efficiency']:<12} "
          f"{details['scalability']:<12}")

# Detailed analysis
print("\\n🔍 Detailed Algorithm Analysis:")
for algo, details in comparison.items():
    print(f"\\n{algo.upper()} Analysis:")
    print(f"  ✅ Strengths: {', '.join(details['strengths'])}")
    print(f"  ⚠️  Limitations: {', '.join(details['limitations'])}")
    print(f"  🎛️  Key Hyperparameters: {', '.join(details['key_hyperparameters'])}")

# Research recommendations based on criteria
print("\\n🎯 Algorithm Selection Guidance:")

criteria_analysis = {
    'sample_efficiency': {},
    'implementation_complexity': {},
    'scalability': {}
}

for algo, details in comparison.items():
    for criterion in criteria_analysis:
        if criterion == 'sample_efficiency':
            score = details['sample_efficiency']
        elif criterion == 'implementation_complexity':
            complexity_scores = {'Beginner': 4, 'Intermediate': 3, 'Advanced': 2, 'Expert': 1}
            score = complexity_scores.get(details['complexity'], 2)
        elif criterion == 'scalability':
            scalability_scores = {'Excellent': 5, 'Good': 4, 'Medium': 3, 'Poor': 2}
            score = scalability_scores.get(details['scalability'], 3)
        
        criteria_analysis[criterion][algo] = score

# Find best algorithms for each criterion
for criterion, scores in criteria_analysis.items():
    best_algo = max(scores.keys(), key=lambda x: scores[x] if isinstance(scores[x], (int, float)) else 0)
    print(f"  🏆 Best for {criterion}: {best_algo.upper()}")

# Generate research comparison report
comparison_report = {
    'algorithms_tested': comparison_algorithms,
    'comparison_matrix': comparison,
    'recommendations': criteria_analysis,
    'research_notes': "Comprehensive comparison of cooperative MARL algorithms",
    'timestamp': datetime.now().isoformat()
}

# Save comparison report
import json
with open('algorithm_comparison_report.json', 'w') as f:
    json.dump(comparison_report, f, indent=2, default=str)

print("\\n💾 Detailed comparison report saved to 'algorithm_comparison_report.json'")`,
            explanation: 'Algorithm comparison provides systematic analysis to help researchers choose the most appropriate algorithms for their specific requirements and constraints.'
          },
          {
            title: 'Hyperparameter Optimization Studies',
            code: `# Advanced hyperparameter optimization for research
from easymarl.core.research_interface import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()

print("🎯 Starting hyperparameter optimization study...")

# Define comprehensive parameter search space
parameter_ranges = {
    'learning_rate': {
        'min': 1e-5, 
        'max': 1e-2, 
        'type': 'log_uniform',
        'description': 'Learning rate for neural networks'
    },
    'batch_size': {
        'values': [16, 32, 64, 128, 256], 
        'type': 'categorical',
        'description': 'Batch size for training'
    },
    'mixer_hidden_dim': {
        'min': 64, 
        'max': 512, 
        'step': 64, 
        'type': 'int',
        'description': 'Hidden dimension of mixing network'
    },
    'epsilon_decay': {
        'min': 0.99, 
        'max': 0.9999, 
        'type': 'uniform',
        'description': 'Epsilon decay rate for exploration'
    },
    'gamma': {
        'min': 0.95, 
        'max': 0.99, 
        'type': 'uniform',
        'description': 'Discount factor'
    },
    'target_update_frequency': {
        'min': 50, 
        'max': 500, 
        'step': 50, 
        'type': 'int',
        'description': 'Target network update frequency'
    }
}

print(f"📋 Optimizing {len(parameter_ranges)} hyperparameters:")
for param, config in parameter_ranges.items():
    print(f"  - {param}: {config['description']}")

# Run Bayesian optimization
optimization_results = optimizer.optimize(
    algorithm='qmix',
    environment='MultiGrid-Cooperative-8x8',
    parameter_ranges=parameter_ranges,
    optimization_method='bayesian',
    n_trials=50,
    evaluation_episodes=100,
    n_seeds=3,  # Multiple seeds for statistical significance
    timeout_hours=6
)

print("\\n🏆 Optimization Results:")
print(f"  Best Score: {optimization_results['best_score']:.4f}")
print(f"  Optimization Time: {optimization_results['optimization_time']:.2f} hours")
print(f"  Total Trials: {optimization_results['n_trials']}")

print("\\n🎛️ Best Hyperparameters:")
for param, value in optimization_results['best_params'].items():
    print(f"  {param}: {value}")

# Statistical analysis of results
print("\\n📊 Statistical Analysis:")
print(f"  Mean Score: {optimization_results['score_statistics']['mean']:.4f}")
print(f"  Std Dev: {optimization_results['score_statistics']['std']:.4f}")
print(f"  95% Confidence Interval: [{optimization_results['score_statistics']['ci_lower']:.4f}, {optimization_results['score_statistics']['ci_upper']:.4f}]")

# Parameter importance analysis
print("\\n🔍 Parameter Importance Ranking:")
for param, importance in optimization_results['parameter_importance'].items():
    print(f"  {param}: {importance:.3f}")

# Visualization and analysis
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Optimization progress
axes[0, 0].plot(optimization_results['scores_history'])
axes[0, 0].set_title('Optimization Progress')
axes[0, 0].set_xlabel('Trial')
axes[0, 0].set_ylabel('Performance Score')

# Best score over time
axes[0, 1].plot(optimization_results['best_scores_history'])
axes[0, 1].set_title('Best Score Over Time')
axes[0, 1].set_xlabel('Trial')
axes[0, 1].set_ylabel('Best Score So Far')

# Parameter importance
params = list(optimization_results['parameter_importance'].keys())
importance = list(optimization_results['parameter_importance'].values())
axes[1, 0].barh(params, importance)
axes[1, 0].set_title('Parameter Importance')
axes[1, 0].set_xlabel('Importance Score')

# Score distribution
axes[1, 1].hist(optimization_results['scores_history'], bins=20, alpha=0.7)
axes[1, 1].set_title('Score Distribution')
axes[1, 1].set_xlabel('Performance Score')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('hyperparameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n📈 Optimization analysis plots saved to 'hyperparameter_optimization_analysis.png'")

# Save complete optimization study
optimization_study = {
    'experiment_name': 'qmix_hyperparameter_optimization',
    'algorithm': 'qmix',
    'environment': 'MultiGrid-Cooperative-8x8',
    'parameter_ranges': parameter_ranges,
    'results': optimization_results,
    'methodology': 'Bayesian optimization with statistical significance testing',
    'timestamp': datetime.now().isoformat()
}

with open('hyperparameter_optimization_study.json', 'w') as f:
    json.dump(optimization_study, f, indent=2, default=str)

print("💾 Complete optimization study saved to 'hyperparameter_optimization_study.json'")`,
            explanation: 'Hyperparameter optimization provides systematic tuning with statistical analysis to find optimal algorithm configurations for research applications.'
          },
          {
            title: 'Research Workflow Integration',
            code: `# Complete research workflow using optimized parameters
from easymarl.controllers import UnifiedMultiAgentController
from datetime import datetime
import wandb

print("🚀 Starting complete research workflow with optimized parameters...")

# Initialize experiment tracking
wandb.init(
    project="marl_research_study",
    name=f"qmix_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=optimization_results['best_params'],
    tags=['optimized', 'qmix', 'cooperative', 'research']
)

# Create controller with optimized parameters
optimized_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',
    config=optimization_results['best_params'],
    n_envs=16,  # Increased parallelization for faster training
    educational_mode=False,  # Production mode for research
    wandb_logging=True
)

# Extended training with comprehensive evaluation
print("📈 Training with optimized hyperparameters...")
training_results = optimized_controller.train(
    total_episodes=5000,
    save_frequency=1000,
    eval_frequency=500,
    eval_episodes=50
)

# Comprehensive evaluation
print("🧪 Conducting comprehensive evaluation...")
evaluation_results = optimized_controller.evaluate(
    num_episodes=200,
    render=False,
    detailed_analysis=True,
    save_trajectories=True
)

# Research analysis
research_metrics = {
    'final_performance': evaluation_results['average_reward'],
    'training_stability': np.std(training_results['episode_rewards'][-1000:]),
    'sample_efficiency': training_results.get('episodes_to_convergence', 5000),
    'computational_efficiency': training_results['training_time'] / training_results['total_episodes'],
    'success_rate': evaluation_results.get('success_rate', 0),
    'coordination_score': evaluation_results.get('coordination_metrics', {}).get('average_coordination', 0)
}

print("\\n📊 Final Research Metrics:")
for metric, value in research_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Compare with baseline (non-optimized)
print("\\n🔄 Comparing with baseline performance...")
baseline_controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Cooperative-8x8',
    algorithm='qmix',
    config={},  # Default parameters
    n_envs=16
)

baseline_results = baseline_controller.train(total_episodes=5000)
baseline_eval = baseline_controller.evaluate(num_episodes=200)

improvement_metrics = {
    'reward_improvement': (evaluation_results['average_reward'] - baseline_eval['average_reward']) / baseline_eval['average_reward'] * 100,
    'training_time_reduction': (baseline_results['training_time'] - training_results['training_time']) / baseline_results['training_time'] * 100,
    'sample_efficiency_gain': (baseline_results.get('episodes_to_convergence', 5000) - training_results.get('episodes_to_convergence', 5000)) / baseline_results.get('episodes_to_convergence', 5000) * 100
}

print("\\n🏆 Optimization Impact:")
for metric, improvement in improvement_metrics.items():
    print(f"  {metric}: {improvement:+.2f}%")

# Generate research report
research_report = {
    'study_title': 'QMIX Hyperparameter Optimization Study',
    'algorithm': 'qmix',
    'environment': 'MultiGrid-Cooperative-8x8',
    'optimization_method': 'Bayesian Optimization',
    'best_parameters': optimization_results['best_params'],
    'final_metrics': research_metrics,
    'improvement_over_baseline': improvement_metrics,
    'statistical_significance': optimization_results['score_statistics'],
    'methodology': 'Systematic hyperparameter optimization followed by comprehensive evaluation',
    'conclusions': [
        'Optimized parameters significantly improve performance',
        'Bayesian optimization effectively explores parameter space',
        'Statistical significance achieved with multiple seeds'
    ],
    'future_work': [
        'Test on larger environments',
        'Compare with other optimization methods',
        'Investigate parameter transfer across environments'
    ],
    'timestamp': datetime.now().isoformat()
}

# Save complete research study
with open('complete_research_study.json', 'w') as f:
    json.dump(research_report, f, indent=2, default=str)

# Log to wandb
wandb.log(research_metrics)
wandb.log(improvement_metrics)
wandb.finish()

print("\\n✅ Research study completed successfully!")
print("📄 Complete research report saved to 'complete_research_study.json'")
print("📊 Experiment data logged to Weights & Biases")
print("🎯 Ready for publication or further research!")`,
            explanation: 'The complete research workflow integrates optimization, evaluation, and analysis to produce publication-ready results with statistical significance and comprehensive documentation.'
          }
        ]
      }
    },
    {
      id: 'production-deployment',
      title: 'Production Deployment',
      level: 'Advanced',
      duration: '40 minutes',
      icon: <Monitor className="w-6 h-6" />,
      description: 'Deploy trained models to production with monitoring and scaling.',
      objectives: [
        'Export trained models',
        'Set up inference pipeline',
        'Monitor model performance',
        'Handle model updates'
      ],
      content: {
        steps: [
          {
            title: 'Model Export and Saving',
            code: `# Train and save model
controller = UnifiedMultiAgentController(
    env_name='MultiGrid-Production-10x10',
    algorithm='qmix',
    n_envs=16,
    production_mode=True  # Optimized for production
)

results = controller.train(total_episodes=5000)

# Save production model
model_path = controller.save_models("production_model_v1")
print(f"Model saved to: {model_path}")

# Save deployment config
import json
deployment_config = {
    'algorithm': 'qmix',
    'env_name': 'MultiGrid-Production-10x10',
    'model_path': model_path,
    'n_agents': controller.n_agents,
    'observation_space': str(controller.observation_space),
    'action_space': str(controller.action_space)
}

with open('deployment_config.json', 'w') as f:
    json.dump(deployment_config, f, indent=2)`,
            explanation: 'Save trained models with complete configuration for reproducible deployment.'
          },
          {
            title: 'Inference Pipeline',
            code: `class ProductionInference:
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load trained model
        self.controller = UnifiedMultiAgentController(
            env_name=self.config['env_name'],
            algorithm=self.config['algorithm'],
            n_envs=1,  # Single environment for inference
            training=False  # Inference mode
        )
        self.controller.load_models(model_path)
        
    def predict(self, observations):
        """Get actions for given observations"""
        with torch.no_grad():
            actions = self.controller.algorithm.select_actions(
                observations, 
                deterministic=True  # Deterministic for production
            )
        return actions
    
    def predict_batch(self, observation_batch):
        """Handle batch predictions efficiently"""
        batch_actions = []
        for obs in observation_batch:
            actions = self.predict(obs)
            batch_actions.append(actions)
        return batch_actions

# Usage
inference = ProductionInference('deployment_config.json', model_path)
actions = inference.predict(current_observations)`,
            explanation: 'Create a production inference pipeline that loads models and handles predictions efficiently.'
          },
          {
            title: 'Monitoring and Performance',
            code: `import time
import logging
from collections import deque

class ProductionMonitor:
    def __init__(self, window_size=1000):
        self.metrics = {
            'prediction_times': deque(maxlen=window_size),
            'prediction_count': 0,
            'error_count': 0
        }
        
    def log_prediction(self, prediction_time, success=True):
        self.metrics['prediction_times'].append(prediction_time)
        self.metrics['prediction_count'] += 1
        
        if not success:
            self.metrics['error_count'] += 1
            
        # Log performance every 100 predictions
        if self.metrics['prediction_count'] % 100 == 0:
            self.report_performance()
    
    def report_performance(self):
        avg_time = np.mean(self.metrics['prediction_times'])
        error_rate = self.metrics['error_count'] / self.metrics['prediction_count']
        
        logging.info(f"Performance Report:")
        logging.info(f"  Average prediction time: {avg_time:.3f}s")
        logging.info(f"  Error rate: {error_rate:.3%}")
        logging.info(f"  Predictions per second: {1/avg_time:.1f}")

# Integration with inference
monitor = ProductionMonitor()

def monitored_predict(observations):
    start_time = time.time()
    try:
        actions = inference.predict(observations)
        prediction_time = time.time() - start_time
        monitor.log_prediction(prediction_time, success=True)
        return actions
    except Exception as e:
        prediction_time = time.time() - start_time
        monitor.log_prediction(prediction_time, success=False)
        logging.error(f"Prediction error: {e}")
        raise`,
            explanation: 'Monitor production performance including prediction times, error rates, and throughput metrics.'
          }
        ]
      }
    },

    // Algorithm Documentation and Code Understanding
    {
      id: 'algorithm-documentation',
      title: 'Understanding EasyMARL Algorithm Documentation',
      description: 'Deep dive into our comprehensive algorithm documentation and commenting system',
      level: 'Beginner',
      duration: '15 minutes',
      category: 'Documentation',
      content: {
        overview: 'EasyMARL features the most comprehensive algorithm documentation in the MARL ecosystem. Every single line of algorithm code is commented and explained, making it perfect for learning and research.',
        steps: [
          {
            title: 'Unified Algorithm Library Structure',
            code: `# All 21+ algorithms are organized in a single file:
# algorithms/__init__.py - The central algorithm registry

from easymarl.algorithms import AlgorithmFactory

# View all available algorithms organized by taxonomy
print("Value-Based Algorithms:")
value_based = AlgorithmFactory.get_algorithms_by_category('value_based')
for name, algorithm_class in value_based.items():
    print(f"  {name}: {algorithm_class.__doc__.split('.')[0]}")

print("\\nPolicy-Based Algorithms:")
policy_based = AlgorithmFactory.get_algorithms_by_category('policy_based')  
for name, algorithm_class in policy_based.items():
    print(f"  {name}: {algorithm_class.__doc__.split('.')[0]}")

# Example output:
# Value-Based Algorithms:
#   qmix: Monotonic value function factorization for cooperative MARL
#   vdn: Simple additive value decomposition for team coordination
#   qtran: General value factorization without monotonicity constraints
#   iql: Independent Q-learning baseline
#   mfq: Mean field Q-learning for large-scale systems`,
            explanation: 'All algorithms are centrally organized with clear taxonomical structure and comprehensive documentation.'
          },
          {
            title: 'Reading Algorithm Source Code',
            code: `# Every algorithm implementation has extensive documentation
# Example: Reading QMIX algorithm source code

# 1. Navigate to algorithm implementation
import inspect
from easymarl.algorithms.model_free.value_based.approximation.qmix import QMIX

# 2. View the comprehensive class documentation
print("QMIX Algorithm Documentation:")
print(QMIX.__doc__)

# 3. Each method is thoroughly documented
print("\\nQMIX Training Method Documentation:")  
print(QMIX.train_step.__doc__)

# 4. View source code with comments
source_code = inspect.getsource(QMIX.train_step)
print("\\nSource Code with Line-by-Line Comments:")
print(source_code[:500] + "...")  # First 500 characters

# The code includes:
# - Theoretical background and paper references
# - When to use each algorithm
# - Step-by-step explanations of complex operations
# - Beginner-friendly analogies and examples
# - Implementation details and optimization notes`,
            explanation: 'Every algorithm includes comprehensive documentation with theoretical background, implementation details, and usage guidance.'
          },
          {
            title: 'Algorithm Documentation Structure',
            code: `# Each algorithm file follows this comprehensive documentation structure:

"""
Algorithm Name: Clear, descriptive title
Paper Reference: Original research paper with link
Theoretical Background: How the algorithm works conceptually
Key Innovation: What makes this algorithm unique
Mathematical Foundation: Core equations and principles
When to Use: Specific scenarios where algorithm excels
Advantages: Strengths compared to other methods
Limitations: Known weaknesses and constraints
For Beginners: Simplified explanations and analogies
Use Cases: Real-world applications and examples
"""

class AlgorithmImplementation:
    """
    High-level class documentation explaining:
    - Algorithm purpose and goals
    - Key components and architecture
    - Relationships between components
    - Learning paradigm (centralized/decentralized)
    """
    
    def __init__(self, config):
        """Detailed parameter documentation"""
        # Every line explains why it exists
        self.parameter = config.get('param', default_value)  # Purpose of this parameter
        
    def core_method(self, inputs):
        """
        Method-level documentation explains:
        - What this method accomplishes
        - Input/output specifications
        - Key algorithmic steps
        - Mathematical operations
        """
        # Step 1: Clear explanation of this operation
        intermediate_result = self.process_inputs(inputs)
        
        # Step 2: Why this computation is necessary
        final_result = self.apply_algorithm_logic(intermediate_result)
        
        return final_result  # What this return value represents`,
            explanation: 'Every algorithm follows a consistent documentation structure making the codebase accessible to learners and researchers.'
          },
          {
            title: 'Comparing Algorithm Documentation',
            code: `# Compare EasyMARL documentation with other libraries:

# 1. EasyMARL: Comprehensive educational documentation
print("EasyMARL QMIX Documentation Preview:")
print("""
QMIX Algorithm for Multi-Agent Reinforcement Learning

Key Innovation - Individual-Global-Max (IGM) Principle:
QMIX ensures that the optimal joint action corresponds
to each agent taking their individually optimal action.

How QMIX Works:
1. Each agent has its own Q-network 
2. A mixing network combines individual Q-values
3. The mixing network has only positive weights
4. Training is centralized but execution is decentralized

For MARL Beginners:
QMIX is an excellent starting point for learning MARL...
""")

# 2. Other libraries: Minimal documentation
print("\\nTypical Other Library Documentation:")
print("""
class QMIX:
    def __init__(self, config):
        self.mixer = MixingNetwork()
    def forward(self, q_values):
        return self.mixer(q_values)
""")

# 3. EasyMARL advantage: Line-by-line explanations
print("\\nEasyMARL Line-by-Line Documentation:")
print("""
# Exploration Parameters for Epsilon-Greedy Action Selection
# Start with high exploration and gradually reduce it
self.epsilon = config.get('epsilon_start', 1.0)        # Current exploration rate (100% initially)
self.epsilon_end = config.get('epsilon_end', 0.05)     # Minimum exploration rate (5% final)  
self.epsilon_decay = config.get('epsilon_decay', 0.995) # How fast to reduce exploration
""")`,
            explanation: 'EasyMARL provides unmatched documentation quality compared to other MARL libraries, making complex algorithms accessible to everyone.'
          },
          {
            title: 'Using Documentation for Learning',
            code: `# How to use EasyMARL documentation for learning MARL:

# 1. Start with algorithm overview
from easymarl import AlgorithmGuide

guide = AlgorithmGuide()

# Get learning progression recommendation
learning_path = guide.get_learning_progression()
print("Recommended Learning Path:")
for level, algorithms in learning_path.items():
    print(f"{level}: {', '.join(algorithms)}")

# Output:
# Beginner: VDN, IPPO, IQL
# Intermediate: QMIX, MAPPO, MADDPG
# Advanced: QTRAN, COMA, MAVEN

# 2. Read algorithm documentation in order
for algorithm in learning_path['Beginner']:
    print(f"\\nStudying {algorithm}:")
    algorithm_class = AlgorithmFactory.get_algorithm(algorithm)
    
    # Read comprehensive documentation
    print("Theory:", algorithm_class.get_theory_summary())
    print("Implementation:", algorithm_class.get_implementation_guide()) 
    print("Examples:", algorithm_class.get_usage_examples())

# 3. Understand code with inline comments
print("\\nExample: Understanding VDN step by step")
print("Each line explains the mathematical operation:")
print("# Q_total = Q_1 + Q_2 + ... + Q_n (additive decomposition)")
print("total_q_value = torch.sum(individual_q_values, dim=1)")`,
            explanation: 'Use the comprehensive documentation to learn MARL concepts systematically, from basic algorithms to advanced research methods.'
          },
          {
            title: 'Algorithm Implementation Best Practices',
            code: `# EasyMARL demonstrates best practices in algorithm implementation:

# 1. Clear variable naming with documentation
class QMIXAgent:
    def __init__(self, agent_id, obs_space, action_space, config):
        # Exploration Parameters (clearly documented purpose)
        self.epsilon_start = config.get('epsilon_start', 1.0)    # Initial exploration rate
        self.epsilon_end = config.get('epsilon_end', 0.05)       # Final exploration rate
        self.epsilon_decay = config.get('epsilon_decay', 0.995)  # Decay rate per episode
        
        # Network Architecture (documented components)
        self.q_network = QNetwork(obs_space, action_space)      # Main Q-function approximator
        self.target_network = copy.deepcopy(self.q_network)     # Stable target for TD learning
        
# 2. Mathematical operations with explanations
def compute_td_target(self, rewards, next_q_values, dones, gamma=0.99):
    """
    Compute Temporal Difference target for Q-learning.
    
    TD Target = reward + gamma * max(next_Q_values) * (1 - done)
    This represents the "true" Q-value we want our network to predict.
    """
    # Handle episode termination (done=True means no future rewards)
    next_q_max = torch.max(next_q_values, dim=-1)[0]  # Best action in next state
    td_targets = rewards + gamma * next_q_max * (1 - dones.float())
    return td_targets

# 3. Educational comments for complex operations
def mixing_network_forward(self, agent_q_values, state):
    """
    QMIX Mixing Network: Combines individual Q-values into team Q-value.
    
    Key Insight: Uses positive weights to ensure Individual-Global-Max property.
    This means: argmax(team_Q) = [argmax(Q1), argmax(Q2), ..., argmax(Qn)]
    """
    # Generate mixing weights from global state (positive weights ensure IGM)
    mixing_weights = F.softplus(self.hyper_w1(state))  # Softplus ensures positivity
    
    # Weighted combination of individual Q-values
    team_q_value = torch.sum(mixing_weights * agent_q_values, dim=-1)
    return team_q_value`,
            explanation: 'Learn from production-quality code that demonstrates best practices in MARL implementation, documentation, and mathematical clarity.'
          }
        ]
      }
    }
  ];

  const getDifficultyColor = (level) => {
    switch(level) {
      case 'Beginner': return 'bg-green-100 text-green-700 border-green-200';
      case 'Intermediate': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'Advanced': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">Tutorials</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Step-by-step tutorials to master Multi-Agent Reinforcement Learning with EasyMARL.
            From basic concepts to advanced deployment strategies.
          </p>
        </div>

        {/* Tutorial Selection */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {tutorials.map((tutorial) => (
            <div
              key={tutorial.id}
              className={`bg-white rounded-lg shadow-lg p-6 cursor-pointer transition-all duration-300 hover:shadow-xl border-2 ${
                selectedTutorial?.id === tutorial.id ? 'border-blue-500' : 'border-transparent'
              }`}
              onClick={() => setSelectedTutorial(tutorial)}
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                  {tutorial.icon}
                </div>
                <div>
                  <h3 className="text-lg font-bold">{tutorial.title}</h3>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Clock className="w-4 h-4" />
                    <span>{tutorial.duration}</span>
                  </div>
                </div>
              </div>
              
              <p className="text-gray-700 mb-4">{tutorial.description}</p>
              
              <div className="flex items-center justify-between">
                <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(tutorial.level)}`}>
                  {tutorial.level}
                </span>
                <button className="text-blue-600 hover:text-blue-800 font-medium text-sm flex items-center space-x-1">
                  <span>Start Tutorial</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Tutorial Content */}
        {selectedTutorial && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center space-x-4 mb-6">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg text-white">
                {selectedTutorial.icon}
              </div>
              <div>
                <h2 className="text-3xl font-bold">{selectedTutorial.title}</h2>
                <div className="flex items-center space-x-4 text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{selectedTutorial.duration}</span>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(selectedTutorial.level)}`}>
                    {selectedTutorial.level}
                  </span>
                </div>
              </div>
            </div>

            {/* Learning Objectives */}
            <div className="mb-8 p-6 bg-blue-50 rounded-lg">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2 text-blue-600" />
                Learning Objectives
              </h3>
              <ul className="space-y-2">
                {selectedTutorial.objectives.map((objective, idx) => (
                  <li key={idx} className="flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span>{objective}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Tutorial Steps */}
            <div className="space-y-8">
              {selectedTutorial.content.steps.map((step, idx) => (
                <div key={idx} className="border-l-4 border-blue-500 pl-6">
                  <h3 className="text-xl font-semibold mb-3 flex items-center">
                    <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm mr-3">
                      {idx + 1}
                    </div>
                    {step.title}
                  </h3>
                  <p className="text-gray-700 mb-4">{step.explanation}</p>
                  <CodeBlock code={step.code} language="python" />
                </div>
              ))}
            </div>

            {/* Next Steps */}
            <div className="mt-12 p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
              <h3 className="text-xl font-semibold mb-3 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
                Next Steps
              </h3>
              <p className="text-gray-700 mb-4">
                Congratulations on completing this tutorial! Here are some recommended next steps:
              </p>
              <ul className="space-y-2">
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Try the code examples in your own environment</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Explore the API reference for advanced features</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Check out more advanced tutorials</span>
                </li>
                <li className="flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4 text-green-600" />
                  <span>Join the community for questions and discussions</span>
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TutorialsPage;
