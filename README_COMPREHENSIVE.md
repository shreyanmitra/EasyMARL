# EasyMARL - Multi-Agent Reinforcement Learning Made Simple 🚀

EasyMARL is a comprehensive framework for Multi-Agent Reinforcement Learning (MARL) that provides both beginner-friendly and advanced interfaces for training and evaluating MARL algorithms.

## 🌟 Key Features

### 🎯 Dual Controller Architecture
- **Simple Controller**: Beginner-friendly structure similar to original metacontroller, perfect for learning MARL concepts
- **Modern Controller**: Advanced features with comprehensive metrics and professional experiment tracking

### 🧠 Comprehensive Algorithm Support
21+ state-of-the-art MARL algorithms across multiple categories:

| Category | Algorithms | Best For |
|----------|------------|----------|
| **Value Decomposition** | QMIX, VDN, QTRAN | Cooperative tasks, team coordination |
| **Actor-Critic** | MAPPO, MADDPG, COMA | Continuous control, policy learning |
| **Game-Theoretic** | Nash-Q, WoLF-PHC | Competitive scenarios, equilibrium |
| **Large-Scale** | MFAC, MFQ | Many agents, scalability |
| **Communication** | MADDPG-Comm, COMA-Comm | Information sharing, coordination |

### 🎨 Professional GUI Interface
- Real-time training visualization with matplotlib graphs
- Comprehensive algorithm descriptions and documentation
- WandB integration for experiment tracking
- Downloadable training data in JSON format
- Environment visualization and monitoring

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YourUsername/EasyMARL.git
cd EasyMARL
pip install -r requirements.txt
```

### Launch GUI
```bash
python gui.py
```
Opens at `http://localhost:7860` with full interface.

### Quick Training Example

#### For Beginners (Simple Controller)
```python
from simple_multiagent_controller import SimpleMultiAgentController
import utils
import torch

# Create environment
env = utils.make_env('MultiGrid-Cluttered-Fixed-15x15')

# Basic configuration
config = {
    'max_episodes': 1000,
    'learning_rate': 0.001,
    'max_steps': 100
}

# Create simple controller
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controller = SimpleMultiAgentController(
    env=env,
    config=config,
    device=device,
    algorithm='qmix',
    training=True
)

# Train agents (similar to original metacontroller)
controller.train()

# Evaluate performance
results = controller.evaluate(num_episodes=10)
print(f"Average reward: {results['mean_reward']}")
```

#### For Advanced Users (Modern Controller)
```python
from modern_multiagent_controller import ModernMultiAgentController
import utils
import torch

# Create environment
env = utils.make_env('MultiGrid-Cluttered-Fixed-15x15')

# Advanced configuration with WandB
config = {
    'max_episodes': 1000,
    'learning_rate': 0.001,
    'use_wandb': True,
    'wandb_project': 'my-marl-project',
    'log_interval': 10,
    'save_interval': 500
}

# Create modern controller
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controller = ModernMultiAgentController(
    env=env,
    config=config,
    device=device,
    algorithm='mappo',
    training=True
)

# Train with comprehensive logging
controller.train()
```

## 🎮 Controller Comparison

| Feature | Simple Controller | Modern Controller |
|---------|------------------|-------------------|
| **Target Audience** | MARL beginners, students | Advanced researchers, professionals |
| **Code Structure** | Clean, educational, well-commented | Feature-rich, production-ready |
| **Similarity to Original** | Very similar to metacontroller | Enhanced with modern features |
| **Metrics Logging** | Basic episode statistics | Comprehensive metrics with WandB |
| **Documentation** | Extensive line-by-line comments | Professional docstrings |
| **Memory Usage** | Lightweight | Full-featured |
| **Learning Curve** | Gentle, educational | Steep but powerful |
| **Method Names** | Simple, descriptive (run_one_episode) | Advanced, feature-rich |
| **Best For** | Learning MARL concepts | Research and production |

## 📁 Project Structure (Organized for Beginners)

```
EasyMARL/
├── 🎨 Interface Files
│   ├── gui.py                           # Professional web interface
│   └── demo_gui.py                      # Simple demo launcher
├── 🎯 Controllers (Choose Your Level)
│   ├── simple_multiagent_controller.py  # 🟢 Beginner-friendly
│   └── modern_multiagent_controller.py  # 🔵 Advanced features
├── 🧠 Algorithm Implementations
│   ├── algorithms/
│   │   ├── qmix.py                     # QMIX algorithm
│   │   ├── mappo.py                    # Multi-Agent PPO
│   │   └── ...                         # 19+ other algorithms
├── 🌍 Environments
│   └── envs/gym_multigrid/             # MultiGrid environment suite
├── ⚙️ Configuration
│   ├── config/default.yaml            # Default settings
│   └── config/mode/                    # Algorithm-specific configs
├── 🔧 Utilities
│   ├── utils.py                        # Helper functions
│   └── networks/                       # Neural network architectures
└── 📚 Documentation
    ├── README_new.md                   # This file
    ├── GUI_FEATURES.md                 # GUI documentation
    └── IMPLEMENTATION_SUMMARY.md       # Technical details
```

## 🧠 Algorithm Guide for Beginners

### 🟢 Start Here (Beginner-Friendly)
1. **QMIX**: Best for learning MARL basics
   - Easy to understand
   - Good performance on cooperative tasks
   - Clear value decomposition concept

2. **VDN**: Simplest possible MARL
   - Additive value decomposition
   - Great for understanding fundamentals

### 🟡 Intermediate Level
3. **MAPPO**: Industry standard
   - Reliable performance
   - Used in many real applications
   - Good balance of simplicity and power

4. **MADDPG**: Continuous control
   - Learn when you need continuous actions
   - Good for robotics applications

### 🔴 Advanced Algorithms
5. **QTRAN**: Complex value decomposition
6. **MAVEN**: Advanced exploration
7. **Communication variants**: For coordination research

## 🌍 Environments

### MultiGrid Suite
Perfect for learning MARL concepts:

- **MultiGrid-Cluttered**: Navigate through obstacles (Recommended for beginners)
- **MultiGrid-FourRooms**: Room-to-room coordination
- **MultiGrid-Empty**: Open space navigation

Each environment supports:
- 2-4 agents
- Partial observability (7x7 view per agent)
- Discrete action spaces (6 actions)
- Customizable difficulty

## 📊 GUI Features

### 🎯 Training Tab
- Controller type selection (Simple vs Modern)
- Algorithm selection with descriptions
- Environment configuration
- Real-time training graphs
- Training status monitoring

### 🧠 Algorithms Tab
- Detailed algorithm descriptions
- Use cases and recommendations
- Pros and cons for each method
- Implementation details

### 🌍 Environments Tab
- Environment specifications
- Challenge descriptions
- Visual environment layouts

### ❓ Help Tab
- Comprehensive usage guide
- Feature explanations
- Best practices for training

## 🎓 Educational Features

### For MARL Beginners
- **Simple Controller**: Structure similar to original metacontroller
- **Extensive Comments**: Every line explained for learning
- **Clear Method Names**: `run_one_episode()`, `train()`, `evaluate()`
- **Step-by-Step Tutorials**: In GUI and documentation

### For Instructors
- **Gradual Complexity**: Start with Simple, progress to Modern
- **Visual Learning**: Real-time training graphs and environment visualization
- **Hands-On Examples**: Multiple complete code examples
- **Algorithm Comparisons**: Side-by-side feature tables

## 🔧 Configuration Examples

### Beginner Configuration
```yaml
# Use with Simple Controller
algorithm: qmix
max_episodes: 1000
max_steps: 100
learning_rate: 0.001
```

### Advanced Configuration
```yaml
# Use with Modern Controller
algorithm: mappo
max_episodes: 5000
use_wandb: true
wandb_project: "advanced-marl"
log_interval: 10
save_interval: 500
eval_interval: 250
```

## 🎯 Learning Path

### Week 1: Basics
1. Start with GUI → Training Tab → Simple Controller
2. Try QMIX on MultiGrid-Cluttered
3. Watch real-time training graphs
4. Read algorithm descriptions

### Week 2: Code Understanding
1. Open `simple_multiagent_controller.py`
2. Read through `run_one_episode()` method
3. Understand how training loop works
4. Try different algorithms (VDN, MAPPO)

### Week 3: Advanced Features
1. Switch to Modern Controller
2. Enable WandB logging
3. Try communication algorithms
4. Export training data for analysis

### Week 4: Research
1. Implement custom modifications
2. Compare algorithm performance
3. Use downloaded data for research
4. Contribute to the project!

## 🚀 Performance Tips

### For Training
- **Start Small**: 1000 episodes → scale up
- **Use GPU**: Automatic CUDA detection
- **Monitor Memory**: Check system resources
- **Save Checkpoints**: Enable periodic saves

### Algorithm Selection Guide
- **First Time**: QMIX or VDN
- **Continuous Control**: MADDPG or MAPPO
- **Large Teams**: MFAC or MFQ
- **Communication Needed**: MADDPG-Comm
- **Research**: MAVEN or QTRAN

## 🤝 Contributing

We welcome contributions from MARL beginners to experts!

### Easy Contributions
- Fix typos in comments
- Add more algorithm examples
- Improve documentation
- Report bugs

### Advanced Contributions
- New algorithm implementations
- Additional environments
- GUI improvements
- Performance optimizations

### Development Setup
```bash
git clone https://github.com/YourUsername/EasyMARL.git
cd EasyMARL
pip install -r requirements.txt
pip install -e .  # Development install
```

## 📚 Learning Resources

### Recommended Reading
1. **MARL Basics**: Sutton & Barto (2018) - Chapter 13
2. **QMIX Paper**: Rashid et al. (2018)
3. **MAPPO Paper**: Yu et al. (2021)
4. **Survey Paper**: Zhang et al. (2021) - MARL Survey

### Video Tutorials
- GUI walkthrough (coming soon)
- Algorithm explanations (coming soon)
- Code structure overview (coming soon)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Built With Love Using
- **Gradio**: Beautiful web interfaces
- **PyTorch**: Deep learning framework
- **WandB**: Experiment tracking
- **MultiGrid**: Multi-agent environments

### Inspired By
- Original metacontroller design
- Research community feedback
- Educational needs of MARL learners

## 📞 Support & Community

- **🐛 Bug Reports**: GitHub Issues
- **💬 Discussions**: GitHub Discussions
- **📧 Email**: Contact maintainers
- **📖 Documentation**: See GUI Help tab

## 📈 Citation

If you use EasyMARL in your research, please cite:

```bibtex
@software{easymarl2025,
  title={EasyMARL: Multi-Agent Reinforcement Learning Made Simple},
  author={EasyMARL Team},
  year={2025},
  url={https://github.com/YourUsername/EasyMARL},
  note={Framework for accessible MARL research and education}
}
```

---

**🎯 Ready to start your MARL journey? Choose your controller and begin training!** 

- **New to MARL?** → Use Simple Controller with QMIX
- **Experienced?** → Try Modern Controller with MAPPO  
- **Researcher?** → Enable WandB and explore advanced algorithms

**Launch the GUI**: `python gui.py` and start exploring! 🚀
