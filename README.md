# EasyMARL - Multi-Agent Reinforcement Learning Made Simple 🚀

A comprehensive framework for Multi-Agent Reinforcement Learning (MARL) that provides both beginner-friendly and advanced interfaces for training and evaluating MARL algorithms.

## 🌟 Key Features

### 🎯 Dual Controller Architecture
- **Simple Controller**: Beginner-friendly structure similar to original metacontroller, perfect for learning MARL concepts
- **Modern Controller**: Advanced features with comprehensive metrics and professional experiment tracking

### 🧠 Comprehensive Algorithm Support (21+ Algorithms)
Currently supported algorithms across multiple categories:

**Value Decomposition Methods:**
- QMIX, VDN, QTRAN

**Actor-Critic Methods:**
- MAPPO, MADDPG, COMA, IPPO, IQL

**Game-Theoretic Approaches:**
- Nash-Q, WoLF-PHC, Minimax-Q, NFSP

**Large-Scale Methods:**
- DCG, MFQ (Mean Field approaches)

**Communication-Based:**
- MADDPG-Comm, COMA-Comm

**Advanced Methods:**
- MAVEN, HQL, LQL, MAACC

### 🎨 Professional Interface Options
- **Python GUI**: Enhanced Gradio interface with real-time training visualization
- **React Web App**: Modern, mobile-responsive interface deployable to GitHub Pages
- **Command Line**: Traditional CLI interface for scripting and automation

### 🌍 Multi-Agent Environments
- **MultiGrid**: Fully cooperative navigation environments with partial observability
- Support for 2-4 agents with customizable difficulty levels
- Discrete action spaces with challenging coordination tasks

## 🚀 Quick Start

### Option 1: Python GUI (Recommended for Beginners)
```bash
git clone https://github.com/yourusername/EasyMARL.git
cd EasyMARL
pip install -r requirements.txt
python gui.py
```

### Option 2: React Web Interface
```bash
# Deploy to GitHub Pages (see deployment guide below)
cd react-frontend
npm install
npm run deploy
```

### Option 3: Command Line
```bash
python main.py --env_name MultiGrid-Cluttered-Fixed-15x15 --mode qmix --debug
```

## 🎓 Learning Path for MARL Beginners

### Week 1: Start with Simple Controller
1. Launch GUI: `python gui.py`
2. Select "Simple Controller" and QMIX algorithm
3. Train on MultiGrid-Cluttered environment
4. Read code comments in `simple_multiagent_controller.py`

### Week 2: Explore Different Algorithms
1. Try VDN (simpler than QMIX)
2. Experiment with MAPPO (actor-critic method)
3. Compare performance across algorithms
4. Read algorithm descriptions in GUI

### Week 3: Advanced Features
1. Switch to "Modern Controller"
2. Enable WandB logging for experiment tracking
3. Try communication algorithms (MADDPG-Comm)
4. Export training data for analysis

## 🎮 Controller Comparison

| Feature | Simple Controller | Modern Controller |
|---------|------------------|-------------------|
| **Target Users** | MARL beginners, students | Advanced researchers |
| **Code Style** | Extensive comments, educational | Professional, production-ready |
| **Structure** | Similar to original metacontroller | Enhanced with modern features |
| **Logging** | Basic episode statistics | Comprehensive metrics with WandB |
| **Best For** | Learning MARL concepts | Research and development |

## 📱 Interface Options

### Python GUI Features
- Real-time training visualization with matplotlib
- Algorithm catalog with detailed descriptions
- Controller selection (Simple vs Modern)
- WandB integration for experiment tracking
- Downloadable training data

### React Web App Features
- Modern, mobile-responsive design
- GitHub Pages deployment
- Interactive algorithm exploration
- Professional UI/UX
- Cross-platform accessibility

## 🔧 Configuration

### Basic Training (Simple Controller)
```python
from simple_multiagent_controller import SimpleMultiAgentController
import utils

env = utils.make_env('MultiGrid-Cluttered-Fixed-15x15')
config = {'max_episodes': 1000, 'learning_rate': 0.001}

controller = SimpleMultiAgentController(env, config, algorithm='qmix')
controller.train()
```

### Advanced Training (Modern Controller)
```python
from modern_multiagent_controller import ModernMultiAgentController
import utils

env = utils.make_env('MultiGrid-Cluttered-Fixed-15x15')
config = {
    'max_episodes': 1000,
    'use_wandb': True,
    'wandb_project': 'my-marl-research'
}

controller = ModernMultiAgentController(env, config, algorithm='mappo')
controller.train()
```

## 🌐 GitHub Pages Deployment

See detailed deployment guide below for step-by-step instructions to deploy the React interface to GitHub Pages.

## 🤝 Contributing

We welcome contributions from MARL beginners to experts!

### Easy Contributions
- Fix typos or improve comments
- Add algorithm examples
- Improve documentation

### Advanced Contributions
- New algorithm implementations
- Additional environments
- Performance optimizations

## 📚 Educational Resources

### For Beginners
- Extensive code comments in Simple Controller
- Algorithm descriptions with pros/cons
- Step-by-step tutorials in GUI
- Progressive complexity from simple to advanced

### For Instructors
- Ready-to-use classroom demonstrations
- Cross-platform web interface
- Real-time training visualization
- Multiple difficulty levels

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original project adapted from natashamjacques/multigrid
- Built with modern frameworks: Gradio, React, Flask, PyTorch
- Community contributions and feedback

---

**Ready to start your MARL journey? Choose your interface and begin exploring!** 🎯
