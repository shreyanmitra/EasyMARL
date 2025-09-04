# 🎓 EasyMARL - Educational Multi-Agent Reinforcement Learning Framework

[![PyPI version](https://badge.fury.io/py/easymarl.svg)](https://badge.fury.io/py/easymarl)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/shreyanmitra/EasyMARL.svg)](https://github.com/shreyanmitra/EasyMARL/stargazers)
[![Tests](https://img.shields.io/badge/Tests-Comprehensive-green.svg)](#testing)
[![CLI](https://img.shields.io/badge/CLI-Full_Support-blue.svg)](#command-line-interface)
[![Documentation](https://img.shields.io/badge/Docs-📖_Official-blue?logo=react)](https://shreyanmitra.github.io/EasyMARL)

> *Making Multi-Agent Reinforcement Learning accessible to everyone - from students to researchers*

📖 **[View Official Documentation](https://shreyanmitra.github.io/EasyMARL)** | 🚀 **[Quick Start Guide](https://shreyanmitra.github.io/EasyMARL/docs)** | 🧮 **[Algorithm Library](https://shreyanmitra.github.io/EasyMARL/algorithms)**

EasyMARL is a comprehensive, beginner-friendly framework for multi-agent reinforcement learning that bridges the gap between educational simplicity and research-ready performance. Whether you're learning MARL for the first time or conducting cutting-edge research, EasyMARL provides the tools you need.

## 🌟 Key Features

### 🎓 **Educational Excellence**
- **Beginner-Friendly Design**: Clear, well-documented code structure
- **Interactive GUI**: Web-based interface requiring no coding experience
- **Step-by-Step Learning**: Detailed tutorials and examples
- **Algorithm Comparisons**: Side-by-side performance analysis

### 🚀 **Research Ready**
- **🧠 21+ MARL Algorithms**: All algorithms comprehensively implemented in a single unified framework
- **📚 Single Algorithm Library**: All state-of-the-art algorithms organized by taxonomy in [`algorithms/__init__.py`](algorithms/__init__.py)
- **🖥️ Full CLI Support**: Complete command-line interface for training, evaluation, and management
- **🧪 Comprehensive Testing**: Extensive test suite ensuring reliability and correctness
- **🧠 Custom Neural Networks**: 4 built-in architectures (FeedForward, CNN, Attention, Residual) + custom network support
- **🔬 Enhanced Performance**: Vectorized environments and optional JIT compilation
- **📈 Scalable Architecture**: Handle complex multi-agent scenarios
- **📊 Professional Logging**: Weights & Biases integration

### 🎮 **Rich Environment Support**
- **MultiGrid Environments**: 12+ cooperative and competitive scenarios
- **Custom Environment API**: Easy integration of new environments
- **Real-time Visualization**: Watch agents learn in interactive environments

### 📊 **Advanced Analytics**
- **Real-time Monitoring**: Live training graphs and metrics
- **Experiment Tracking**: Comprehensive experiment management
- **Performance Profiling**: Memory and compute usage analysis
- **Video Generation**: Create videos of trained agent behavior

## 🚀 Four Ways to Use EasyMARL

EasyMARL offers **4 primary deployment methods** to suit different needs:

### ⚡ **Quick Start Options**

```bash
# Method 1: Command Line Interface (30 seconds)
pip install easymarl && easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0

# Method 2: Python Library (30 seconds)
pip install easymarl && python -c "import easymarl; easymarl.launch_gui()"

# Method 3: GitHub Codespaces (30 seconds)  
# Click: https://codespaces.new/shreyanmitra/EasyMARL

# Method 4: Local Web Interface (5 minutes)
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL && ./tools/start-easymarl.sh
```

### 🖥️ **Method 1: Command Line Interface (NEW!)**

**Best for**: Researchers, batch experiments, automated workflows

```bash
# Install EasyMARL
pip install easymarl

# Train agents with CLI
easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0 --episodes 1000

# Launch GUI
easymarl-gui

# Run demos
easymarl-demo --example comparison --quick

# Manage environments
easymarl-env --list
easymarl-env --info MultiGrid-Empty-6x6-v0

# Algorithm information
easymarl-algo --list
easymarl-algo --info qmix
easymarl-algo --compare qmix vdn ippo

# Weights & Biases integration
easymarl-wandb --login
easymarl-wandb --project my_research
```

**Features**: 
- ✅ **Complete CLI suite** with 6 main commands
- ✅ **Batch experiment support** for research workflows  
- ✅ **Algorithm comparison tools** built-in
- ✅ **Environment management** and validation
- ✅ **Weights & Biases integration** for experiment tracking
- ✅ **Comprehensive help system** and documentation

### 🌐 **Method 2: React Frontend + Flask Backend (Local)**

**Best for**: Full-featured development, complete control, local resources

```bash
# Install EasyMARL
pip install easymarl

# Clone repository for web interface
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL

# Start both frontend and backend
./tools/start-easymarl.sh

# Access web interface:
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000/api
```

**Features**: 
- ✅ Complete web interface with real-time training monitoring
- ✅ REST API for programmatic access
- ✅ All algorithms and environments available
- ✅ Experiment tracking with Weights & Biases

### ☁️ **Method 3: GitHub Codespaces (Cloud)**

**Best for**: Students, no-setup experience, cloud development

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shreyanmitra/EasyMARL)

```bash
# 1. Click "Open in GitHub Codespaces" above
# 2. Wait 3-5 minutes for automatic environment setup
# 3. Services start automatically on creation

# Access your cloud environment:
# Frontend: https://CODESPACE-NAME-3000.app.github.dev  
# Backend: https://CODESPACE-NAME-5000.app.github.dev/api
```

**Benefits**: 
- ✅ **FREE with GitHub Student Pack** (180 hours/month)
- ✅ **Zero installation** - works in your browser
- ✅ **Full ML environment** with GPU support
- ✅ **8GB RAM + 4 CPU cores**

### 🐍 **Method 4: Gradio Interface (Python Library)**

**Best for**: Quick experimentation, Jupyter notebooks, research workflows

```python
# Install and use directly in Python
pip install easymarl

import easymarl

# Launch Gradio web interface (one line!)
easymarl.launch_gui()

# Or use programmatic API
env = easymarl.make_env("MultiGrid-Empty-6x6-v0")
controller = easymarl.UnifiedMultiAgentController(
    env=env,
    algorithm="qmix",
    educational_mode=True  # Detailed explanations for learning
)

# Train agents with progress tracking
controller.train(episodes=1000)
results = controller.evaluate()
print(f"Average reward: {results['avg_reward']:.2f}")
```

**Advantages**:
- ✅ **Simplest setup** - just one `pip install`
- ✅ **Jupyter notebook friendly**
- ✅ **Educational mode** with detailed explanations
- ✅ **Self-contained** - no separate backend needed

## 🖥️ Command Line Interface

EasyMARL provides a comprehensive CLI for all operations:

```bash
# Training and evaluation
easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0 --episodes 1000
easymarl-train --algorithm ippo --vectorized --n_envs 8

# GUI and demos
easymarl-gui --interface gradio
easymarl-demo --example comparison --quick

# Environment management
easymarl-env --list
easymarl-env --info MultiGrid-Empty-6x6-v0
easymarl-env --create my_env --template cluttered

# Algorithm information and comparison
easymarl-algo --list
easymarl-algo --info qmix
easymarl-algo --compare qmix vdn ippo --episodes 200
easymarl-algo --benchmark --env MultiGrid-Empty-6x6-v0

# Weights & Biases integration
easymarl-wandb --login
easymarl-wandb --project my_research --view
```

**CLI Features:**
- ✅ **6 main command groups** covering all functionality
- ✅ **Comprehensive help system** with examples
- ✅ **Algorithm comparison tools** for research
- ✅ **Environment validation** and testing
- ✅ **Experiment management** with W&B integration
- ✅ **Batch processing** support for large experiments

> **📖 [Complete CLI Documentation](docs/CLI.md)** - Detailed CLI reference and examples

## 🧠 Complete Algorithm Library - All 21+ Algorithms in One Framework

> **🎯 Unique Feature**: All MARL algorithms are implemented in a **single unified framework** with consistent APIs, comprehensive documentation, and taxonomical organization in [`algorithms/__init__.py`](algorithms/__init__.py)

**� What Makes Our Algorithm Library Special:**
- ✅ **Unified Implementation**: All algorithms share the same base architecture
- ✅ **Taxonomical Organization**: Algorithms organized by their fundamental principles  
- ✅ **Comprehensive Comments**: Every single line of algorithm code is documented
- ✅ **Educational Progression**: Beginner → Intermediate → Advanced learning path
- ✅ **Research Ready**: Production-quality implementations with latest optimizations

### Value-Based Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **QMIX** | [Rashid et al., 2018](https://arxiv.org/abs/1803.11485) | Monotonic value function factorization | Cooperative tasks with partial observability |
| **VDN** | [Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) | Simple value decomposition | Basic cooperative learning |
| **IQL** | Independent Q-Learning | Each agent learns independently | Baseline for comparison |
| **QTRAN** | [Son et al., 2019](https://arxiv.org/abs/1905.05408) | General value decomposition | Complex cooperative scenarios |

### Policy-Based Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **IPPO** | Independent PPO | Independent policy optimization | Continuous action spaces |
| **MAPPO** | [Yu et al., 2021](https://arxiv.org/abs/2103.01955) | Multi-agent PPO with centralized training | Large-scale cooperation |
| **MADDPG** | [Lowe et al., 2017](https://arxiv.org/abs/1706.02275) | Multi-agent DDPG | Continuous control tasks |
| **MADDPG+Comm** | MADDPG with communication | Communication-enabled MADDPG | Coordination requiring communication |

### Actor-Critic Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **COMA** | [Foerster et al., 2018](https://arxiv.org/abs/1705.08926) | Counterfactual multi-agent policy gradients | Credit assignment problems |
| **COMA+Comm** | COMA with communication | Communication-enabled COMA | Complex coordination |
| **MAACC** | Multi-Agent Actor-Critic-Critic | Advanced actor-critic architecture | Challenging cooperative tasks |
| **DCG** | [Zhang et al., 2018](https://arxiv.org/abs/1810.09202) | Deep coordination graphs | Structured multi-agent problems |
| **MAVEN** | [Mahajan et al., 2019](https://arxiv.org/abs/1910.07483) | Multi-agent variational exploration | Exploration-heavy environments |

### Game-Theoretic Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **NFSP** | [Heinrich & Silver, 2016](https://arxiv.org/abs/1603.01121) | Neural fictitious self-play | Two-player competitive games |
| **MINIMAX-Q** | [Littman, 1994](https://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf) | Minimax Q-learning | Zero-sum games |
| **WoLF-PHC** | [Bowling & Veloso, 2002](https://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf) | Win-or-learn-fast policy hill climbing | Mixed-motive games |

### Hierarchical Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **HQL** | Hierarchical Q-Learning | Multi-level decision making | Complex task decomposition |
| **LQL** | Layered Q-Learning | Structured hierarchical learning | Hierarchical environments |

### Mean Field Methods

| Algorithm | Paper | Description | Best For |
|-----------|-------|-------------|----------|
| **MFQ** | [Yang et al., 2018](https://arxiv.org/abs/1802.05438) | Mean field Q-learning | Large population games |

## 🎮 Supported Environments

### MultiGrid Environments

| Environment | Size | Agents | Difficulty | Description |
|-------------|------|--------|------------|-------------|
| **Empty** | 6x6, 8x8, 16x16 | 2-4 | ⭐ | Basic navigation and coordination |
| **FourRooms** | 19x19 | 2-4 | ⭐⭐ | Navigation through connected rooms |
| **DoorKey** | 6x6, 8x8 | 2-4 | ⭐⭐⭐ | Coordination to unlock doors |
| **Cluttered** | 6x6, 8x8 | 2-4 | ⭐⭐ | Navigation with obstacles |
| **Maze** | 6x6, 8x8 | 2-4 | ⭐⭐⭐ | Complex maze navigation |
| **CoinGame** | Variable | 2 | ⭐⭐⭐⭐ | Competitive coin collection |
| **Gather** | Variable | 2-8 | ⭐⭐⭐ | Resource gathering cooperation |

## 🧠 Custom Neural Networks

EasyMARL supports flexible custom neural network architectures that work with all algorithms:

### Built-in Network Types

| Network Type | Best For | Key Features |
|--------------|----------|--------------|
| **FeedForward** | Baselines, simple tasks | Multi-layer perceptron, configurable depth |
| **Convolutional** | Visual environments | CNN layers, spatial processing |
| **Attention** | Coordination, complex reasoning | Self-attention, positional encoding |
| **Residual** | Deep learning, stability | Skip connections, batch normalization |

### Usage Examples

```python
# Use convolutional network for visual tasks
config = {
    'network_type': 'convolutional',
    'conv_layers': [32, 64, 128],
    'hidden_dim': 256
}

# Use attention network for coordination
config = {
    'network_type': 'attention', 
    'num_heads': 8,
    'attention_dim': 128
}

# Create custom network
config = {
    'network_type': 'my_networks.CustomNetwork',
    'custom_param': 'value'
}
```

**📖 [Full Custom Networks Guide](networks/README.md)** | **🔧 [Example Script](examples/demo_custom_networks.py)**

## 🏗️ Framework Architecture

### Three Controller Types

#### 1. 🎓 **SimpleMultiAgentController** - For Beginners
```python
# Educational design with clear step-by-step learning
controller = easymarl.SimpleMultiAgentController(
    env=env,
    algorithm="qmix",
    config=config,
    use_enhanced_features=True  # Optional performance boost
)
```

#### 2. 🚀 **ModernMultiAgentController** - For Researchers
```python
# Research-ready with advanced features
controller = easymarl.ModernMultiAgentController(
    env=env,
    algorithm="qmix", 
    config=config,
    experiment_name="research_exp_1",
    enable_advanced_tracking=True,
    enable_performance_monitoring=True
)
```

#### 3. ⚡ **VectorizedController** - For Performance
```python
# Maximum performance optimization
controller = easymarl.VectorizedController(
    env_fn=lambda: easymarl.make_env("MultiGrid-Empty-6x6-v0"),
    num_envs=8,  # Parallel environments
    algorithm="qmix",
    use_enhanced_vectorization=True  # JIT compilation
)
```

## 📈 Performance Features

### Enhanced Vectorization (Faster Training)

```python
# Enable enhanced features for maximum performance
controller = easymarl.SimpleMultiAgentController(
    env=env,
    algorithm="qmix",
    use_enhanced_features=True
)

# Or use vectorized controller for parallel training
controller = easymarl.VectorizedController(
    env_fn=lambda: easymarl.make_env("MultiGrid-Empty-6x6-v0"),
    num_envs=16,  # 16 parallel environments
    use_enhanced_vectorization=True
)
```

## 📊 Performance Benchmarks

| Environment | Algorithm | Vectorized Envs | Performance Benefit |
|-------------|-----------|-----------------|-------------------|
| MultiGrid-Empty-6x6 | QMIX | 8 parallel | ~8x faster sampling |
| MultiGrid-Empty-8x8 | VDN | 8 parallel | ~8x faster sampling |
| MultiGrid-DoorKey-6x6 | COMA | 8 parallel | ~8x faster sampling |
| MultiGrid-Maze-8x8 | MADDPG | 8 parallel | ~8x faster sampling |

*Speedup from running 8 environments in parallel vs single environment*

## 🛠️ Installation Options

### Basic Installation
```bash
pip install easymarl
```

### Enhanced Performance
```bash
pip install easymarl[enhanced]
```

### With GUI Support
```bash
pip install easymarl[gui]
```

### Complete Installation
```bash
pip install easymarl[all]
```

### Development Installation
```bash
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL
pip install -e .[all]
```

## 📁 Project Structure

EasyMARL follows a clean, modular package structure designed for both ease of use and professional development:

```
EasyMARL/
├── core/                           # Core MARL functionality
│   ├── utils/                      # Consolidated utilities
│   │   ├── base.py                 # Core utilities
│   │   ├── advanced.py             # Advanced features
│   │   └── enhanced.py             # Performance optimization
│   ├── config_manager.py           # Configuration management
│   └── research_interface.py       # Research tools
├── algorithms/                     # 21+ MARL algorithms (ALL IMPLEMENTED)
│   ├── model_free/                # Model-free algorithms
│   │   ├── value_based/           # QMIX, VDN, QTRAN, IQL, MFQ + tabular
│   │   ├── policy_based/          # IPPO, MAPPO, MADDPG variants  
│   │   └── actor_critic/          # COMA, MAVEN, DCG, NFSP, MAACC
│   ├── model_based/               # Model-based approaches (future)
│   ├── base.py                    # Algorithm base classes
│   └── taxonomy.py                # Algorithm classification
├── environments/                   # Environment management
│   ├── vectorized_env.py          # Vectorized environments  
│   └── gym_multigrid/             # MultiGrid environments
├── controllers/                    # Training controllers
│   └── unified_multiagent_controller.py  # Unified controller (IMPLEMENTED)
├── networks/                       # Neural network architectures
│   ├── base_network.py            # Base network classes
│   ├── custom_networks.py         # 4 built-in architectures
│   └── multigrid_network.py       # MultiGrid-specific networks
├── cli/                           # Command Line Interface (NEW!)
│   ├── main_cli.py                # Training commands (easymarl-train)
│   ├── wandb_cli.py               # W&B integration (easymarl-wandb) 
│   ├── env_cli.py                 # Environment management (easymarl-env)
│   └── algo_cli.py                # Algorithm info (easymarl-algo)
├── tests/                         # Comprehensive Test Suite (NEW!)
│   ├── test_algorithms.py         # Algorithm functionality tests
│   ├── test_environments.py       # Environment validation tests
│   ├── test_controllers.py        # Controller integration tests
│   ├── test_utils.py              # Utility function tests
│   ├── test_integration.py        # End-to-end integration tests
│   └── __main__.py                # CLI test runner
├── docs/                          # Documentation (NEW!)
│   ├── CLI.md                     # Complete CLI documentation
│   ├── API.md                     # API reference documentation
│   └── TESTING.md                 # Testing framework documentation
├── api/                           # Web API and deployment
│   ├── flask_backend.py           # Main API server
│   ├── minimal.py                 # Lightweight deployment
│   └── deployment/                # Deployment configs
├── gui/                           # User interfaces
│   ├── gradio_interface.py        # Web-based GUI
│   └── react-frontend/            # React components
├── config/                        # Configuration templates
│   ├── default.yaml               # Default settings
│   ├── domain/                    # Environment configs
│   ├── mode/                      # Algorithm configs
│   └── templates/                 # Config templates
├── examples/                      # Tutorial examples
├── tools/                         # Utility scripts
│   ├── start-easymarl.sh          # Start both frontend and backend
│   ├── start-backend.sh           # Start Flask backend only
│   └── start-frontend.sh          # Start React frontend only
├── main.py                        # Main CLI entry point
├── setup.py                       # Package configuration
└── requirements.txt               # Dependencies
```

### Key Entry Points

- **CLI**: `easymarl-train --algorithm qmix` or `python main.py`
- **GUI**: `easymarl-gui` or `python -c "import easymarl; easymarl.launch_gui()"`
- **API**: Start with `./tools/start-easymarl.sh` or individual scripts
- **Tests**: `python -m tests` or `python -m tests --quick`
- **Import**: `import easymarl` (for Python integration)

## 🧪 Testing

EasyMARL includes a comprehensive test suite to ensure reliability:

```bash
# Run all tests
python -m tests

# Run quick validation tests  
python -m tests --quick

# Run specific test suite
python -m tests --suite algorithms
python -m tests --suite environments

# List available test suites
python -m tests --list

# Verbose test output
python -m tests --verbose
```

**Test Coverage:**
- ✅ **Algorithm Tests**: Import, creation, basic training functionality
- ✅ **Environment Tests**: Creation, reset/step, vectorization
- ✅ **Controller Tests**: Unified controller integration
- ✅ **Utility Tests**: Configuration, advanced features, research tools
- ✅ **Integration Tests**: End-to-end pipeline, CLI, API, GUI
- ✅ **CLI Test Runner**: Easy test management and reporting

> **📖 [Complete Testing Documentation](docs/TESTING.md)** - Detailed testing guide and best practices

## 🎛️ Web-Based GUI

Launch the professional web interface:

```python
easymarl.launch_gui()
```

### GUI Features

#### 📋 **Algorithm Selection Tab**
- Interactive algorithm descriptions
- Performance comparisons
- Hyperparameter explanations
- Real-time algorithm switching

#### ⚙️ **Configuration Tab**
- Intuitive parameter tuning
- Real-time validation
- Configuration presets
- Export/import configurations

#### 🚀 **Training Tab**
- One-click training start
- Real-time progress monitoring
- Live performance graphs
- Training status updates

#### 📊 **Results Tab**
- Comprehensive performance analysis
- Training curve visualization
- Statistical summaries
- Data export capabilities

#### � **Visualization Tab**
- Watch trained agents in action
- Environment interaction videos
- Agent behavior analysis
- Custom rendering options

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[all]

# Run tests
python -m tests
python -m tests --quick

# Run specific test suites
python -m tests --suite algorithms
python -m tests --suite integration

# Run linting (if available)
flake8 algorithms/ controllers/ core/ cli/ api/ gui/
black algorithms/ controllers/ core/ cli/ api/ gui/
```

### Requirements

- **Python 3.11+** (Required)
- **Node.js 16+** (For React frontend only)
- **Git** (For cloning and development)

**System Requirements:**
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **GPU**: Optional but recommended for large experiments

## 📄 Citation

If you use EasyMARL in your research, please cite:

```bibtex
@software{easymarl2024,
  title={EasyMARL: Educational Multi-Agent Reinforcement Learning Framework},
  author={Mitra, Shreyan},
  year={2024},
  url={https://github.com/shreyanmitra/EasyMARL}
}
```

## 🌟 Why EasyMARL?

### For Students
- **Zero barriers to entry**: Start learning MARL in minutes
- **Interactive learning**: Visual feedback and real-time monitoring
- **Comprehensive coverage**: All major MARL paradigms included
- **Educational design**: Code structure mirrors textbook concepts

### For Researchers  
- **Research ready**: Scale from prototype to publication
- **Comprehensive algorithms**: 21+ state-of-the-art implementations
- **CLI for automation**: Complete command-line interface for batch experiments
- **Comprehensive testing**: Extensive test suite ensuring reliability
- **Experiment management**: Professional tracking and analysis
- **Extensible framework**: Easy to add new algorithms and environments

### For Educators
- **Classroom ready**: GUI requires no programming experience
- **Comparative analysis**: Easy algorithm comparisons
- **Visual learning**: Rich visualizations and animations
- **Local development**: Optimized for GitHub Codespaces and local environments

## 🎯 Deployment Options

Choose the deployment method that best fits your needs:

| Method | Best For | Setup Time | Features | Cost |
|--------|----------|------------|----------|------|
| **CLI Commands** | Research, automation, batch experiments | 30 seconds | Complete CLI suite, algorithm comparison | Free |
| **React + Flask** | Full development, local control | 5 minutes | Complete web interface, all features | Free |
| **GitHub Codespaces** | Students, zero-setup, cloud | 30 seconds | Browser-based, GPU support | Free* |
| **Gradio (Python)** | Quick experiments, notebooks | 30 seconds | Self-contained, educational mode | Free |

*Free with GitHub Student Pack (180 hours/month)

### Quick Setup Commands

```bash
# Method 1: CLI Commands (NEW!)
pip install easymarl
easymarl-train --algorithm qmix --env MultiGrid-Empty-6x6-v0

# Method 2: Local React + Flask
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL && ./tools/start-easymarl.sh

# Method 3: GitHub Codespaces  
# Click: https://codespaces.new/shreyanmitra/EasyMARL

# Method 4: Gradio (Python)
pip install easymarl && python -c "import easymarl; easymarl.launch_gui()"
```

## �📞 Support

- **📧 Email**: shreyan.m.mitra@gmail.com
- **💬 Discussions**: [GitHub Discussions](https://github.com/shreyanmitra/EasyMARL/discussions)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/shreyanmitra/EasyMARL/issues)
- **📖 Documentation**: [Official Docs](https://shreyanmitra.github.io/EasyMARL)

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Natasha Jaques** - Original metacontroller framework inspiration
- **DeepMind** - QMIX and related algorithm implementations
- **OpenAI** - Multi-agent environment design principles
- **The MARL Community** - Continuous feedback and contributions

---

<div align="center">

**🎓 Making Multi-Agent Reinforcement Learning Accessible to Everyone 🎓**

[![GitHub](https://img.shields.io/badge/GitHub-EasyMARL-blue?logo=github)](https://github.com/shreyanmitra/EasyMARL)
[![PyPI](https://img.shields.io/badge/PyPI-easymarl-blue?logo=pypi)](https://pypi.org/project/easymarl/)
[![Documentation](https://img.shields.io/badge/Docs-Official-blue?logo=react)](https://shreyanmitra.github.io/EasyMARL)

</div>
