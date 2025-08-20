# 🚀 EasyMARL Deployment Guide

This guide covers the **3 primary ways** to deploy and use EasyMARL, designed to meet different user needs and preferences.

## 📋 Quick Comparison

| Method | Best For | Setup Time | Features | Requirements |
|--------|----------|------------|----------|--------------|
| **🌐 React + Flask** | Full development, local control | 5 minutes | Complete web interface, all features | Git, Node.js, Python |
| **☁️ GitHub Codespaces** | Students, zero-setup, cloud | 30 seconds | Browser-based, GPU support | GitHub account |
| **🐍 Gradio (Python)** | Quick experiments, notebooks | 30 seconds | Self-contained, educational mode | Python only |

---

## 🌐 Method 1: React Frontend + Flask Backend (Local)

**Perfect for**: Developers who want full control and all features locally.

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/shreyanmitra/EasyMARL.git
cd EasyMARL

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install React dependencies
cd gui/react-frontend
npm install
cd ../..

# 4. Start both services with one command
./tools/start-easymarl.sh
```

### Manual Service Management

```bash
# Start backend only (Terminal 1)
cd EasyMARL
python api/flask_backend.py

# Start frontend only (Terminal 2)  
cd EasyMARL/gui/react-frontend
npm start
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000/api
- **Features**: Real-time training monitoring, experiment tracking, all algorithms

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 3000/5000 in use | Change ports in environment variables |
| React build fails | Run `npm install` in react-frontend directory |
| Python import errors | Ensure all dependencies installed with `pip install -r requirements.txt` |

---

## ☁️ Method 2: GitHub Codespaces (Cloud)

**Perfect for**: Students, educators, and anyone wanting zero-setup cloud development.

### Setup (30 seconds)

1. **Click to start**: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/shreyanmitra/EasyMARL)

2. **Wait for setup**: Environment automatically configures (3-5 minutes)

3. **Auto-start**: Services start automatically when Codespace is ready

### Access Your Environment
```
Frontend: https://CODESPACE-NAME-3000.app.github.dev
Backend:  https://CODESPACE-NAME-5000.app.github.dev/api
```

Replace `CODESPACE-NAME` with your actual Codespace name (shown in browser URL).

### Benefits
- ✅ **FREE**: 120+ hours/month with GitHub account (180 with Student Pack)
- ✅ **No Installation**: Everything pre-configured in the cloud
- ✅ **GPU Support**: Available for intensive training
- ✅ **8GB RAM + 4 CPU**: Powerful cloud environment
- ✅ **Collaborative**: Share your Codespace URL with team members

### GitHub Student Pack
- **Extra Benefits**: More free hours, advanced features
- **Apply**: [GitHub Student Pack](https://education.github.com/pack)

---

## 🐍 Method 3: Gradio Interface (Python Library)

**Perfect for**: Quick experimentation, Jupyter notebooks, and research workflows.

### Simplest Setup

```bash
# Install EasyMARL
pip install easymarl

# Launch GUI in one line
python -c "import easymarl; easymarl.launch_gui()"
```

### Windows Quick Start
```cmd
# Download and run the quick-start script
curl -O https://raw.githubusercontent.com/shreyanmitra/EasyMARL/main/quick-start.bat
quick-start.bat
```

### Python Script Usage

```python
import easymarl

# Method A: Launch web interface
easymarl.launch_gui(
    port=7860,              # Custom port
    share=False,            # Set True for public link
    educational_mode=True   # Detailed explanations
)

# Method B: Direct API usage
env = easymarl.make_env("MultiGrid-Empty-6x6-v0")
controller = easymarl.UnifiedMultiAgentController(
    env=env,
    algorithm="qmix",
    educational_mode=True
)

# Train with educational output
results = controller.train(episodes=1000)
print(f"Training completed! Average reward: {results['avg_reward']:.2f}")
```

### Jupyter Notebook Integration

```python
# In a Jupyter cell
import easymarl

# Quick experiment
env = easymarl.make_env("MultiGrid-Cluttered-Fixed-15x15")
controller = easymarl.UnifiedMultiAgentController(
    env=env, 
    algorithm="ippo",
    educational_mode=True
)

# This will show detailed explanations
controller.train(episodes=100)
```

### Command Line Tools

After installing EasyMARL, you get command-line tools:

```bash
# Launch GUI from anywhere
easymarl-gui

# Train directly from command line
easymarl-train --algorithm qmix --episodes 1000

# Run demo
easymarl-demo
```

---

## 🔧 Configuration Options

### Environment Variables

```bash
# For local development
export FLASK_PORT=5000
export REACT_PORT=3000
export WANDB_API_KEY=your_key_here

# For Codespaces (auto-configured)
export CODESPACE_NAME=your-codespace-name
```

### Algorithm Selection
All methods support these algorithms:
- **Policy-Based**: IPPO, MAPPO, MADDPG
- **Value-Based**: QMIX, VDN, IQL, QTRAN
- **Actor-Critic**: COMA, MAACC
- **Advanced**: MAVEN, DCG, Nash-Q

---

## 🚨 Troubleshooting

### Common Issues

| Problem | Method 1 (Local) | Method 2 (Codespaces) | Method 3 (Gradio) |
|---------|-------------------|------------------------|-------------------|
| **Port conflicts** | Change PORT env vars | Automatic handling | Use `port` parameter |
| **Dependencies missing** | `pip install -r requirements.txt` | Pre-installed | `pip install easymarl[all]` |
| **GUI not loading** | Check React build | Refresh browser | Check Gradio version |
| **Training slow** | Use vectorization | Enable GPU | Install enhanced features |

### Performance Optimization

```bash
# Install enhanced features for 10x speedup
pip install easymarl[enhanced]

# Enable vectorization
controller = easymarl.UnifiedMultiAgentController(
    env=env,
    vectorization=True,  # 8x faster training
    num_envs=8
)
```

---

## 📞 Support

- **📧 Email**: shreyan.m.mitra@gmail.com
- **💬 Discussions**: [GitHub Discussions](https://github.com/shreyanmitra/EasyMARL/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/shreyanmitra/EasyMARL/issues)
- **📖 Documentation**: [Official Docs](https://shreyanmitra.github.io/EasyMARL)

---

## 🎯 Which Method Should I Choose?

### Choose **React + Flask** if:
- You want full development features
- You're building on top of EasyMARL
- You need access to all configuration options
- You prefer local development

### Choose **GitHub Codespaces** if:
- You're a student or educator
- You want zero setup time
- You need to share your environment
- You want cloud-based development

### Choose **Gradio (Python)** if:
- You want the quickest start
- You're using Jupyter notebooks
- You need a lightweight solution
- You're doing research experiments

**Still unsure?** Start with **Method 3 (Gradio)** for the quickest experience, then upgrade to Methods 1 or 2 as your needs grow!
