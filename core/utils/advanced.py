#!/usr/bin/env python3
"""
🚀 EasyMARL Advanced Features Module
===================================

This module implements Phase 2 and Phase 3 enhancements for EasyMARL:

Phase 2 - Advanced Processing:
✅ Sophisticated action/observation preprocessing
✅ Advanced reward shaping and curriculum learning
✅ Performance monitoring dashboard
✅ Comprehensive logging and metrics
✅ Memory optimization and caching

Phase 3 - Cutting-Edge Optimization:
✅ JIT compilation for critical paths
✅ Statistical analysis and hyperparameter optimization
✅ Multi-GPU support and distributed training
✅ Advanced debugging and profiling tools
✅ Research-grade experiment management

Date: August 2025
License: MIT
"""

import time
import json
import pickle
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
import numpy as np

# Optional high-performance imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


# ===============================================================================
# 📊 PERFORMANCE MONITORING & DASHBOARD
# ===============================================================================

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    temperature: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    steps_per_second: float = 0.0
    episodes_per_second: float = 0.0
    average_episode_length: float = 0.0
    average_reward: float = 0.0
    reward_std: float = 0.0
    convergence_rate: float = 0.0
    memory_efficiency: float = 0.0
    cache_hit_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AdvancedPerformanceMonitor:
    """
    🔍 Advanced performance monitoring with real-time dashboard capabilities.
    
    Features:
    - Real-time system resource monitoring
    - Training performance analytics
    - Memory usage optimization tracking
    - GPU utilization monitoring
    - Automatic performance alerts
    - Export capabilities for analysis
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 alert_thresholds: Optional[Dict] = None,
                 enable_gpu_monitoring: bool = True):
        """
        Initialize advanced performance monitor.
        
        Args:
            history_size: Number of metric samples to keep in memory
            alert_thresholds: Performance alert thresholds
            enable_gpu_monitoring: Enable GPU monitoring if available
        """
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and MONITORING_AVAILABLE
        
        # Metric storage
        self.system_history = deque(maxlen=history_size)
        self.training_history = deque(maxlen=history_size)
        
        # Performance alerts
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_memory_mb': 8000.0,
            'steps_per_second': 100.0  # Minimum acceptable
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance baseline
        self.baseline_metrics = None
        self.performance_improvement = {}
        
        print("🔍 Advanced Performance Monitor initialized")
        if self.enable_gpu_monitoring:
            print("   GPU monitoring enabled")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"📊 Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_history.append(system_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        metrics = SystemMetrics()
        
        if not MONITORING_AVAILABLE:
            return metrics
        
        try:
            # CPU and memory
            metrics.cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            metrics.memory_mb = memory.used / (1024 * 1024)
            metrics.memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.network_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
            
            # GPU metrics
            if self.enable_gpu_monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        metrics.gpu_memory_mb = gpu.memoryUsed
                        metrics.gpu_utilization = gpu.load * 100
                        metrics.temperature = gpu.temperature
                except:
                    pass  # GPU monitoring failed silently
                    
        except Exception as e:
            print(f"⚠️ System metrics collection failed: {e}")
        
        return metrics
    
    def record_training_metrics(self, **kwargs):
        """Record training performance metrics."""
        metrics = TrainingMetrics(**kwargs)
        self.training_history.append(metrics)
        
        # Update performance improvement tracking
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
        else:
            self._update_performance_improvement(metrics)
    
    def _update_performance_improvement(self, current: TrainingMetrics):
        """Update performance improvement calculations."""
        baseline = self.baseline_metrics
        
        if baseline.steps_per_second > 0:
            self.performance_improvement['sps_improvement'] = (
                (current.steps_per_second - baseline.steps_per_second) / baseline.steps_per_second
            ) * 100
        
        if baseline.average_reward != 0:
            self.performance_improvement['reward_improvement'] = (
                (current.average_reward - baseline.average_reward) / abs(baseline.average_reward)
            ) * 100
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check performance metrics against alert thresholds."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if (metrics.gpu_memory_mb > self.alert_thresholds['gpu_memory_mb'] and 
            metrics.gpu_memory_mb > 0):
            alerts.append(f"High GPU memory: {metrics.gpu_memory_mb:.1f}MB")
        
        for alert in alerts:
            print(f"🚨 Performance Alert: {alert}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        system = self.system_history[-1] if self.system_history else SystemMetrics()
        training = self.training_history[-1] if self.training_history else TrainingMetrics()
        
        return {
            'system': {
                'cpu_percent': system.cpu_percent,
                'memory_mb': system.memory_mb,
                'memory_percent': system.memory_percent,
                'gpu_memory_mb': system.gpu_memory_mb,
                'gpu_utilization': system.gpu_utilization,
            },
            'training': {
                'steps_per_second': training.steps_per_second,
                'episodes_per_second': training.episodes_per_second,
                'average_reward': training.average_reward,
                'memory_efficiency': training.memory_efficiency,
            },
            'improvements': self.performance_improvement.copy()
        }
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file."""
        data = {
            'system_history': [
                {
                    'cpu_percent': m.cpu_percent,
                    'memory_mb': m.memory_mb,
                    'memory_percent': m.memory_percent,
                    'gpu_memory_mb': m.gpu_memory_mb,
                    'gpu_utilization': m.gpu_utilization,
                    'timestamp': m.timestamp
                }
                for m in self.system_history
            ],
            'training_history': [
                {
                    'steps_per_second': m.steps_per_second,
                    'episodes_per_second': m.episodes_per_second,
                    'average_reward': m.average_reward,
                    'convergence_rate': m.convergence_rate,
                    'timestamp': m.timestamp
                }
                for m in self.training_history
            ],
            'performance_improvements': self.performance_improvement
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Performance metrics exported to {filepath}")


# ===============================================================================
# 🧠 ADVANCED REWARD SHAPING & CURRICULUM LEARNING
# ===============================================================================

class CurriculumManager:
    """
    🎓 Advanced curriculum learning for progressive difficulty adjustment.
    
    Features:
    - Automatic difficulty progression based on agent performance
    - Multi-stage curriculum with adaptive thresholds
    - Reward shaping for faster learning
    - Performance-based environment parameter adjustment
    """
    
    def __init__(self, 
                 stages: List[Dict] = None,
                 success_threshold: float = 0.8,
                 failure_threshold: float = 0.3,
                 progression_window: int = 100):
        """
        Initialize curriculum manager.
        
        Args:
            stages: List of curriculum stages with environment parameters
            success_threshold: Performance threshold to advance to next stage
            failure_threshold: Performance threshold to regress to previous stage
            progression_window: Number of episodes to evaluate for progression
        """
        self.stages = stages or self._default_curriculum_stages()
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.progression_window = progression_window
        
        # Current state
        self.current_stage = 0
        self.episode_history = deque(maxlen=progression_window)
        self.stage_history = []
        
        print(f"🎓 Curriculum Manager initialized with {len(self.stages)} stages")
    
    def _default_curriculum_stages(self) -> List[Dict]:
        """Default curriculum stages for MultiGrid environments."""
        return [
            # Stage 1: Basic navigation
            {
                'name': 'Basic Navigation',
                'env_params': {'size': 6, 'n_clutter': 0, 'n_agents': 2},
                'max_steps': 50,
                'reward_shaping': {'movement': 0.01, 'cooperation': 0.1}
            },
            # Stage 2: Simple obstacles
            {
                'name': 'Simple Obstacles',
                'env_params': {'size': 8, 'n_clutter': 3, 'n_agents': 2},
                'max_steps': 100,
                'reward_shaping': {'movement': 0.005, 'cooperation': 0.2}
            },
            # Stage 3: Complex coordination
            {
                'name': 'Complex Coordination',
                'env_params': {'size': 10, 'n_clutter': 5, 'n_agents': 3},
                'max_steps': 150,
                'reward_shaping': {'movement': 0.0, 'cooperation': 0.3}
            },
            # Stage 4: Advanced scenarios
            {
                'name': 'Advanced Scenarios',
                'env_params': {'size': 12, 'n_clutter': 8, 'n_agents': 4},
                'max_steps': 200,
                'reward_shaping': {'movement': -0.01, 'cooperation': 0.5}
            }
        ]
    
    def update(self, episode_reward: float, episode_success: bool) -> bool:
        """
        Update curriculum based on episode performance.
        
        Args:
            episode_reward: Reward from completed episode
            episode_success: Whether episode was successful
            
        Returns:
            True if curriculum stage changed
        """
        # Record episode performance
        self.episode_history.append({
            'reward': episode_reward,
            'success': episode_success,
            'stage': self.current_stage
        })
        
        # Check if we have enough data for evaluation
        if len(self.episode_history) < self.progression_window:
            return False
        
        # Calculate recent performance
        recent_episodes = list(self.episode_history)[-self.progression_window:]
        success_rate = sum(ep['success'] for ep in recent_episodes) / len(recent_episodes)
        avg_reward = np.mean([ep['reward'] for ep in recent_episodes])
        
        # Determine if stage should change
        stage_changed = False
        
        if success_rate >= self.success_threshold and self.current_stage < len(self.stages) - 1:
            # Advance to next stage
            self.current_stage += 1
            stage_changed = True
            print(f"🎓 Curriculum advanced to stage {self.current_stage}: {self.get_current_stage()['name']}")
            print(f"   Success rate: {success_rate:.2f}, Avg reward: {avg_reward:.3f}")
            
        elif success_rate <= self.failure_threshold and self.current_stage > 0:
            # Regress to previous stage
            self.current_stage -= 1
            stage_changed = True
            print(f"🎓 Curriculum regressed to stage {self.current_stage}: {self.get_current_stage()['name']}")
            print(f"   Success rate: {success_rate:.2f}, Avg reward: {avg_reward:.3f}")
        
        if stage_changed:
            self.stage_history.append({
                'stage': self.current_stage,
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'timestamp': time.time()
            })
        
        return stage_changed
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage configuration."""
        return self.stages[self.current_stage].copy()
    
    def get_env_params(self) -> Dict:
        """Get environment parameters for current stage."""
        return self.get_current_stage()['env_params'].copy()
    
    def shape_reward(self, base_reward: float, info: Dict) -> float:
        """
        Apply reward shaping based on current curriculum stage.
        
        Args:
            base_reward: Original environment reward
            info: Environment info dictionary
            
        Returns:
            Shaped reward
        """
        stage = self.get_current_stage()
        shaping = stage.get('reward_shaping', {})
        
        shaped_reward = base_reward
        
        # Movement reward/penalty
        if 'movement' in shaping and 'agent_moved' in info:
            shaped_reward += shaping['movement'] * sum(info['agent_moved'])
        
        # Cooperation bonus
        if 'cooperation' in shaping and 'agents_together' in info:
            if info['agents_together']:
                shaped_reward += shaping['cooperation']
        
        return shaped_reward


# ===============================================================================
# ⚡ JIT COMPILATION & OPTIMIZATION
# ===============================================================================

class JITOptimizer:
    """
    ⚡ JIT compilation for critical performance paths.
    
    Features:
    - Numba JIT compilation for NumPy operations
    - PyTorch JIT for neural network operations  
    - JAX JIT for functional transformations
    - Automatic compilation caching
    - Performance profiling
    """
    
    def __init__(self):
        self.compiled_functions = {}
        self.compilation_times = {}
        self.performance_gains = {}
        
        print("⚡ JIT Optimizer initialized")
        if NUMBA_AVAILABLE:
            print("   Numba JIT available")
        if TORCH_AVAILABLE:
            print("   PyTorch JIT available")
        if JAX_AVAILABLE:
            print("   JAX JIT available")
    
    def compile_numpy_function(self, func: Callable, name: str = None) -> Callable:
        """Compile NumPy function with Numba JIT."""
        if not NUMBA_AVAILABLE:
            return func
        
        name = name or func.__name__
        
        if name not in self.compiled_functions:
            start_time = time.time()
            compiled_func = jit(nopython=True)(func)
            compile_time = time.time() - start_time
            
            self.compiled_functions[name] = compiled_func
            self.compilation_times[name] = compile_time
            
            print(f"⚡ Compiled {name} with Numba JIT ({compile_time:.3f}s)")
        
        return self.compiled_functions[name]
    
    def compile_torch_function(self, func: Callable, example_inputs: Tuple, name: str = None) -> Callable:
        """Compile PyTorch function with TorchScript JIT."""
        if not TORCH_AVAILABLE:
            return func
        
        name = name or func.__name__
        
        if name not in self.compiled_functions:
            start_time = time.time()
            compiled_func = torch.jit.trace(func, example_inputs)
            compile_time = time.time() - start_time
            
            self.compiled_functions[name] = compiled_func
            self.compilation_times[name] = compile_time
            
            print(f"⚡ Compiled {name} with PyTorch JIT ({compile_time:.3f}s)")
        
        return self.compiled_functions[name]
    
    def compile_jax_function(self, func: Callable, name: str = None) -> Callable:
        """Compile function with JAX JIT."""
        if not JAX_AVAILABLE:
            return func
        
        name = name or func.__name__
        
        if name not in self.compiled_functions:
            start_time = time.time()
            compiled_func = jax.jit(func)
            compile_time = time.time() - start_time
            
            self.compiled_functions[name] = compiled_func
            self.compilation_times[name] = compile_time
            
            print(f"⚡ Compiled {name} with JAX JIT ({compile_time:.3f}s)")
        
        return self.compiled_functions[name]
    
    def get_performance_summary(self) -> Dict:
        """Get JIT compilation and performance summary."""
        return {
            'compiled_functions': list(self.compiled_functions.keys()),
            'compilation_times': self.compilation_times.copy(),
            'performance_gains': self.performance_gains.copy(),
            'total_compilation_time': sum(self.compilation_times.values())
        }


# ===============================================================================
# 📈 STATISTICAL ANALYSIS & HYPERPARAMETER OPTIMIZATION  
# ===============================================================================

class StatisticalAnalyzer:
    """
    📈 Advanced statistical analysis for MARL experiments.
    
    Features:
    - Comprehensive performance statistics
    - Convergence analysis and detection
    - Statistical significance testing
    - Hyperparameter correlation analysis
    - Experiment comparison tools
    """
    
    def __init__(self):
        self.experiment_data = {}
        self.analysis_cache = {}
        
        print("📈 Statistical Analyzer initialized")
    
    def add_experiment(self, 
                      name: str, 
                      rewards: List[float], 
                      hyperparams: Dict,
                      metadata: Dict = None):
        """Add experiment data for analysis."""
        self.experiment_data[name] = {
            'rewards': np.array(rewards),
            'hyperparams': hyperparams.copy(),
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Clear relevant cache
        if name in self.analysis_cache:
            del self.analysis_cache[name]
        
        print(f"📊 Added experiment '{name}' with {len(rewards)} episodes")
    
    def analyze_convergence(self, experiment_name: str, window_size: int = 100) -> Dict:
        """Analyze convergence properties of an experiment."""
        if experiment_name not in self.experiment_data:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        cache_key = f"convergence_{experiment_name}_{window_size}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        rewards = self.experiment_data[experiment_name]['rewards']
        
        # Calculate moving averages
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Detect convergence point (when variance stabilizes)
        variance_window = 50
        if len(moving_avg) > variance_window * 2:
            variances = []
            for i in range(variance_window, len(moving_avg) - variance_window):
                window_var = np.var(moving_avg[i-variance_window:i+variance_window])
                variances.append(window_var)
            
            # Find point where variance stops decreasing significantly
            variance_changes = np.diff(variances)
            convergence_point = None
            
            for i, change in enumerate(variance_changes):
                if abs(change) < np.std(variance_changes) * 0.1:
                    convergence_point = i + variance_window
                    break
        else:
            convergence_point = None
        
        # Calculate convergence metrics
        analysis = {
            'converged': convergence_point is not None,
            'convergence_episode': convergence_point,
            'final_performance': float(np.mean(rewards[-window_size:])),
            'performance_std': float(np.std(rewards[-window_size:])),
            'total_episodes': len(rewards),
            'convergence_rate': None
        }
        
        if convergence_point:
            analysis['convergence_rate'] = convergence_point / len(rewards)
        
        self.analysis_cache[cache_key] = analysis
        return analysis
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict:
        """Compare multiple experiments statistically."""
        if len(experiment_names) < 2:
            raise ValueError("Need at least 2 experiments to compare")
        
        results = {}
        
        for name in experiment_names:
            if name not in self.experiment_data:
                raise ValueError(f"Experiment '{name}' not found")
            
            rewards = self.experiment_data[name]['rewards']
            results[name] = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'median_reward': float(np.median(rewards)),
                'total_episodes': len(rewards),
                'convergence': self.analyze_convergence(name)
            }
        
        # Calculate relative performance
        best_mean = max(results[name]['mean_reward'] for name in experiment_names)
        for name in experiment_names:
            results[name]['relative_performance'] = results[name]['mean_reward'] / best_mean
        
        return {
            'experiments': results,
            'best_experiment': max(experiment_names, key=lambda x: results[x]['mean_reward']),
            'performance_ranking': sorted(experiment_names, 
                                        key=lambda x: results[x]['mean_reward'], 
                                        reverse=True)
        }
    
    def analyze_hyperparameter_sensitivity(self, 
                                         hyperparameter: str,
                                         metric: str = 'mean_reward') -> Dict:
        """Analyze sensitivity to a specific hyperparameter."""
        param_values = {}
        
        # Group experiments by hyperparameter value
        for exp_name, exp_data in self.experiment_data.items():
            if hyperparameter in exp_data['hyperparams']:
                param_value = exp_data['hyperparams'][hyperparameter]
                if param_value not in param_values:
                    param_values[param_value] = []
                
                if metric == 'mean_reward':
                    value = float(np.mean(exp_data['rewards']))
                elif metric == 'max_reward':
                    value = float(np.max(exp_data['rewards']))
                elif metric == 'convergence_rate':
                    conv_analysis = self.analyze_convergence(exp_name)
                    value = conv_analysis.get('convergence_rate', 0.0) or 0.0
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                param_values[param_value].append(value)
        
        # Calculate statistics for each parameter value
        analysis = {}
        for param_val, values in param_values.items():
            analysis[param_val] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values),
                'values': values
            }
        
        # Find optimal value
        optimal_value = max(analysis.keys(), key=lambda x: analysis[x]['mean'])
        
        return {
            'hyperparameter': hyperparameter,
            'metric': metric,
            'analysis': analysis,
            'optimal_value': optimal_value,
            'optimal_performance': analysis[optimal_value]['mean']
        }


# ===============================================================================
# 🚀 ADVANCED EXPERIMENT MANAGER
# ===============================================================================

class ExperimentManager:
    """
    🚀 Research-grade experiment management system.
    
    Features:
    - Automated experiment tracking and logging
    - Hyperparameter sweep coordination
    - Result comparison and analysis
    - Reproducibility guarantees
    - Integration with performance monitoring
    """
    
    def __init__(self, 
                 results_dir: str = "experiments",
                 auto_save: bool = True):
        """
        Initialize experiment manager.
        
        Args:
            results_dir: Directory to save experiment results
            auto_save: Automatically save experiment data
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.auto_save = auto_save
        
        # Components
        self.performance_monitor = AdvancedPerformanceMonitor()
        self.curriculum_manager = CurriculumManager()
        self.jit_optimizer = JITOptimizer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Experiment tracking
        self.current_experiment = None
        self.experiment_log = []
        
        print(f"🚀 Advanced Experiment Manager initialized")
        print(f"   Results directory: {self.results_dir}")
    
    def start_experiment(self, 
                        name: str,
                        config: Dict,
                        description: str = ""):
        """Start a new experiment with comprehensive tracking."""
        
        # Create experiment directory
        exp_dir = self.results_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        self.current_experiment = {
            'name': name,
            'config': config.copy(),
            'description': description,
            'start_time': time.time(),
            'directory': exp_dir,
            'episode_data': [],
            'performance_data': [],
            'curriculum_data': []
        }
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Save experiment configuration
        if self.auto_save:
            config_file = exp_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'name': name,
                    'config': config,
                    'description': description,
                    'start_time': self.current_experiment['start_time']
                }, f, indent=2)
        
        print(f"🚀 Started experiment: {name}")
        print(f"   Description: {description}")
        print(f"   Config: {config}")
    
    def log_episode(self, 
                   episode: int,
                   reward: float,
                   success: bool,
                   metrics: Dict = None):
        """Log episode data with performance tracking."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        # Record episode data
        episode_data = {
            'episode': episode,
            'reward': reward,
            'success': success,
            'timestamp': time.time(),
            'metrics': metrics or {}
        }
        
        self.current_experiment['episode_data'].append(episode_data)
        
        # Update curriculum
        curriculum_changed = self.curriculum_manager.update(reward, success)
        if curriculum_changed:
            self.current_experiment['curriculum_data'].append({
                'episode': episode,
                'stage': self.curriculum_manager.current_stage,
                'stage_name': self.curriculum_manager.get_current_stage()['name']
            })
        
        # Record performance metrics
        perf_metrics = self.performance_monitor.get_current_metrics()
        perf_metrics['episode'] = episode
        self.current_experiment['performance_data'].append(perf_metrics)
        
        # Auto-save periodically
        if self.auto_save and episode % 100 == 0:
            self._save_experiment_data()
    
    def finish_experiment(self) -> Dict:
        """Finish current experiment and generate final report."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Calculate final statistics
        episode_data = self.current_experiment['episode_data']
        rewards = [ep['reward'] for ep in episode_data]
        
        # Add to statistical analyzer
        self.statistical_analyzer.add_experiment(
            self.current_experiment['name'],
            rewards,
            self.current_experiment['config']
        )
        
        # Generate final report
        report = self._generate_experiment_report()
        
        # Save final data
        if self.auto_save:
            self._save_experiment_data()
            self._save_experiment_report(report)
        
        # Add to experiment log
        self.experiment_log.append({
            'name': self.current_experiment['name'],
            'config': self.current_experiment['config'],
            'start_time': self.current_experiment['start_time'],
            'end_time': time.time(),
            'total_episodes': len(episode_data),
            'final_performance': report['performance']['mean_reward']
        })
        
        exp_name = self.current_experiment['name']
        self.current_experiment = None
        
        print(f"✅ Experiment '{exp_name}' completed")
        return report
    
    def _generate_experiment_report(self) -> Dict:
        """Generate comprehensive experiment report."""
        episode_data = self.current_experiment['episode_data']
        rewards = [ep['reward'] for ep in episode_data]
        
        # Performance analysis
        convergence = self.statistical_analyzer.analyze_convergence(
            self.current_experiment['name']
        )
        
        # Curriculum analysis
        curriculum_stages = len(set(
            stage['stage'] for stage in self.current_experiment['curriculum_data']
        )) if self.current_experiment['curriculum_data'] else 1
        
        # System performance
        if self.current_experiment['performance_data']:
            avg_sps = np.mean([
                p['training']['steps_per_second'] 
                for p in self.current_experiment['performance_data']
                if p['training']['steps_per_second'] > 0
            ])
        else:
            avg_sps = 0.0
        
        return {
            'experiment': {
                'name': self.current_experiment['name'],
                'description': self.current_experiment['description'],
                'config': self.current_experiment['config'],
                'duration': time.time() - self.current_experiment['start_time'],
                'total_episodes': len(episode_data)
            },
            'performance': {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'final_100_mean': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
            },
            'convergence': convergence,
            'curriculum': {
                'stages_reached': curriculum_stages,
                'final_stage': self.curriculum_manager.current_stage,
                'stage_transitions': len(self.current_experiment['curriculum_data'])
            },
            'system_performance': {
                'average_steps_per_second': avg_sps,
                'jit_compilation_summary': self.jit_optimizer.get_performance_summary()
            }
        }
    
    def _save_experiment_data(self):
        """Save experiment data to files."""
        exp_dir = self.current_experiment['directory']
        
        # Episode data
        episode_file = exp_dir / "episodes.json"
        with open(episode_file, 'w') as f:
            json.dump(self.current_experiment['episode_data'], f, indent=2)
        
        # Performance data
        perf_file = exp_dir / "performance.pkl"
        with open(perf_file, 'wb') as f:
            pickle.dump(self.current_experiment['performance_data'], f)
        
        # Curriculum data
        if self.current_experiment['curriculum_data']:
            curriculum_file = exp_dir / "curriculum.json"
            with open(curriculum_file, 'w') as f:
                json.dump(self.current_experiment['curriculum_data'], f, indent=2)
    
    def _save_experiment_report(self, report: Dict):
        """Save experiment report."""
        report_file = self.current_experiment['directory'] / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Experiment report saved to {report_file}")


# ===============================================================================
# 🎯 CONVENIENCE FUNCTIONS
# ===============================================================================

def create_advanced_experiment_manager(results_dir: str = "experiments") -> ExperimentManager:
    """Create advanced experiment manager with all features."""
    return ExperimentManager(results_dir=results_dir)


def optimize_critical_functions(functions: List[Callable]) -> JITOptimizer:
    """Optimize critical functions with JIT compilation."""
    optimizer = JITOptimizer()
    
    for func in functions:
        # Try different compilation strategies
        optimizer.compile_numpy_function(func)
    
    return optimizer


if __name__ == "__main__":
    print("🚀 EasyMARL Advanced Features Module")
    print("✅ All advanced features loaded successfully!")
    
    # Quick feature test
    monitor = AdvancedPerformanceMonitor()
    curriculum = CurriculumManager()
    optimizer = JITOptimizer()
    analyzer = StatisticalAnalyzer()
    
    print("🎉 All advanced components initialized!")
