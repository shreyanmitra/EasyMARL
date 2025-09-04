#!/usr/bin/env python3
"""
Algorithm CLI for EasyMARL

This module provides command-line interface for exploring algorithms,
getting algorithm information, and comparing algorithm performance.

Usage:
    easymarl-algo --list
    easymarl-algo --info qmix
    easymarl-algo --compare qmix vdn ippo
"""

import argparse
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_algo_parser():
    """Create argument parser for algorithm command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Algorithm CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easymarl-algo --list
  easymarl-algo --info qmix
  easymarl-algo --taxonomy
  easymarl-algo --compare qmix vdn ippo
  easymarl-algo --benchmark --env MultiGrid-Empty-6x6-v0
        """
    )
    
    # Algorithm discovery
    parser.add_argument('--list', action='store_true',
                       help='List all available algorithms')
    parser.add_argument('--info', type=str, default=None,
                       help='Show detailed information about an algorithm')
    parser.add_argument('--taxonomy', action='store_true',
                       help='Show algorithm taxonomy and organization')
    
    # Algorithm comparison
    parser.add_argument('--compare', nargs='+', default=None,
                       help='Compare multiple algorithms')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison')
    
    # Configuration
    parser.add_argument('--env', type=str, default='MultiGrid-Empty-6x6-v0',
                       help='Environment for comparison (default: MultiGrid-Empty-6x6-v0)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Episodes for comparison (default: 100)')
    
    return parser

def list_algorithms():
    """List all available algorithms."""
    try:
        from algorithms import list_available_algorithms, ALGORITHM_REGISTRY
        
        print("🧠 Available MARL Algorithms:")
        print("=" * 50)
        
        algorithms = list_available_algorithms()
        
        for category, algs in algorithms.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for alg in algs:
                # Get algorithm info from registry if available
                if alg.upper() in ALGORITHM_REGISTRY:
                    info = ALGORITHM_REGISTRY[alg.upper()]
                    print(f"  • {alg.upper()} - {info.get('description', 'No description')}")
                else:
                    print(f"  • {alg.upper()}")
                    
        print(f"\nTotal: {sum(len(algs) for algs in algorithms.values())} algorithms")
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import algorithm modules: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to list algorithms: {e}")
        return 1

def show_algorithm_info(algo_name):
    """Show detailed information about an algorithm."""
    try:
        from algorithms import ALGORITHM_REGISTRY
        from algorithms.taxonomy import get_algorithm_taxonomy
        
        algo_upper = algo_name.upper()
        
        print(f"🔍 Algorithm Information: {algo_upper}")
        print("=" * 50)
        
        if algo_upper in ALGORITHM_REGISTRY:
            info = ALGORITHM_REGISTRY[algo_upper]
            
            print(f"Name: {info.get('name', algo_upper)}")
            print(f"Description: {info.get('description', 'No description available')}")
            print(f"Paper: {info.get('paper', 'N/A')}")
            print(f"Year: {info.get('year', 'N/A')}")
            print(f"Category: {info.get('category', 'N/A')}")
            print(f"Type: {info.get('type', 'N/A')}")
            
            # Taxonomy information
            taxonomy = get_algorithm_taxonomy(algo_upper)
            if taxonomy:
                print(f"\nTaxonomy:")
                print(f"  Paradigm: {taxonomy.get('paradigm', 'N/A')}")
                print(f"  Learning: {taxonomy.get('learning', 'N/A')}")
                print(f"  Coordination: {taxonomy.get('coordination', 'N/A')}")
                
            # Implementation details
            if 'advantages' in info:
                print(f"\nAdvantages:")
                for adv in info['advantages']:
                    print(f"  • {adv}")
                    
            if 'disadvantages' in info:
                print(f"\nDisadvantages:")
                for dis in info['disadvantages']:
                    print(f"  • {dis}")
                    
            if 'best_for' in info:
                print(f"\nBest For:")
                for use_case in info['best_for']:
                    print(f"  • {use_case}")
                    
        else:
            print(f"❌ Algorithm '{algo_name}' not found in registry")
            print("💡 Use 'easymarl-algo --list' to see available algorithms")
            return 1
            
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import algorithm modules: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to get algorithm info: {e}")
        return 1

def show_taxonomy():
    """Show algorithm taxonomy."""
    try:
        from algorithms.taxonomy import ALGORITHM_TAXONOMY
        
        print("📚 MARL Algorithm Taxonomy:")
        print("=" * 50)
        
        for paradigm, categories in ALGORITHM_TAXONOMY.items():
            print(f"\n{paradigm.replace('_', ' ').title()}:")
            
            for category, algorithms in categories.items():
                print(f"  {category.replace('_', ' ').title()}:")
                for algo in algorithms:
                    print(f"    • {algo}")
                    
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import taxonomy: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to show taxonomy: {e}")
        return 1

def compare_algorithms(algorithms, env_name, episodes):
    """Compare multiple algorithms."""
    try:
        print(f"⚖️ Comparing Algorithms: {', '.join(algorithms)}")
        print(f"Environment: {env_name}")
        print(f"Episodes: {episodes}")
        print("=" * 50)
        
        results = {}
        
        for algo in algorithms:
            print(f"\n🧠 Training {algo.upper()}...")
            
            # Import and run algorithm
            try:
                from main import initialize, main as main_func
                import argparse
                
                # Create args for this algorithm
                args = argparse.Namespace(
                    algorithm=algo.lower(),
                    env_name=env_name,
                    episodes=episodes,
                    evaluate=True,
                    visualize=False,
                    debug=True,
                    vectorized=False,
                    n_envs=1,
                    seed=42,
                    keep_training=False,
                    wandb_project=f'EasyMARL-Compare-{algo}',
                    list_algorithms=False
                )
                
                # Run training and evaluation
                result = main_func(args)
                results[algo] = result
                
            except Exception as e:
                print(f"❌ Failed to run {algo}: {e}")
                results[algo] = None
                
        # Display comparison results
        print("\n📊 Comparison Results:")
        print("-" * 30)
        
        for algo, result in results.items():
            if result:
                print(f"{algo.upper()}: ✅ Success")
            else:
                print(f"{algo.upper()}: ❌ Failed")
                
        return 0
        
    except Exception as e:
        print(f"❌ Algorithm comparison failed: {e}")
        return 1

def run_benchmark(env_name, episodes):
    """Run benchmark comparison of top algorithms."""
    benchmark_algorithms = ['ippo', 'qmix', 'mappo', 'maddpg', 'vdn']
    
    print("🏆 Running EasyMARL Algorithm Benchmark")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")
    print("Algorithms: " + ", ".join([a.upper() for a in benchmark_algorithms]))
    
    return compare_algorithms(benchmark_algorithms, env_name, episodes)

def main():
    """Main algorithm CLI entry point."""
    parser = create_algo_parser()
    args = parser.parse_args()
    
    if args.list:
        return list_algorithms()
    elif args.info:
        return show_algorithm_info(args.info)
    elif args.taxonomy:
        return show_taxonomy()
    elif args.compare:
        return compare_algorithms(args.compare, args.env, args.episodes)
    elif args.benchmark:
        return run_benchmark(args.env, args.episodes)
    else:
        parser.print_help()
        return 0

if __name__ == '__main__':
    sys.exit(main())
