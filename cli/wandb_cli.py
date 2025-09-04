#!/usr/bin/env python3
"""
Weights & Biases CLI for EasyMARL

This module provides command-line interface for Weights & Biases integration,
making it easy to manage experiments, view results, and sync data.

Usage:
    easymarl-wandb --login
    easymarl-wandb --project my_project --sweep config.yaml
    easymarl-wandb --view --project EasyMARL
"""

import argparse
import sys
import os

def create_wandb_parser():
    """Create argument parser for wandb command."""
    parser = argparse.ArgumentParser(
        description="EasyMARL Weights & Biases CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  easymarl-wandb --login
  easymarl-wandb --project my_project --entity my_team
  easymarl-wandb --view --project EasyMARL
  easymarl-wandb --sweep config.yaml
        """
    )
    
    # Authentication
    parser.add_argument('--login', action='store_true',
                       help='Login to Weights & Biases')
    parser.add_argument('--logout', action='store_true',
                       help='Logout from Weights & Biases')
    
    # Project configuration
    parser.add_argument('--project', type=str, default='EasyMARL',
                       help='W&B project name (default: EasyMARL)')
    parser.add_argument('--entity', type=str, default=None,
                       help='W&B entity/team name')
    
    # Experiment management
    parser.add_argument('--view', action='store_true',
                       help='Open project in browser')
    parser.add_argument('--sweep', type=str, default=None,
                       help='Create sweep from YAML config file')
    parser.add_argument('--agent', action='store_true',
                       help='Start sweep agent')
    
    # Data management
    parser.add_argument('--sync', action='store_true',
                       help='Sync offline runs')
    parser.add_argument('--list', action='store_true',
                       help='List recent runs')
    
    return parser

def wandb_login():
    """Handle W&B login."""
    try:
        import wandb
        wandb.login()
        print("✅ Successfully logged into Weights & Biases")
        return 0
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return 1
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return 1

def wandb_logout():
    """Handle W&B logout."""
    try:
        import wandb
        wandb.logout()
        print("✅ Successfully logged out from Weights & Biases")
        return 0
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return 1
    except Exception as e:
        print(f"❌ Logout failed: {e}")
        return 1

def wandb_view(project, entity):
    """Open project in browser."""
    try:
        import wandb
        import webbrowser
        
        if entity:
            url = f"https://wandb.ai/{entity}/{project}"
        else:
            url = f"https://wandb.ai/project/{project}"
            
        webbrowser.open(url)
        print(f"🌐 Opening {url}")
        return 0
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return 1
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        return 1

def wandb_sweep(config_file, project, entity):
    """Create W&B sweep."""
    try:
        import wandb
        import yaml
        
        with open(config_file, 'r') as f:
            sweep_config = yaml.safe_load(f)
            
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        print(f"✅ Created sweep: {sweep_id}")
        print(f"🚀 Start agent with: wandb agent {sweep_id}")
        return 0
    except ImportError:
        print("❌ wandb or pyyaml not installed. Install with: pip install wandb pyyaml")
        return 1
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_file}")
        return 1
    except Exception as e:
        print(f"❌ Sweep creation failed: {e}")
        return 1

def wandb_list_runs(project, entity):
    """List recent runs."""
    try:
        import wandb
        
        api = wandb.Api()
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            runs = api.runs(project)
            
        print(f"📊 Recent runs for {project}:")
        print("-" * 60)
        
        for i, run in enumerate(runs[:10]):  # Show last 10 runs
            status = "✅" if run.state == "finished" else "🔄" if run.state == "running" else "❌"
            print(f"{status} {run.name} ({run.state}) - {run.created_at}")
            
        return 0
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return 1
    except Exception as e:
        print(f"❌ Failed to list runs: {e}")
        return 1

def main():
    """Main W&B CLI entry point."""
    parser = create_wandb_parser()
    args = parser.parse_args()
    
    if args.login:
        return wandb_login()
    elif args.logout:
        return wandb_logout()
    elif args.view:
        return wandb_view(args.project, args.entity)
    elif args.sweep:
        return wandb_sweep(args.sweep, args.project, args.entity)
    elif args.list:
        return wandb_list_runs(args.project, args.entity)
    elif args.sync:
        try:
            import wandb
            wandb.sync()
            print("✅ Offline runs synced")
            return 0
        except ImportError:
            print("❌ wandb not installed. Install with: pip install wandb")
            return 1
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            return 1
    else:
        parser.print_help()
        return 0

if __name__ == '__main__':
    sys.exit(main())
