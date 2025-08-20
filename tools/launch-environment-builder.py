#!/usr/bin/env python3
"""
Launch script for EasyMARL Environment Builder
Provides easy access to the environment builder GUI with proper configuration.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='EasyMARL Environment Builder Launcher')
    parser.add_argument('--port', type=int, default=7861, help='Port to run the interface on (default: 7861)')
    parser.add_argument('--share', action='store_true', help='Create a shareable public link')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--integrated', action='store_true', help='Launch as part of main GUI (port 7860)')
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Check if we're in the right directory
    if not (script_dir / 'environment_builder.py').exists():
        print("❌ Error: environment_builder.py not found!")
        print(f"Please run this script from the EasyMARL/gui directory")
        print(f"Current directory: {script_dir}")
        return 1
    
    # Set environment variables
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    
    if args.integrated:
        print("🔗 Starting integrated Environment Builder with main GUI...")
        port = 7860
        cmd = [sys.executable, 'gradio_interface.py']
    else:
        print("🏗️ Starting standalone Environment Builder...")
        port = args.port
        cmd = [sys.executable, 'environment_builder.py']
        
        # Add arguments to the command
        if args.share:
            cmd.extend(['--share'])
        cmd.extend(['--port', str(port)])
        cmd.extend(['--host', args.host])
    
    print(f"🌐 Environment Builder will be available at:")
    print(f"   http://localhost:{port}")
    if args.host != '127.0.0.1' and args.host != 'localhost':
        print(f"   http://{args.host}:{port}")
    
    if args.share:
        print("🔗 A shareable public link will be generated")
    
    print("\n📋 Available features:")
    print("   • Visual grid designer with drag-and-drop")
    print("   • 16+ pre-built environment templates")
    print("   • Custom object placement (walls, doors, keys, goals)")
    print("   • Agent configuration and positioning")
    print("   • Export to YAML, Python, or JSON formats")
    print("   • Integration with all EasyMARL algorithms")
    
    print(f"\n🚀 Launching Environment Builder...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Launch the environment builder
        result = subprocess.run(cmd, cwd=script_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Environment Builder stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching Environment Builder: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
