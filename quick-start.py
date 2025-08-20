#!/usr/bin/env python3
"""
EasyMARL Quick Start Script

This script provides the fastest way to get started with EasyMARL.
Simply run: python quick-start.py

It will automatically:
1. Check if EasyMARL is installed
2. Launch the appropriate interface based on availability
3. Provide helpful guidance for first-time users
"""

import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"📦 Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def main():
    """Main quick start function."""
    print("🎓 EasyMARL Quick Start")
    print("=" * 40)
    
    # Check if EasyMARL is installed
    if not check_package_installed("easymarl"):
        print("❌ EasyMARL not found. Installing...")
        try:
            install_package("easymarl")
            print("✅ EasyMARL installed successfully!")
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print("Please try: pip install easymarl")
            return
    
    # Try to launch Gradio interface
    try:
        print("🚀 Launching EasyMARL Gradio Interface...")
        import easymarl
        easymarl.launch_gui()
    except ImportError:
        print("❌ Gradio interface not available.")
        print("🔧 Installing GUI dependencies...")
        try:
            install_package("gradio")
            print("✅ GUI dependencies installed!")
            print("🚀 Launching interface...")
            import easymarl
            easymarl.launch_gui()
        except Exception as e:
            print(f"❌ Could not launch GUI: {e}")
            print("\n📖 Alternative options:")
            print("1. Use Python API: import easymarl; env = easymarl.make_env('MultiGrid-Empty-6x6-v0')")
            print("2. Start web interface: ./tools/start-easymarl.sh")
            print("3. Use GitHub Codespaces: https://codespaces.new/shreyanmitra/EasyMARL")

if __name__ == "__main__":
    main()
