"""EasyMARL GUI Components"""

def launch_gui():
    """Launch the EasyMARL GUI"""
    from .gradio_interface import main
    main()

__all__ = ['launch_gui']
