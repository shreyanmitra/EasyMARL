"""EasyMARL API Components"""

try:
    from .flask_backend import app
except ImportError:
    app = None  # API dependencies not available
