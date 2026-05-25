"""
Travel Plan Editing System (TPE)

A grounded-conflict runtime for editing existing travel plans.
"""

__version__ = "0.1.0"

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip environment loading
    pass
