"""
Streamlit application entry point.
"""

import sys
import os

# Add the project root directory to the Python path
# This allows importing the app package from anywhere
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import and run the main application
from app.main import main

# Run the application
if __name__ == "__main__":
    main()
