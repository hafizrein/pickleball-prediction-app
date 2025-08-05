# Hugging Face Spaces entry point
import sys
import os

# Add src directory to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pickleball.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  
    uvicorn.run(app, host="0.0.0.0", port=port)