import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    current_dir = Path(__file__).parent
    env_file = current_dir / '.env'
    
    print(f"\nEnvironment Setup Debug:")
    print(f"Current directory: {current_dir}")
    print(f"Looking for .env at: {env_file}")
    print(f"File exists: {env_file.exists()}")
    
    if not env_file.exists():
        raise ValueError(f".env file not found at {env_file}")
        
    # Load environment variables with override
    load_dotenv(env_file, override=True)
    
    # Validate OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    elif "YOUR_OPENAI_API" in api_key:  # More flexible check
        raise ValueError("OPENAI_API_KEY contains placeholder value")
    elif not api_key.startswith(('sk-', 'sk-proj-')):
        raise ValueError(f"Invalid OPENAI_API_KEY format. Found: {api_key[:10]}...")
    
    return api_key 