import os
from supabase import create_client
from dotenv import load_dotenv

# Load your environment file if not already loaded
load_dotenv()

# Get Supabase credentials from env
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found.")

supabase = create_client(url, key)

# Test query: Adjust table name as needed
result = supabase.from_("site_pages").select("*").execute()
print("Supabase query result:")
print(result)