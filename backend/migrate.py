import sys
import re

src = r"c:\Users\nipun\OneDrive\Desktop\SwarmNet final\backend\server.py"
dst = r"c:\Users\nipun\OneDrive\Desktop\SwarmNet final\combined_backend\routers\inference.py"

with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Add APIRouter import
content = "from fastapi import APIRouter\n" + content

# Replace app = FastAPI(...) with router
content = re.sub(r"app\s*=\s*FastAPI\([^)]*\)", "router = APIRouter(tags=['Legacy AI inference'])", content)

# Replace decors
content = content.replace("@app.", "@router.")

# Remove middleware
content = re.sub(r"app\.add_middleware\([^)]*\)", "", content)
content = re.sub(r"app\.add_exception_handler\([^)]*\)", "", content)

# Remove uvicorn block
content = re.sub(r"if __name__ == .__main__.:[\s\S]*", "", content)

with open(dst, "w", encoding="utf-8") as f:
    f.write(content)

print("Migration successful")
