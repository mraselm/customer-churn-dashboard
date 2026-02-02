#!/usr/bin/env python3
import subprocess
import sys

print("Installing dice-ml...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "dice-ml"], 
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode == 0:
    print("✅ dice-ml installed successfully!")
else:
    print("❌ Installation failed:")
    print(result.stderr)
