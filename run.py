import subprocess, sys

try:
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5000"])
except KeyboardInterrupt:
    print("\nImgAuth AI server stopped.")
    sys.exit(0)

