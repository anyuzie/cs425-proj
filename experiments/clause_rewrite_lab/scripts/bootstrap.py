"""Create an isolated CUDA environment; never alter the original project venv."""
from pathlib import Path
import subprocess
import sys
import venv

root = Path(__file__).resolve().parents[1]
environment = root / ".venv"
python = environment / "Scripts/python.exe"
if not python.exists():
    venv.EnvBuilder(with_pip=True).create(environment)
subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
subprocess.run([str(python), "-m", "pip", "install", "torch==2.7.1", "--index-url",
                "https://download.pytorch.org/whl/cu128"], check=True)
subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True)
subprocess.run([str(python), "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"], check=True)
print("Isolated GPU environment ready")
