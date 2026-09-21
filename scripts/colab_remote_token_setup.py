"""Install the HF token uploaded as /content/.env.local, then remove the source."""
from pathlib import Path
import os

source = Path("/content/.env.local")
token = None
if source.is_file():
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_ACCESS_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
if not token:
    raise RuntimeError("HF_ACCESS_TOKEN not found in uploaded env")
target = Path("/content/.icgs_hf_token")
target.write_text(token + "\n", encoding="utf-8")
os.chmod(target, 0o600)
source.unlink(missing_ok=True)
print("HF token installed; source env removed")
