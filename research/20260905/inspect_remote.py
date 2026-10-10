"""Read-only inventory of the user-authorized experiment server."""

import json
import os
import socket
import subprocess
from pathlib import Path


print(json.dumps({"host": socket.gethostname(), "user": os.environ.get("USER")}, ensure_ascii=True))
for argv in (
    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"],
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    print(result.stdout or result.stderr)

for directory in (
    "/data/sunzixuan",
    "/data/sunzixuan/expirement/LOBSTER/data",
    "/data/sunzixuan/bridge_official",
    "/data/sunzixuan/bridge_official/src/saved_models/BRIDGE",
):
    path = Path(directory)
    print("DIRECTORY", directory)
    if not path.is_dir():
        continue
    for child in sorted(path.iterdir()):
        print(child.name, "dir" if child.is_dir() else child.stat().st_size)

data = Path("/data/sunzixuan/expirement/LOBSTER/data")
for domain in ("baby", "sports", "elec"):
    path = data / domain
    print("DATASET", domain)
    if path.is_dir():
        for child in sorted(path.iterdir()):
            print(child.name, "dir" if child.is_dir() else child.stat().st_size)
