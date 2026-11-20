import subprocess
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv-feast" / "Scripts" / "python.exe"
MONITOR = PROJECT_ROOT / "src" / "live_drift_monitor.py"

LOG_DIR = PROJECT_ROOT / "reports" / "live_drift_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"drift_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Live Drift Monitor started")

try:
    result = subprocess.run(
        [str(PYTHON), str(MONITOR)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.stdout:
        logging.info("STDOUT:\n%s", result.stdout)

    if result.stderr:
        logging.error("STDERR:\n%s", result.stderr)

    logging.info("Python exit code: %s", result.returncode)

except Exception:
    logging.exception("Live Drift Monitor failed")

print(f"Log saved: {log_file}")