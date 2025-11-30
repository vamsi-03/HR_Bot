import logging
from pathlib import Path

from config.settings import settings


def setup_logging() -> None:
    log_path: Path = settings.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
