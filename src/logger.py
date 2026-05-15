import logging
from pathlib import Path

class ProjectLogger:
    """Клас для налаштування логування"""
    
    def __init__(self, logs_dir: str | Path, run_id: str, logger_name: str = __name__):
        self.logs_dir = Path(logs_dir)
        self.run_id = run_id
        self.logger_name = logger_name
        self.logger = logging.getLogger(self.logger_name)
        
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"run_{self.run_id}.log"
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger