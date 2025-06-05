import logging
import sys

def setup_logging(level_str: str = "INFO"):
    """Configures basic logging for the application."""
    numeric_level = getattr(logging, level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_str}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-5.5s] [%(name)-20.20s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], # Changed to stdout for better compatibility with tqdm
    )
    # Quiet down some noisy libraries if not in DEBUG mode
    if numeric_level > logging.DEBUG:
        for lib_logger_name in ["httpx", "httpcore", "requests", "urllib3", "huggingface_hub", "datasets", "nltk"]:
            logging.getLogger(lib_logger_name).setLevel(logging.WARNING)

    logging.getLogger("antirefusal_sampler").info(f"Logging initialized at level {level_str.upper()}")