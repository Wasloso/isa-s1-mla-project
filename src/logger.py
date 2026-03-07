import logging
import sys

_INITIALIZED = False


def get_logger(name: str = "isa") -> logging.Logger:
    global _INITIALIZED  # noqa: PLW0603

    if not _INITIALIZED:
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        if not root.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            root.addHandler(handler)

        _INITIALIZED = True

    return logging.getLogger(name)
