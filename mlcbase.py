import logging
from contextlib import AbstractContextManager


class Logger:
    def __init__(self):
        self._logger = logging.getLogger("mlcbase")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

    def init_logger(self):
        return self

    def set_quiet(self):
        self._logger.setLevel(logging.ERROR)

    def set_activate(self):
        self._logger.setLevel(logging.INFO)

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg):
        self._logger.error(msg)

    def warning(self, msg):
        self._logger.warning(msg)


class EmojiProgressBar(AbstractContextManager):
    def __init__(self, total, desc="Progress"):
        self.total = max(int(total), 0)
        self.desc = desc
        self.count = 0
        self.postfix = None

    def __enter__(self):
        if self.total > 0:
            print(f"{self.desc}: 0/{self.total}", flush=True)
        return self

    def update(self, n=1):
        self.count += int(n)
        if self.total > 0:
            line = f"{self.desc}: {min(self.count, self.total)}/{self.total}"
            if self.postfix:
                line = f"{line} {self.postfix}"
            print(line, flush=True)

    def set_postfix(self, values):
        if not values:
            self.postfix = None
            return
        if isinstance(values, dict):
            parts = [f"{key}={value}" for key, value in values.items()]
            self.postfix = " ".join(parts)
            return
        self.postfix = str(values)

    def __exit__(self, exc_type, exc, tb):
        return False
