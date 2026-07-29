from __future__ import annotations

from collections import deque
import numpy as np


class ActionChunkExecutor:
    """Owns replan truncation and queued command execution for all adapters."""

    def __init__(self, execute_steps, action_transform=lambda value: value):
        if int(execute_steps) <= 0:
            raise ValueError("execute_steps must be positive")
        self.execute_steps = int(execute_steps)
        self.action_transform = action_transform
        self._queue = deque()

    def push(self, chunk):
        values = np.asarray(chunk)
        if values.ndim != 2:
            raise ValueError(f"action chunk must be rank 2, got {values.shape}")
        self._queue.clear()
        selected = values[:self.execute_steps]
        transformed = np.asarray(self.action_transform(selected))
        if transformed.ndim != 2 or transformed.shape[0] != selected.shape[0]:
            raise ValueError(
                "action transform must preserve rank and horizon; "
                f"got {transformed.shape} from {selected.shape}")
        self._queue.extend(transformed)
        return len(self._queue)

    def next(self):
        if not self._queue:
            return None
        return self._queue.popleft()

    def run(self, chunk, step):
        self.push(chunk)
        executed = 0
        result = None
        while self._queue:
            result = step(self.next())
            executed += 1
            done = False
            if isinstance(result, tuple) and len(result) >= 5:
                done = bool(result[2]) or bool(result[3])
            elif isinstance(result, tuple) and len(result) > 2:
                done = bool(result[2])
            if done:
                self._queue.clear()
        return result, executed

    def reset(self):
        self._queue.clear()

    @property
    def pending(self):
        return len(self._queue)
