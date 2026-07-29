from __future__ import annotations

from pathlib import Path


class VideoWriter:
    def __init__(self, path=None, *, fps=20):
        self.path = None if path is None else Path(path)
        self.fps = int(fps)
        self._writer = None

    def open(self):
        if self.path is None or self._writer is not None:
            return self
        import imageio.v2 as imageio
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(self.path), fps=self.fps)
        return self

    def append(self, frame):
        if self.path is not None:
            self.open()._writer.append_data(frame)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
