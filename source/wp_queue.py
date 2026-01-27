from queue import Queue, Empty

from source.models import Position


class WayPointsQueue:
    def __init__(self, initial: Position):
        self._queue: Queue[Position] = Queue()
        self._current: Position = initial

    def push(self, wp: Position):
        self._queue.put(wp)

    def get_current(self) -> Position:
        return self._current

    def update_current(self) -> bool:
        try:
            self._current = self._queue.get_nowait()
            return True
        except Empty:
            return False
