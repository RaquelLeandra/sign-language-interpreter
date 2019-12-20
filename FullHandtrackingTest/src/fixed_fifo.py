from collections import deque


class FixedFIFO(deque):

    def __init__(self, max_size):
        self.max_size = max_size
        super().__init__()

    def put(self, item):
        super().append(item)
        if len(self) > self.max_size:
            return self.popleft()
        return None

    def get(self):
        return self.popleft()