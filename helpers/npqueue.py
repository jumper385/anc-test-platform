import numpy as np

class NPFIFO:
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = np.zeros(max_length, dtype=np.float32)  # or any other data type
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item):
        if self.size == self.max_length:
            raise OverflowError("Queue is full")
        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.max_length
        self.size += 1

    def dequeue(self):
        if self.size == 0:
            raise IndexError("Queue is empty")
        item = self.queue[self.head]
        self.head = (self.head + 1) % self.max_length
        self.size -= 1
        return item

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.max_length

    def __len__(self):
        return self.size

    def peek(self):
        if self.size == 0:
            raise IndexError("Queue is empty")
        return self.queue[self.head]
