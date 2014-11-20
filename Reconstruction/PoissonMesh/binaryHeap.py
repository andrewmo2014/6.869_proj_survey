import random

class BinaryHeap:
    '''An implementation of a Priority Queue as a Min Heap'''

    def __init__(self):
        # (key, value) pairs
        self.heap = [None]

    def insert(self, key, value):
        self.heap.append((key, value))
        self._heapify_up(len(self.heap)-1)

    def min(self):
        return self.heap[1][1]

    def extract_min(self):
        value = self.heap[1][1]
        if len(self.heap) > 2:
            self.heap[1] = self.heap.pop()
            self._heapify_down(1)
        elif len(self.heap) == 2:
            self.heap.pop()
        else:
            print 'Cannot extract min from empty heap'
        return value

    def _heapify_up(self, index):
        # python takes floor automatically
        if index > 1:
            parent = index / 2
            if self.heap[index] < self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self._heapify_up(parent)

    def _heapify_down(self, index):
        if index < len(self.heap) / 2:
            left = index * 2
            right = index * 2 + 1
            if self.heap[left] < self.heap[index]:
                if self.heap[right] < self.heap[left]:
                    self.heap[index], self.heap[right] = self.heap[right], self.heap[index]
                    self._heapify_down(right)
                else:
                    self.heap[index], self.heap[left] = self.heap[left], self.heap[index]
                    self._heapify_down(left)

    def _validate(self, index=1):
        if index >= len(self.heap) / 2:
            return True
        else:
            left = index * 2
            right = index * 2 + 1
            if (self.heap[index] <= self.heap[left]) and (self.heap[index] <= self.heap[right]):
                return self._validate(left) and self._validate(right)
            else:
                return False

    def is_empty(self):
        return len(self.heap) == 1


if __name__ == '__main__':
    heap = BinaryHeap()
    for _ in xrange(1000000):
        k = random.random()
        heap.insert(k,k)
    print heap._validate()
