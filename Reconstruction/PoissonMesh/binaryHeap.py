import random

class BinaryHeap:
    '''An implementation of a Priority Queue as a Min Heap'''

    def __init__(self):
        # (key, value) pairs
        self.heap = [None]
        self.map = {}

    def insert(self, key, value):
        self.heap.append((key, value))
        self.map[value] = len(self.heap)-1
        self._heapify_up(len(self.heap)-1)

    def min(self):
        return self.heap[1][1]

    def extract_min(self):
        value = self.heap[1][1]
        del self.map[value]
        if len(self.heap) > 2:
            self.heap[1] = self.heap.pop()
            self._update_map(1)
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
            if self.heap[index][0] < self.heap[parent][0]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self._update_map(index)
                self._update_map(parent)
                self._heapify_up(parent)

    def _heapify_down(self, index):
        left = 2 * index
        right = 2 * index + 1
        if left == len(self.heap) - 1:
            if self.heap[left][0] < self.heap[index][0]:
                self.heap[index], self.heap[left] = self.heap[left], self.heap[index]
                self._update_map(index)
                self._update_map(left)
        elif left < len(self.heap) - 1:
            if self.heap[left][0] < self.heap[index][0]:
                if self.heap[right][0] < self.heap[left][0]:
                    self.heap[index], self.heap[right] = self.heap[right], self.heap[index]
                    self._update_map(index)
                    self._update_map(right)
                    self._heapify_down(right)
                else:
                    self.heap[index], self.heap[left] = self.heap[left], self.heap[index]
                    self._update_map(index)
                    self._update_map(left)
                    self._heapify_down(left)

    def _validate(self, index=1):
        # Check map
        if len(self.heap) - 1 != len(self.map):
            print 'Map is wrong length'
            return False
        else:
            for i, (key, value) in enumerate(self.heap[1:]):
                # We don't want to enumerate over the first entry
                i = i + 1
                if self.map[value] != i:
                    print 'Map contains wrong values'
                    return False

        # Check min heap property
        left = index * 2
        right = index * 2 + 1
        if left >= len(self.heap):
            return True
        elif left == len(self.heap) - 1:
            return self.heap[index][0] <= self.heap[left][0]
        else:
            print self.heap[index][0], self.heap[left][0], self.heap[right][0]
            if (self.heap[index][0] <= self.heap[left][0]) and (self.heap[index][0] <= self.heap[right][0]):
                return self._validate(left) and self._validate(right)
            else:
                return False

    def is_empty(self):
        return len(self.heap) == 1

    def update_key(self, new_key, value):
        index = self.map[value]
        old_key = self.heap[index][0]
        self.heap[index] = (new_key, value)
        if new_key < old_key:
            self._heapify_up(index)
        elif new_key > old_key:
            self._heapify_down(index)

    def _update_map(self, index):
        value = self.heap[index][1]
        self.map[value] = index

    def get_key(self, value):
        index = self.map[value]
        return self.heap[index][0]



if __name__ == '__main__':
    heap = BinaryHeap()
    heap.insert(1, 'a')
    heap.insert(2, 'b')
    heap.insert(3, 'c')
    heap.insert(4, 'd')
    heap.update_key(5, 'a')
    print heap.heap
    print heap.map
    print heap._validate()
