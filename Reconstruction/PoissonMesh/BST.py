from node import Node

class BST:

    def __init__(self):
        self.root = None

    def insert(self, key, value):
        node = Node(key, value)
        if self.root == None:
            self.root == node
        else:
            parent = self.root
            next = parent
            while next:
                parent = next
                if key > next.key:
                    parent = next
                    next = next.right
                elif key < next.key:
                    parent = next
                    next = next.left
                else:
                    print 'Key already in BST!'
            assert parent.key != key
            if key > parent.key:
                assert parent.right == None
                parent.right = node
                node.parent = parent
            else:
                assert parent.left == None
                parent.left = node
                node.parent = parent

    def delete(self, key):
        if key == self.root:
            node = _replacementChild(self.root)
            self.root.right.parent = node
            self.root.left.parent = node
            self.root = node
        else:
            parent = self.root
            while parent and parent.key != key:
                if key > parent.key:
                    parent = parent.right
                elif key < parent.key:
                    parent = parent.left
            if parent:
                    node = _replacementChild(parent)
                    parent.right.parent = node
                    parent.left.parent = node

    def _replacementChild(self, node):
        '''Finds a replacement node. With equal probability, it will choose the minChild of node.right and the maxChild of node.left. The node is removed from the tree.'''
        if random.random() > 0.5:
            return _minChild(node.right)
        else:
            return _maxChild(node.left)


    def _minChild(self, node):
        while node.left:
            node = node.left
        node.parent.left = None
        return node

    def _maxChild(self, node):
        while node.right:
            node = node.right
        node.parent.right = None
        return node

    def max(self):
        node = self.root
        while node.right:
            node = node.right
        return node.key

    def to_list(self):
        keys = []
        layer = [self.root]
        next = []
        while layer:
            for node in layer:
                keys.append(node.value)
                next.append(node.left)
                next.append(node.right)
            layer = next
            next = []
        return keys
