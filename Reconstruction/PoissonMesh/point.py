import numpy as np

class Point:

    def __init__(self, x, y, z):
        self.position = np.array([x,y,z])
        self.normal = None
        self.nodeID = None

    def dist_to(self, pt):
        delta = self.position - pt.position
        return np.sqrt(np.inner(delta, delta))

    def __repr__(self):
        x, y, z = self.position
        return '<Point: %.3f, %.3f, %.3f>' % (x, y, z)
