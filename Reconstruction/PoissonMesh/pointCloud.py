import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from binaryHeap import BinaryHeap
import numpy as np
import random

class PointCloud:

    def __init__(self, points=[]):
        self.points = points

    def add_point(self, x, y, z):
        pt = Point(x, y, z)
        self.points.append(pt)

    def nearest_neighbors(self, pt, k=1):
        '''Returns the k nearest neighbors. Eventually, this should be implemented using a kd tree or an R tree. For now, we use exhaustive search'''
        neighbors = [(None, float('inf')) for _ in range(k)]
        for pt2 in self.points:
            dist = pt2.distTo(pt)
            index = k
            while index > 0 and dist < neighbors[index-1][1]:
                index -= 1
            if index < k:
                neighbors = neighbors[:index] + [(pt2, dist)] + neighbors[index:-1]
        return [pt2 for pt2, dist in neighbors]

    def _compute_normals(self, k=3):
        for pt in self.points:
            neighbors = self.nearest_neighbors(pt, k=k)
            centroid = np.mean([pt2.position for pt2 in neighbors], axis=0)
            covar = np.zeros((3,3))
            for pt2 in neighbors:
                var = pt2.position - centroid
                covar += np.outer(var, var)
            covar /= k
            evalues, evectors = np.linalg.eig(covar)
            print np.min(evalues)
            pt.normal = evectors[np.argmin(evalues)]
            # assert np.inner(pt.normal, pt.normal) == 1
        # self._orient_normals(k)

    def _orient_normals(self, k):
        # find pt with maximum z value
        index = np.argmax([pt.position[2] for pt in self.points])
        root = self.points[index]
        if root.normal[0] < 0:
            root.normal *= -1
        prev = root
        visited = {root}
        heap = BinaryHeap()
        heap.insert(0, root)
        while not heap.is_empty():
            pt = heap.extract_min()
            if pt not in visited:
                visited.add(pt)
            if np.dot(prev.normal, pt.normal) < 0:
                pt.normal *= -1
            prev = pt

            neighbors = self.nearest_neighbors(pt, k)
            for pt2 in neighbors:
                if pt2 not in visited:
                    dist = 1. - np.abs(np.dot(pt.normal, pt2.normal))
                    heap.insert(dist, pt2)

    def display(self, normals=False):
        '''To do: display normals'''
        fig = plt.figure()
        axis = fig.gca(projection='3d')
        for pt in self.points:
            if normals:
                args = np.concatenate([pt.position, pt.normal])
                axis.quiver(*args, length=0.03)
            else:
                axis.scatter(*pt.position)
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        plt.show()


class Point:

    def __init__(self, x, y, z):
        self.position = np.array([x,y,z])
        self.normal = np.array([None, None, None])

    def distTo(self, pt):
        return np.sqrt(np.inner(self.position, pt.position))

    def __repr__(self):
        x, y, z = self.position
        return '<Point: %.3f, %.3f, %.3f>' % (x, y, z)

def make_sphere():
    r = 1
    pc = PointCloud()
    for phi in np.linspace(0, np.pi, 20):
        for theta in np.linspace(0, 2*np.pi, 20):
            z = r * np.cos(phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            pc.add_point(x,y,z)
    return pc

def make_plane():
    s = 1
    pc = PointCloud()
    for x in np.linspace(0, s, 10):
        for y in np.linspace(0, s, 10):
            pc.add_point(x,y,0)
    return pc

if __name__ == '__main__':
    # pc = PointCloud()
    # for _ in xrange(100):
    #     x = 10*random.random()
    #     y = 10*random.random()
    #     z = 10*random.random()
    #     pc.add_point(x,y,z)
    # print len(pc.points)
    # pc._compute_normals()
    # pc = make_sphere()
    pc = make_plane()
    # random.shuffle(pc.points)
    pc._compute_normals(10)
    pc.display(normals=True)
