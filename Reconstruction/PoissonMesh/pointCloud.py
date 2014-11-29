import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from binaryHeap import BinaryHeap
import numpy as np
import random
from point import Point

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
            dist = pt2.dist_to(pt)
            index = k
            while index > 0 and dist < neighbors[index-1][1]:
                index -= 1
            if index < k:
                neighbors = neighbors[:index] + [(pt2, dist)] + neighbors[index:-1]
        return [pt2 for pt2, dist in neighbors]

    def _compute_normals(self, k=3):
        print 'Computing normals'
        for pt in self.points:
            neighbors = self.nearest_neighbors(pt, k=k)
            centroid = np.mean([pt2.position for pt2 in neighbors], axis=0)
            covar = np.zeros((3,3))
            for pt2 in neighbors:
                var = pt2.position - centroid
                covar += np.outer(var, var)
            covar /= k
            evalues, evectors = np.linalg.eig(covar)
            pt.normal = evectors[:, np.argmin(evalues)]
            # assert np.inner(pt.normal, pt.normal) == 1
        self._orient_normals(k)

    def _orient_normals(self, k):
        print 'Orienting normals'
        # find pt with maximum z value
        index = np.argmax([pt.position[2] for pt in self.points])
        root = self.points[index]
        if root.normal[2] < 0:
            root.normal *= -1
        parents = {}
        heap = BinaryHeap()
        for pt in self.points:
            if pt == root:
                heap.insert(0, root)
                parents[root] = root
            else:
                parents[pt] = root
                dist = !!!!!
                heap.insert(float('-inf'), pt)
        while not heap.is_empty():
            pt = heap.extract_min()
            # print 'Point:', pt, pt.normal
            # print 'Prev:', prev, prev.normal
            # print
            visited.add(pt)
            prev = parents[pt]
            if np.dot(prev.normal, pt.normal) < 0:
                # print 'flipping'
                pt.normal *= -1

            neighbors = self.nearest_neighbors(pt, len(self.points))
            for pt2 in neighbors:
                if pt2 not in visited:
                    old_dist = heap.get_key(pt2)
                    norm_dist = 1. - np.abs(np.dot(pt.normal, pt2.normal))
                    euclidean_dist = pt.dist_to(pt2)
                    dist = norm_dist * euclidean_dist
                    if dist < old_dist:
                        parents[pt2] = pt
                        heap.update_key(dist, pt2)

        print 'Visited %d vertices' % len(visited)

    def _construct_mesh(self, k):
        for pt in self.points:
            neighbors = self.nearest_neighbors(pt, k)
            for pt2 in neighbors:
                M = np.array([pt.normal, pt2.normal])
                nullspace = null(M)
                if m.shape[0] == 2:
                    line = np.cross(*nullspace)

    def display(self, normals=False):
        '''To do: display normals'''
        print 'Displaying'
        fig = plt.figure()
        axis = fig.gca(projection='3d')
        for pt in self.points:
            if normals:
                args = np.concatenate([pt.position, pt.normal])
                axis.quiver(*args, length=0.1)
            else:
                axis.scatter(*pt.position)
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')
        axis.set_xlim([0,1])
        axis.set_ylim([0,1])
        axis.set_zlim([0,1])
        plt.show()


def make_sphere():
    r = 1
    pc = PointCloud()
    for phi in np.linspace(0, np.pi, 10):
        for theta in np.linspace(0, 2*np.pi, 10):
            z = r * np.cos(phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            pc.add_point(x,y,z)
    return pc

def make_plane():
    s = 1
    pc = PointCloud()
    for x in np.linspace(0, s, 3):
        for y in np.linspace(0, s, 3):
            pc.add_point(x,y,0)
    # for i in range(20):
    #     for j in range(20):
    #         x = random.random()
    #         y = random.random()
    #         # z = 0.01 * random.random()
    #         pc.add_point(x,y,x)
    return pc

def null(M):
    '''Computes the nullspace of M using SVD decomposition. Singular values which are 0 correspond to right singular vectors which are in the nullspace of M. These vectors are orthogonal, so they span the nullspace'''
    u, s, v = np.linalg.svd(M)
    mask = s < 1e-15
    return v[mask]


if __name__ == '__main__':
    pc = make_sphere()
    # for pt in pc.points:
    #     pt.normal = np.array([0,0,1])
    #
    # pc.points[0].normal *= -1
    # pc.points[1].normal *= -1
    # pc.points[4].normal *= -1
    # pc.points[5].normal *= -1
    # pc.points[6].normal *= -1
    # pc.points[7].normal *= -1
    # pc.display(normals=True)
    pc._compute_normals(5)
    # pc._orient_normals(20)
    pc.display(normals=True)
    # random.shuffle(pc.points)
    # pt = Point(.5, .5, .5)
    # neighbors = pc.nearest_neighbors(pt, 5)
    # print neighbors
    # centroid = np.mean([pt.position for pt in neighbors], axis=0)
    # M = np.zeros((3,3))
    # for pt2 in neighbors:
    #     v = pt2.position - centroid
    #     M += np.outer(v, v)
    # print np.linalg.eig(M)
    # pc.display(normals=True)
    # pc._construct_mesh(1)
