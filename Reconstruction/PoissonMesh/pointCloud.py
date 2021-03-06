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

    def nearest_neighbors(self, pt, k=1, points=None):
        '''Returns the k nearest neighbors. Eventually, this should be implemented using a kd tree or an R tree. For now, we use exhaustive search'''
        if points is None:
            points = self.points
        neighbors = [(None, float('inf')) for _ in range(k)]
        for pt2 in points:
            dist = pt2.dist_to(pt)
            index = k
            while index > 0 and dist < neighbors[index-1][1]:
                index -= 1
            if index < k:
                neighbors = neighbors[:index] + [(pt2, dist)] + neighbors[index:-1]
        return [pt2 for pt2, dist in neighbors]

    def compute_normals(self, k=3):
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
        return self._orient_normals(k)

    def _orient_normals(self, k):
        print 'Orienting normals'
        # find pt with maximum z value
        index = np.argmax([pt.position[2] for pt in self.points])
        root = self.points[index]
        if root.normal[2] > 0:
            root.normal *= -1
        parents = {}
        heap = BinaryHeap()
        for pt in self.points:
            if pt == root:
                heap.insert(0, pt)
                parents[root] = root
            else:
                heap.insert(float('inf'), pt)
        while not heap.is_empty():
            pt = heap.extract_min()
            if pt in parents:
                prev = parents[pt]
            else:
                prev = self.nearest_neighbors(pt, 1, parents.keys())[0]
                parents[pt] = prev
            if np.dot(prev.normal, pt.normal) < 0:
                pt.normal *= -1

            neighbors = self.nearest_neighbors(pt, k)
            for pt2 in neighbors:
                if pt2 not in parents:
                    old_dist = heap.get_key(pt2)
                    dist = 1. - np.abs(np.dot(pt.normal, pt2.normal))
                    if dist < old_dist:
                        parents[pt2] = pt
                        heap.update_key(dist, pt2)
        return parents


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
        min_x = np.min([pt.position[0] for pt in self.points])
        min_y = np.min([pt.position[1] for pt in self.points])
        min_z = np.min([pt.position[2] for pt in self.points])
        max_x = np.max([pt.position[0] for pt in self.points])
        max_y = np.max([pt.position[1] for pt in self.points])
        max_z = np.max([pt.position[2] for pt in self.points])
        axis.set_xlim([min_x, max_x])
        axis.set_ylim([min_y, max_y])
        axis.set_zlim([min_z, max_z])
        plt.show()


def make_sphere():
    r = 1
    pc = PointCloud()
    for phi in np.linspace(0, np.pi, 20):
        for theta in np.linspace(0, 2*np.pi, 20):
            z = r * np.cos(phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            dx = (random.random() - 0.5)/ 10
            dy = (random.random() - 0.5)/ 10
            dz = (random.random() - 0.5)/ 10
            pc.add_point(x+dx,y+dy,z+dz)
    return pc

def make_plane():
    s = 2*np.pi
    pc = PointCloud()
    for x in np.linspace(0, s, 20):
        for y in np.linspace(0, s, 20):
            pc.add_point(x,y,np.sin(x))
    return pc

def null(M):
    '''Computes the nullspace of M using SVD decomposition. Singular values which are 0 correspond to right singular vectors which are in the nullspace of M. These vectors are orthogonal, so they span the nullspace'''
    u, s, v = np.linalg.svd(M)
    mask = s < 1e-15
    return v[mask]

def display_mst(pc):
    parents = pc.compute_normals(10)
    fig = plt.figure()
    axis = fig.gca(projection='3d')
    cmap = plt.get_cmap('jet')
    for (pt, pt2) in parents.items():
        d11 = np.dot(pt.normal, pt.normal)
        d22 = np.dot(pt2.normal, pt2.normal)
        d12 = np.dot(pt.normal, pt2.normal)
        weight = np.abs(d12 / (d11 * d22))**4
        color = cmap(weight)
        axis.plot(*zip(pt.position, pt2.position), color=color)
    plt.show()


if __name__ == '__main__':
    pc = make_sphere()
    display_mst(pc)
    # random.shuffle(pc.points)
    # parents = pc.compute_normals(10)
    # pc.display(normals=True)
