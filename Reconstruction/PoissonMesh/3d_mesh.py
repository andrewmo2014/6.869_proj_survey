from pointCloud import PointCloud
from octree import Octree
import numpy as np
from point import Point
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def get_plane():
    # s = 2*np.pi
    pc = PointCloud()
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            pc.add_point(x,y,x)
    return pc

def get_circle():
    pc = PointCloud()
    t = np.linspace(0, 2*np.pi, 1000)
    points = []
    for k in t:
        x = np.cos(k) # + .03*np.sin(10*k)
        y = np.sin(k) # + .03*np.sin(10*k)
        pc.add_point(x, y, 0)
    # for _ in xrange(100):
    #     x = 2*(np.random.random()-0.5)
    #     y = 2*(np.random.random()-0.5)
    #     pc.add_point(x, y, 0)
    for pt in pc.points:
        pt.normal = -pt.position / np.sqrt(np.dot(pt.position, pt.position))
    return pc

def get_sphere():
    r = 1
    pc = PointCloud()
    for phi in np.linspace(0, np.pi, 10):
        for theta in np.linspace(0, 2*np.pi, 10):
            z = r * np.cos(phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            # dx = (random.random() - 0.5)/ 10
            # dy = (random.random() - 0.5)/ 10
            # dz = (random.random() - 0.5)/ 10
            # pc.add_point(x+dx,y+dy,z+dz)
            pc.add_point(x, y, z)
    for pt in pc.points:
        pt.normal = -pt.position / np.sqrt(np.dot(pt.position, pt.position))

    return pc

def get_dist_func(pc):
    def dist(x, y, z):
        pt = Point(x, y, z)
        neighbors = pc.nearest_neighbors(pt, 1)
        d = 0
        for pt2 in neighbors:
            diff = pt.position - pt2.position
            d += np.dot(diff, pt2.normal)
        return d
    return dist

if __name__ == '__main__':
    pc = get_plane()
    pc.compute_normals(k=5)
    # pc.display(normals=True)
    dist = get_dist_func(pc)
    octree = Octree(pc.points, 2)
    octree.display()
    contour = octree.compute_contour(dist)
    print 'Found %d faces' % len(contour)
    fig = plt.figure()
    axis = fig.gca(projection='3d')
    #
    # min_x = min_y = min_z = float('inf')
    # max_x = max_y = max_z = float('-inf')
    for (pt1, pt2, pt3) in contour:
        v1 = pt1.position.tolist()
        v2 = pt2.position.tolist()
        v3 = pt3.position.tolist()
        face = Poly3DCollection([[v1, v2, v3]])
        axis.add_collection3d(face)

        # values = zip(v1, v2, v3)
        # min_x = min(min_x, min(values[0]))
        # min_y = min(min_y, min(values[1]))
        # min_z = min(min_z, min(values[2]))
        #
        # max_x = max(max_x, max(values[0]))
        # max_y = max(max_y, max(values[1]))
        # max_z = max(max_z, max(values[2]))

    # print min_x, max_x
    # print min_y, max_y
    # print min_z, max_z
    # axis.set_xlim([2*min_x, 2*max_x])
    # axis.set_ylim([2*min_y, 2*max_y])
    # axis.set_zlim([2*min_z, 2*max_z])

    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1])
    axis.set_zlim([0, 1])


    plt.show()
    # # quadtree.display()
