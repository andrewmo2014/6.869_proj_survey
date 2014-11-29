from pointCloud import PointCloud
from quadtree import Quadtree
import numpy as np
from point import Point
import matplotlib.pyplot as plt

def get_circle():
    pc = PointCloud()
    t = np.linspace(0, 2*np.pi, 1000)
    points = []
    for k in t:
        x = np.cos(k)# + .03*np.sin(10*k)
        y = np.sin(k)# + .03*np.sin(10*k)
        pc.add_point(x, y, 0)
    for _ in xrange(10):
        x = np.random.random()
        y = np.random.random()
        pc.add_point(x, y, 0)
    for pt in pc.points:
        pt.normal = -pt.position
    return pc

def get_dist_func(pc):
    def dist(x, y):
        pt = Point(x, y, 0)
        neighbors = pc.nearest_neighbors(pt, 5)
        d = 0
        for pt2 in neighbors:
            diff = pt.position - pt2.position
            d += np.dot(diff, pt2.normal)
        return d
    return dist

if __name__ == '__main__':
    pc = get_circle()
    # pc.compute_normals()
    # pc.display(normals=True)
    dist = get_dist_func(pc)
    quadtree = Quadtree(pc.points, 3)
    # print len(quadtree._get_leaves())
    contour = quadtree.compute_contour(dist)
    # print contour
    print 'Found %d edges' % len(contour)
    for (pt1, pt2) in contour:
        x1, y1, _ = pt1.position
        x2, y2, _ = pt2.position
        plt.plot([x1, x2], [y1, y2])
    plt.show()
    # quadtree.display()
