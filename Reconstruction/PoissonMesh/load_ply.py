from point import Point
import numpy as np
from pointCloud import PointCloud
from octree import Octree

def load_ply(filename):
    points = []
    with open(filename) as f:
        while 'end_header' not in f.readline():
            pass
        for line in f.readlines()[::100]:
            data = line.split()
            data = map(float, data)
            x, y, z = data[:3]
            nx, ny, nz = data[3:6]
            pt = Point(x,y,z)
            pt.normal = np.array([x, y, z])
            points.append(pt)
    return points

if __name__ == '__main__':
    points = load_ply('../../Rift/temple.ply')
    octree = Octree(points, 3)
    octree.display()
