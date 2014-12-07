import numpy as np
from point import Point
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import triTable


class Octree:
    '''For everything, we assume a cartesian grid, with indices increasing as we move to the top right'''

    def __init__(self, points, depth):
        self.root = Node(points)
        leaves = [self.root]
        new_leaves = []
        for _ in range(depth):
            for node in leaves:
                node.split()
                new_leaves.extend(node.children())
            leaves = new_leaves
            new_leaves = []

    def display(self):
        print 'displaying'
        fig = plt.figure()
        axis = fig.gca(projection='3d')
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')

        self._draw_node(self.root, axis)
        plt.show()

    def _draw_node(self, node, axis):
        node.draw(axis)
        for child in node.children():
            self._draw_node(child, axis)

    def _get_leaves(self):
        nodes = [self.root]
        next_nodes = []
        leaves = []
        while nodes:
            for n in nodes:
                children = n.children()
                if len(children) > 0:
                    next_nodes.extend(children)
                else:
                    leaves.append(n)
            nodes = next_nodes
            next_nodes = []
        return leaves

    def compute_contour(self, f):
        '''Eventually, this should be implemented using a priority queue'''
        leaves = self._get_leaves()
        contour = []
        for node in leaves:
            node_contour = self._compute_node_contour(f, node)
            if node_contour is not None:
                contour.extend(node_contour)
        return contour

    def _compute_node_contour(self, f, node):
        # Cite: http://paulbourke.net/geometry/polygonise/
        f_lll = f(node.x1, node.y1, node.z1)
        f_llu = f(node.x1, node.y1, node.z2)
        f_lul = f(node.x1, node.y2, node.z1)
        f_luu = f(node.x1, node.y2, node.z2)
        f_ull = f(node.x2, node.y1, node.z1)
        f_ulu = f(node.x2, node.y1, node.z2)
        f_uul = f(node.x2, node.y2, node.z1)
        f_uuu = f(node.x2, node.y2, node.z2)

        # The function evaluated at each of the points
        points = {}
        points[0] = f_lul
        points[1] = f_uul
        points[2] = f_ull
        points[3] = f_lll
        points[4] = f_luu
        points[5] = f_uuu
        points[6] = f_ulu
        points[7] = f_llu

        case = self._compute_case(points)

        # Size of the cube (rectangular prism)

        dx = node.x2 - node.x1
        dy = node.y2 - node.y1
        dz = node.z2 - node.z1

        # edges
        # x_ll is the edge along the x axis where y and z are the lower of their two possible values

        x_ll = node.x1 + f_lll / (f_lll - f_ull) * dx
        x_lu = node.x1 + f_lul / (f_llu - f_ulu) * dx
        x_ul = node.x1 + f_ull / (f_lul - f_uul) * dx
        x_uu = node.x1 + f_uul / (f_luu - f_uuu) * dx

        y_ll = node.y1 + f_lll / (f_lll - f_lul) * dy
        y_lu = node.y1 + f_lul / (f_llu - f_luu) * dy
        y_ul = node.y1 + f_ull / (f_ull - f_uul) * dy
        y_uu = node.y1 + f_uul / (f_ulu - f_uuu) * dy

        z_ll = node.z1 + f_lll / (f_lll - f_llu) * dz
        z_lu = node.z1 + f_lul / (f_lul - f_luu) * dz
        z_ul = node.z1 + f_ull / (f_ull - f_ulu) * dz
        z_uu = node.z1 + f_uul / (f_uul - f_uuu) * dz

        edges = {}
        edges[0] = x_ul
        edges[1] = y_ul
        edges[2] = x_ll
        edges[3] = y_ll
        edges[4] = x_uu
        edges[5] = y_uu
        edges[6] = x_lu
        edges[7] = y_lu
        edges[8] = z_lu
        edges[9] = z_uu
        edges[10] = z_ul
        edges[11] = z_ll

        x_ll_pt = Point(x_ll, node.y1, node.z1)
        x_lu_pt = Point(x_lu, node.y1, node.z2)
        x_ul_pt = Point(x_ul, node.y2, node.z1)
        x_uu_pt = Point(x_uu, node.y2, node.z2)

        y_ll_pt = Point(node.x1, y_ll, node.z1)
        y_lu_pt = Point(node.x1, y_lu, node.z2)
        y_ul_pt = Point(node.x2, y_ul, node.z1)
        y_uu_pt = Point(node.x2, y_uu, node.z2)

        z_ll_pt = Point(node.x1, node.y1, z_ll)
        z_lu_pt = Point(node.x1, node.y2, z_lu)
        z_ul_pt = Point(node.x2, node.y1, z_ul)
        z_uu_pt = Point(node.x2, node.y2, z_uu)

        # Given the function evaluated at the two endpoints of an edge, where does the function equal zero along that edge. If one endpoint is positive and the other is negative, the intersection point is between the endpoints. Otherwise, it is outside the segment. We solve for all of them for simplicity.

        zeros = {}
        zeros[0] = x_ul_pt
        zeros[1] = y_ul_pt
        zeros[2] = x_ll_pt
        zeros[3] = y_ll_pt
        zeros[4] = x_uu_pt
        zeros[5] = y_uu_pt
        zeros[6] = x_lu_pt
        zeros[7] = y_lu_pt
        zeros[8] = z_lu_pt
        zeros[9] = z_uu_pt
        zeros[10] = z_ul_pt
        zeros[11] = z_ll_pt

        face_indices = triTable.tri_table[case]
        assert len(face_indices) % 3 == 0
        face_count = len(face_indices) / 3
        face_vec = []
        for index in range(face_count):
            i1 = face_indices[index]
            i2 = face_indices[index+1]
            i3 = face_indices[index+2]
            face = (zeros[i1], zeros[i2], zeros[i3])
            face_vec.append(face)

        return face_vec

    def _compute_case(self, points):
        case = 0
        for index, value in points.items():
            if value < 0:
                case += 2**index
        return case


class Node:
    '''Nodes of the tree. Only the last nodes store points'''

    def __init__(self, points, bounds=None):
        self.points = points
        self.parent = None
        # lll = lower X, lower Y, lower Z
        self.lll = None
        self.llu = None
        self.lul = None
        self.luu = None
        self.ull = None
        self.ulu = None
        self.uul = None
        self.uuu = None

        if bounds:
            self.x1, self.x2, self.y1, self.y2, self.z1, self.z2 = bounds
        else:
            self._get_bounds()

    def _get_bounds(self):
        self.x1 = float('inf')
        self.x2 = float('-inf')

        self.y1 = float('inf')
        self.y2 = float('-inf')

        self.z1 = float('inf')
        self.z2 = float('-inf')

        for pt in self.points:
            x, y, z = pt.position
            self.x1 = min(self.x1, x)
            self.x2 = max(self.x2, x)
            self.y1 = min(self.y1, y)
            self.y2 = max(self.y2, y)
            self.z1 = min(self.z1, z)
            self.z2 = max(self.z2, z)


    def _get_median(self, points, axis):
        vec = [pt.position[axis] for pt in points]
        return np.median(vec)

    def _partition(self, points, value, axis):
        lower = []
        upper = []
        for pt in points:
            if pt.position[axis] <= value:
                lower.append(pt)
            else:
                upper.append(pt)
        return lower, upper

    def split(self):
        '''The left and right sides must have the same y split values. Otherwise, we may get a discontinuity at the x median boundary'''
        x_median = self._get_median(self.points, 0)
        y_median = self._get_median(self.points, 1)
        z_median = self._get_median(self.points, 2)
        l_points, u_points = self._partition(self.points, x_median, 0)

        ll_points, lu_points = self._partition(l_points, y_median, 1)
        ul_points, uu_points = self._partition(u_points, y_median, 1)

        lll_points, llu_points = self._partition(ll_points, z_median, 2)
        lul_points, luu_points = self._partition(lu_points, z_median, 2)
        ull_points, ulu_points = self._partition(ul_points, z_median, 2)
        uul_points, uuu_points = self._partition(uu_points, z_median, 2)


        if len(lll_points) != 0:
            bounds = (self.x1, x_median,
                      self.y1, y_median,
                      self.z1, z_median)
            self.lll = Node(lll_points, bounds)

        if len(llu_points) != 0:
            bounds = (self.x1, x_median,
                      self.y1, y_median,
                      z_median, self.z2)
            self.llu = Node(llu_points, bounds)

        if len(lul_points) != 0:
            bounds = (self.x1, x_median,
                      y_median, self.y2,
                      self.z1, z_median)
            self.lul = Node(lul_points, bounds)

        if len(luu_points) != 0:
            bounds = (self.x1, x_median,
                      y_median, self.y2,
                      z_median, self.z2)
            self.luu = Node(luu_points, bounds)

        if len(ull_points) != 0:
            bounds = (x_median, self.x2,
                      self.y1, y_median,
                      self.z1, z_median)
            self.ull = Node(ull_points, bounds)

        if len(ulu_points) != 0:
            bounds = (x_median, self.x2,
                      self.y1, y_median,
                      z_median, self.z2)
            self.ulu = Node(ulu_points, bounds)

        if len(uul_points) != 0:
            bounds = (x_median, self.x2,
                      y_median, self.y2,
                      self.z1, z_median)
            self.uul = Node(uul_points, bounds)

        if len(uuu_points) != 0:
            bounds = (x_median, self.x2,
                      y_median, self.y2,
                      z_median, self.z2)
            self.uuu = Node(uuu_points, bounds)


    def children(self):
        child_list = []
        if self.lll:
            child_list.append(self.lll)
        if self.llu:
            child_list.append(self.llu)
        if self.lul:
            child_list.append(self.lul)
        if self.luu:
            child_list.append(self.luu)
        if self.ull:
            child_list.append(self.ull)
        if self.ulu:
            child_list.append(self.ulu)
        if self.uul:
            child_list.append(self.uul)
        if self.uuu:
            child_list.append(self.uuu)
        return child_list

    def draw(self, axis):
        # Avoid overlapping lines in visualization
        dx = (self.x2 - self.x1) / 100
        dy = (self.y2 - self.y1) / 100
        dz = (self.z2 - self.z1) / 100

        front_face_x = [self.x1-dx, self.x2+dx, self.x2+dx, self.x1-dx, self.x1-dx]
        front_face_y = [self.y1-dy, self.y1-dy, self.y2+dy, self.y2+dy, self.y1-dy]
        front_face_z = [self.z1-dz, self.z1-dz, self.z1-dz, self.z1-dz, self.z1-dz]


        back_face_x = [self.x1-dx, self.x2+dx, self.x2+dx, self.x1-dx, self.x1-dx]
        back_face_y = [self.y1-dy, self.y1-dy, self.y2+dy, self.y2+dy, self.y1-dy]
        back_face_z = [self.z2+dz, self.z2+dz, self.z2+dz, self.z2+dz, self.z2+dz]

        top_face_x = [self.x1-dx, self.x2+dx, self.x2+dx, self.x1-dx, self.x1-dx]
        top_face_y = [self.y2+dy, self.y2+dy, self.y2+dy, self.y2+dy, self.y2+dy]
        top_face_z = [self.z1-dz, self.z1-dz, self.z2+dz, self.z2+dz, self.z1-dz]

        bottom_face_x = [self.x1-dx, self.x2+dx, self.x2+dx, self.x1-dx, self.x1-dx]
        bottom_face_y = [self.y1-dy, self.y1-dy, self.y1-dy, self.y1-dy, self.y1-dy]
        bottom_face_z = [self.z1-dz, self.z1-dz, self.z2+dz, self.z2+dz, self.z1-dz]

        axis.plot(front_face_x, front_face_y, front_face_z)
        axis.plot(back_face_x, back_face_y, back_face_z)
        axis.plot(top_face_x, top_face_y, top_face_z)
        axis.plot(bottom_face_x, bottom_face_y, bottom_face_z)
        x_vec = [pt.position[0] for pt in self.points]
        y_vec = [pt.position[1] for pt in self.points]
        z_vec = [pt.position[2] for pt in self.points]
        axis.scatter(x_vec, y_vec, z_vec)

def make_sphere():
    points = []
    r = 1
    for phi in np.linspace(0, np.pi, 10):
        for theta in np.linspace(0, 2*np.pi, 10):
            z = r * np.cos(phi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            pt = Point(x, y, z)
            points.append(pt)
    return points

if __name__ == '__main__':
    points = [] #make_sphere()
    for x in np.linspace(0, 2*np.pi, 30):
        for y in np.linspace(0, 2*np.pi, 30):
            pt = Point(x, y, np.sin(x)+np.sin(y))
            points.append(pt)
    octree = Octree(points, 3)

    # print len(octree._get_leaves())
    octree.display()
