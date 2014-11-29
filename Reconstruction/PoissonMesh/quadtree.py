import numpy as np
from point import Point
import matplotlib.pyplot as plt

class Quadtree:
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
        self._draw_node(self.root)
        plt.show()

    def _draw_node(self, node):
        node.draw()
        for child in node.children():
            self._draw_node(child)

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
                contour.append(node_contour)
        return contour

    def _compute_node_contour(self, f, node):
        f_ul = f(node.x1, node.y2)
        f_ll = f(node.x1, node.y1)
        f_ur = f(node.x2, node.y2)
        f_lr = f(node.x2, node.y1)
        signs = [f_ul > 0, f_ll > 0, f_ur > 0, f_lr > 0]

        dx = node.x2 - node.x1
        dy = node.y2 - node.y1
        lower_x = f_ll / (f_ll + f_lr) * dx
        upper_x = f_ul / (f_ul + f_ur) * dx
        left_y = f_ll / (f_ll + f_ul) * dy
        right_y = f_lr / (f_lr + f_ur) * dy

        if signs == [1, 0, 0, 0] or signs == [0, 1, 1, 1]:
            pt1 = Point(node.x1, left_y, 0)
            pt2 = Point(upper_x, node.y2, 0)
            return [(pt1, pt2)]
        elif signs == [0, 1, 0, 0] or signs == [1, 0, 1, 1]:
            pt1 = Point(node.x1, left_y, 0)
            pt2 = Point(lower_x, node.y1, 0)
            return [(pt1, pt2)]
        elif signs == [0, 0, 1, 0] or signs == [1, 1, 0, 1]:
            pt1 = Point(upper_x, node.y2, 0)
            pt2 = Point(node.x2, right_y, 0)
            return [(pt1, pt2)]
        elif signs == [0, 0, 0, 1] or signs == [1, 1, 1, 0]:
            pt1 = Point(lower_x, node.y1, 0)
            pt2 = Point(node.x2, right_y, 0)
            return [(pt1, pt2)]
        elif signs == [1, 0, 1, 0] or signs == [0, 1, 0, 1]:
            pt1 = Point(node.x1, left_y, 0)
            pt2 = Point(node.x2, right_y, 0)
            return [(pt1, pt2)]
        elif signs == [1, 1, 0, 0] or signs == [0, 0, 1, 1]:
            pt1 = Point(lower_x, node.y1, 0)
            pt2 = Point(upper_x, node.y2, 0)
            return [(pt1, pt2)]
        elif signs == [1, 0, 0, 1] or signs == [0, 1, 1, 0]:
            # ambiguous cases
            x_mid = 0.5 * (node.x1 + node.x2)
            y_mid = 0.5 * (node.y1 + node.y2)
            f_mid = f(x_mid, y_mid)
            pt_left = Point(node.x1, left_y)
            pt_right = Point(node.x2, right_y)
            pt_upper = Point(upper_x, node.y2)
            pt_lower = Point(lower_x, node.y1)
            if (f_mid > 0 and signs == [1, 0, 0, 1]) or
               (f_mid < 0 and signs == [0, 1, 1, 0]):
                return [(pt_left, pt_lower), (pt_upper, pt_right)]
            else:
                return [(pt_left, pt_upper), (pt_lower, pt_right)]
        return None










class Node:
    '''Nodes of the tree. Only the last nodes store points'''

    def __init__(self, points, bounds=None):
        self.points = points
        self.parent = None
        self.ul = None
        self.ur = None
        self.ll = None
        self.lr = None
        if bounds:
            self.x1, self.x2, self.y1, self.y2 = bounds
        else:
            self._get_bounds()

    def _get_bounds(self):
        self.x1 = float('inf')
        self.x2 = float('-inf')

        self.y1 = float('inf')
        self.y2 = float('-inf')
        for pt in self.points:
            x, y, z = pt.position
            self.x1 = min(self.x1, x)
            self.x2 = max(self.x2, x)
            self.y1 = min(self.y1, y)
            self.y2 = max(self.y2, y)

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
        x_median = self._get_median(self.points, 0)
        left, right = self._partition(self.points, x_median, 0)
        left_y_median = self._get_median(left, 1)
        right_y_median = self._get_median(right, 1)
        ul_points, ll_points = self._partition(left, left_y_median, 1)
        ur_points, lr_points = self._partition(right, right_y_median, 1)
        if len(ul_points) != 0:
            bounds = (self.x1, x_median, left_y_median, self.y2)
            self.ul = Node(ul_points, bounds)
        if len(ll_points) != 0:
            bounds = (self.x1, x_median, self.y1, left_y_median)
            self.ll = Node(ll_points, bounds)
        if len(ur_points) != 0:
            bounds = (x_median, self.x2, right_y_median, self.y2)
            self.ur = Node(ur_points, bounds)
        if len(lr_points) != 0:
            bounds = (x_median, self.x2, self.y1, left_y_median)
            self.lr = Node(lr_points, bounds)


    def children(self):
        child_list = []
        if self.ul:
            child_list.append(self.ul)
        if self.ur:
            child_list.append(self.ur)
        if self.ll:
            child_list.append(self.ll)
        if self.lr:
            child_list.append(self.lr)
        return child_list

    def draw(self):
        x_box = [self.x1, self.x1, self.x2, self.x2, self.x1]
        y_box = [self.y1, self.y2, self.y2, self.y1, self.y1]
        plt.plot(x_box, y_box)
        x_vec = [pt.position[0] for pt in self.points]
        y_vec = [pt.position[1] for pt in self.points]
        plt.scatter(x_vec, y_vec)

def get_circle():
    t = np.linspace(0, 2*np.pi, 100)
    points = []
    for k in t:
        x = np.cos(k) + np.random.random()/20
        y = np.sin(k) + np.random.random()/20
        pt = Point(x, y, 0)
        points.append(pt)
    return points

if __name__ == '__main__':
    # points = []
    # for _ in xrange(1000):
    #     x, y = np.random.randint(0, 100, 2)
    #     pt = Point(x, y, 0)
    #     points.append(pt)
    points = get_circle()
    quad = Quadtree(points, 4)
    # quad.display()
    print len(quad._get_leaves())
