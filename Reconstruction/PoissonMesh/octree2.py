from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pointCloud as pc
from point import Point
from collections import deque
from itertools import product, combinations

class Octree:

	# Octrees allow partitioning 3D-space into 8 regions (octlets)
	# These are arbitrarily labeled as follows

	#                 X-Y-(-Z) Plane      
	#            - -  +-----------+
	#       - -       |     |     |
	# X-Y-(+Z) Plane  |  2  |  3  |
	# +-----------+   |     |     |
	# |     |     |   +-----------+
	# |  6  |  7  |   |     |     |
	# |     |     |   |  0  |  1  |
	# +-----------+   |     |     |
	# |     |     |   +-----------+
	# |  4  |  5  |           - - 
	# |     |     |      - - 
	# +-----------+ - -

	def __init__(self, point_cloud, max_depth, maxPointsPerNode):
		
		self.pc = point_cloud 			# All points in point cloud
		self.maxDepth = max_depth 		# Max tree depth 
		self.maxPPN = maxPointsPerNode	# Max points each node can hold

		(LB, UB) = self.bounding_box()	# Bounds for hull
		self.root = Node(None, 0, None, UB, LB)

	# Compute hull for points (root octree)
	# Make it a cube
	def bounding_box( self ):
		pointArray = map( lambda point: point.position, self.pc.points )
		min_bounds = np.amin( pointArray, axis = 0)
		max_bounds = np.amax( pointArray, axis = 0)
		LB_val = min(min_bounds)
		UB_val = max(max_bounds)

		LB = np.array([LB_val, LB_val, LB_val])
		UB = np.array([UB_val, UB_val, UB_val])

		return (LB, UB)

	# Insert pt in appropriate node in octree, start from root
	def insert(self, pt, currentNode):

		# Leaf Node is empty
		if ((currentNode.points == None) and (currentNode.isLeaf == True)):
			currentNode.points = [pt]
			pt.nodeID = currentNode.ID

		# Leaf Node has space
		elif (((currentNode.points != None) and (len(currentNode.points) < self.maxPPN)) or
			  (currentNode.GetDepth() == self.maxDepth-1 )):
			currentNode.points.append( pt )
			pt.nodeID = currentNode.ID

		# Neither - Need to subdivide Node
		else:

			maxB = currentNode.size/2 + currentNode.center
			minB = -currentNode.size/2 + currentNode.center
			levelCenter = (maxB + minB)/2

			# Create children if is leaf
			if currentNode.isLeaf:

				# Create 8 children
				###############################################################
				children = [None, None, None, None, None, None, None, None]
				for childNum in xrange(8):
					childID = currentNode.ID*8 + (childNum+1)

					# Bounds for child nodes
					UB = maxB
					LB = minB

					if (childNum == 0):
						UB = levelCenter
						LB = LB
					if (childNum == 1):
						UB = np.array([UB[0], levelCenter[1], levelCenter[2]])
						LB = np.array([levelCenter[0], LB[1], LB[2]])
					if (childNum == 2):
						UB = np.array([levelCenter[0], UB[1], levelCenter[2]])
						LB = np.array([LB[0], levelCenter[1], LB[2]])
					if (childNum == 3):
						UB = np.array([UB[0], UB[1], levelCenter[2]])
						LB = np.array([levelCenter[0], levelCenter[1], LB[2]])
					if (childNum == 4):
						UB = np.array([levelCenter[0], levelCenter[1], UB[2]])
						LB = np.array([LB[0], LB[1], levelCenter[2]])
					if (childNum == 5):
						UB = np.array([UB[0], levelCenter[1], UB[2]])
						LB = np.array([levelCenter[0], LB[1], levelCenter[2]])
					if (childNum == 6):
						UB = np.array([levelCenter[0], UB[1], UB[2]])
						LB = np.array([LB[0], levelCenter[1], levelCenter[2]])
					if (childNum == 7):
						UB = np.array([UB[0], UB[1], UB[2]])
						LB = np.array([levelCenter[0], levelCenter[1], levelCenter[2]])

					children[childNum] = Node(currentNode, childID, None, UB, LB)

				currentNode.children = children
				currentNode.isLeaf = False
				###############################################################

				# Move old points from parent into appropriate child nodes
				for old_pt in currentNode.points:
					childNum= self.get_childID(old_pt, levelCenter)
					self.insert( old_pt, currentNode.children[childNum])
				# Set parent node to empty
				currentNode.points = None

			# Place new pt in child node
			childNum = self.get_childID(pt, levelCenter)
			self.insert( pt, currentNode.children[childNum] )


	# Get child id for pt relative to center
	def get_childID( self, pt, center ):
		octnum = 0
		if (pt.position[0] > center[0]):
			octnum += 1
		if (pt.position[1] > center[1]):
			octnum += 2
		if (pt.position[2] > center[2]):
			octnum += 4
		return octnum


	def plot_voxels(self):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_aspect("equal")
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		S = deque([self.root])

		while len(S) != 0:
			v = S.popleft()

			# voxel center
			ax.scatter([v.center[0]],[v.center[1]],[v.center[2]],color="r",s=100/(v.GetDepth()+1))
			
			# Draw voxel as cube
			r1 = [v.lower[0], v.upper[0]]
			r2 = [v.lower[1], v.upper[1]]
			r3 = [v.lower[2], v.upper[2]]
			for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
				if np.sum(np.abs(s-e)) == r1[1]-r1[0]:
					ax.plot3D(*zip(s,e), color="b")
				#if np.sum(np.abs(s-e)) == r2[1]-r2[0]:
				#	ax.plot3D(*zip(s,e), color="g")
				#if np.sum(np.abs(s-e)) == r3[1]-r3[0]:
				#	ax.plot3D(*zip(s,e), color="k")

			for child in v.children:
				if child != None:
					S.append(child)

		#draw octree actual points
		for pt in octree.pc.points:
			ax.scatter([pt.position[0]],[pt.position[1]],[pt.position[2]],color="g",s=100)

		plt.show()

	# String representation for octree
	def __repr__(self):
		octreeDict = {}
		octreeDict[None] = [self.root.ID]

		layer = [self.root]
		next = []

		while layer:
			for node in layer:
				if node != None:
					childList = []
					for child in node.children:
						if child != None:
							childList.append(child.ID)
						next.append(child)
					if node.isLeaf:
						octreeDict[node.ID]=node.points
					else:	
						octreeDict[node.ID]=childList
			layer = next
			next = []

		treeStr= ''
		for key in sorted(octreeDict.keys()):
			treeStr += 'Parent: ' + str(key) + ', Value: ' + str(octreeDict[key]) + '\n'
		return treeStr


class Node:

	def __init__(self, parent, ID, points, UB, LB):
		#Labels
		self.ID = ID 			#ID
		self.points = points 	#List of points
		self.parent = parent 	#Parent Node

		# Dimensions
		self.lower = LB
		self.upper = UB
		self.center = (LB + UB)/2   #Center of node hull
		self.size = UB - LB     	#Size of node hull

		# Pointers to 8 children
		self.isLeaf = True
		self.children = [None, None, None, None, None, None, None, None]

	def __repr__(self):
		return '<Node: id: ' + str(self.ID) + ', points: ' + str(self.points) + '>'

	def GetPath(self):
		current = self
		path = [current]
		while( current.parent != None ):
			path += [current.parent]
			current = current.parent
		path.reverse()
		return path

	def GetDepth(self):
		return len(self.GetPath())-1	

if __name__ == '__main__':
	# Construct point cloud
	myPointCloud = pc.PointCloud()
	myPointCloud.add_point(0,0,0)
	myPointCloud.add_point(12,12,12)
	myPointCloud.add_point(0,0,12)
	myPointCloud.add_point(0, 12, 0)
	myPointCloud.add_point(3, 10, 3)
	myPointCloud.add_point(3, 8, 5)
	myPointCloud.add_point(3, 2.2, 3)
	myPointCloud.add_point(3, 2.3, 2)

	#Make octree
	octree = Octree(myPointCloud, 7, 1)
	for pt in octree.pc.points:
		octree.insert( pt, octree.root )

	print octree

	octree.plot_voxels()

