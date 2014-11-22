import matplotlib.pyplot as plt
import numpy as np
from point import Point
from node import Node

class Octree:

    def __init__(self):
        
        self.root = None

    def insert(self, pt):
