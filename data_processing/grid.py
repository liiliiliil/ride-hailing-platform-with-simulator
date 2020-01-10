import numpy as np
import math

class Hexagon:
    
    def __init__(self, size=-1, size_to_edge=-1, center=[0, 0]):
        """
        size: length from center to one point
        if size and size_to_edge both are not assigned, size will be 1.
        """
        if size_to_edge != -1:
            self.size_to_edge = size_to_edge
            self.half_size = self.size_to_edge / math.sqrt(3)
            self.size = self.half_size * 2
        else:
            if size == -1:
                self.size = 1
            else:
                self.size = size
            self.half_size = self.size / 2
            self.size_to_edge = self.half_size * math.sqrt(3)


        self.center = np.array(center).astype(np.float)

        self.points = np.zeros((6, 2)).astype(np.float)
        self.init_points()

    def init_points(self):
        self.points[0, :] = np.array([self.center[0]+self.size, self.center[1]]).astype(np.float)
        self.points[1, :] = np.array([self.center[0]+self.half_size, self.center[1]+self.size_to_edge]).astype(np.float)
        self.points[2, :] = np.array([self.center[0]-self.half_size, self.center[1]+self.size_to_edge]).astype(np.float)
        self.points[3, :] = np.array([self.center[0]-self.size, self.center[1]]).astype(np.float)
        self.points[4, :] = np.array([self.center[0]-self.half_size, self.center[1]-self.size_to_edge]).astype(np.float)
        self.points[5, :] = np.array([self.center[0]+self.half_size, self.center[1]-self.size_to_edge]).astype(np.float)

    def set_center(self, new_center):
        """self.center is np.narray"""
        move = new_center - self.center
        self.center = new_center
        self.points += move
    
    def set_size(self, new_size):
        self.size = new_size
        self.half_size = self.size / 2
        self.size_to_edge = self.half_size * math.sqrt(3)
        self.init_points()

    def get_points(self):
        return self.points
    
    def get_x_y_for_plot(self):
        points_list = self.points.T.tolist()
        return points_list[0]+[points_list[0][0]], points_list[1]+[points_list[1][0]]
