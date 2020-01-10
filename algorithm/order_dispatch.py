import os
import sys

import numpy as np
import math

sys.path.append(os.path.join(sys.path[0], "../"))
from algorithm.KM import *

class OrderDispatch:

    def __init__(self, value_map, grids_dist_table, extra_time_for_pickup, max_dispatch_dist=10, gamma=None):
        self.value_map = value_map
        self.grids_dist_table = grids_dist_table
        self.extra_time_for_pickup = extra_time_for_pickup
        self.max_dispatch_dist = max_dispatch_dist

        if gamma is None:
            self.gamma = 1 / math.e
        else:
            self.gamma = gamma
        
        self.km = KM([[1, 100], [0, 1]])  # just initialize KM
        # self.graph = None
        self.match_driver = None
        self.match_order = None
        self.max_match_weight = None

        self.driver_to_grid = None

        self.now = None

    def construct_graph_with_value_map(self, idle_drivers_distribution, orders):
        # order is [start_grid_id, end_grid_id, start_time_unit, duration_time_unit, price, waiting_time]
        graph = []
        idle_drivers_distribution = np.array(idle_drivers_distribution, dtype=int)
        self.driver_to_grid = np.zeros(np.sum(idle_drivers_distribution), dtype=int)
        count = 0
        for g_ind, n_drivers in enumerate(idle_drivers_distribution):
            self.driver_to_grid[count:count+n_drivers] = g_ind
            count += n_drivers
            value_drive_grid = self.value_map[self.now][g_ind]
            weights = []
            for oo in orders:
                assert self.now >= oo[2]
                assert self.now <= oo[2] + oo[5]
                value_dest = self.value_map[self.now][oo[1]]
                total_time = oo[3] + self.extra_time_for_pickup[self.grids_dist_table[g_ind][oo[0]]]
                price = oo[4]

                weight = self.gamma**total_time * value_dest - value_drive_grid + price
                weights.append(weight)
            graph.extend([weights for _ in range(n_drivers)])
        
        # print(self.driver_to_grid)
        # print(idle_drivers_distribution)

        self.km.set_graph(np.array(graph))
    
    def construct_graph_with_only_distance(self, idle_drivers_distribution, orders):
        # order is [start_grid_id, end_grid_id, start_time_unit, duration_time_unit, price, waiting_time]
        graph = []
        idle_drivers_distribution = np.array(idle_drivers_distribution, dtype=int)
        self.driver_to_grid = np.zeros(np.sum(idle_drivers_distribution), dtype=int)
        count = 0
        for g_ind, n_drivers in enumerate(idle_drivers_distribution):
            self.driver_to_grid[count:count+n_drivers] = g_ind
            count += n_drivers
            value_drive_grid = self.value_map[self.now][g_ind]
            weights = []
            for oo in orders:
                assert self.now >= oo[2]
                assert self.now <= oo[2] + oo[5]
                dist = self.grids_dist_table[g_ind][oo[0]]
                if dist == 0:
                    weight = 10
                else:
                    weight = 1 / dist

                weights.append(weight)
            graph.extend([weights for _ in range(n_drivers)])
        
        # print(self.driver_to_grid)
        # print(idle_drivers_distribution)

        self.km.set_graph(np.array(graph))
    
    def dispatch_with_value_map(self, idle_drivers_distribution, orders, now):
        self.now = now
        self.construct_graph_with_value_map(idle_drivers_distribution, orders)
        
        # print(self.graph)
        self.max_match_weight = self.km.match()
        self.match_driver, self.match_order = self.km.get_match_result()

        for d_ind, o_ind in enumerate(self.match_driver):
            if o_ind == NOT_MATCH:
                continue
            elif self.grids_dist_table[orders[o_ind][0]][self.driver_to_grid[d_ind]] > self.max_dispatch_dist:
                self.match_driver[d_ind] = NOT_MATCH
                self.match_order[o_ind] = NOT_MATCH
        
        match_order_to_grid = [NOT_MATCH] * len(self.match_order)
        for i in range(len(self.match_order)):
            if self.match_order[i] != NOT_MATCH:
                match_order_to_grid[i] = self.driver_to_grid[self.match_order[i]]

        return match_order_to_grid, self.max_match_weight
    

    def dispatch_with_only_distance(self, idle_drivers_distribution, orders, now):
        self.now = now
        self.construct_graph_with_only_distance(idle_drivers_distribution, orders)
        
        # print(self.graph)
        self.max_match_weight = self.km.match()
        self.match_driver, self.match_order = self.km.get_match_result()

        for d_ind, o_ind in enumerate(self.match_driver):
            if o_ind == NOT_MATCH:
                continue
            elif self.grids_dist_table[orders[o_ind][0]][self.driver_to_grid[d_ind]] > self.max_dispatch_dist:
                self.match_driver[d_ind] = NOT_MATCH
                self.match_order[o_ind] = NOT_MATCH
        
        match_order_to_grid = [NOT_MATCH] * len(self.match_order)
        for i in range(len(self.match_order)):
            if self.match_order[i] != NOT_MATCH:
                match_order_to_grid[i] = self.driver_to_grid[self.match_order[i]]

        return match_order_to_grid, self.max_match_weight