import sys
import os

sys.path.append(os.path.join(sys.path[0], "../"))
from algorithm.order_dispatch import *
from algorithm.fleet_management import *


class OrderDispatchAndFleetManagement:

    def __init__(self, value_map, grids_dist_table, extra_time_for_pickup, max_dispatch_dist, gamma,
                historical_orders_stat, forecasting_time=2, neighbor_dist=3, ratio=1.0):
        
        self.od = OrderDispatch(value_map, grids_dist_table, extra_time_for_pickup, max_dispatch_dist, gamma)
        self.fm = FleetManagement(historical_orders_stat, grids_dist_table, value_map,forecasting_time, neighbor_dist, ratio)

    def step(self, time, idle_drivers_distribution, orders):

        match_order_to_grid, weight = self.od.dispatch_with_value_map(idle_drivers_distribution, orders, time)

        match_order = self.od.match_order
        match_driver = self.od.match_driver
        driver_to_grid = self.od.driver_to_grid

        for d_ind in range(len(match_driver)):
            if match_driver[d_ind] != NOT_MATCH:
                idle_drivers_distribution[driver_to_grid[d_ind]] -= 1
        
        remain_orders = [oo for o_ind, oo in enumerate(orders) if match_order[o_ind]==NOT_MATCH]

        management = self.fm.manage_with_value_map(time, idle_drivers_distribution)

        return match_order_to_grid, management

