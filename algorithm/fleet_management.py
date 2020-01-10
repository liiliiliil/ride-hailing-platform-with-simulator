import os
import pickle
import numpy as np


class FleetManagement:

    def __init__(self, historical_orders_stat, grids_dist_table, value_map, forecasting_time=2, neighbor_dist=3, ratio=1.0):

        self.historical_orders_stat = np.array(historical_orders_stat)
        self.grids_dist_table = np.array(grids_dist_table)
        self.forecasting_time = forecasting_time
        self.neighbor_dist = neighbor_dist
        self.ratio = ratio
        self.value_map = np.array(value_map)

        self.neighbor_list = None

        self.n_time_unit = self.historical_orders_stat.shape[0]
        self.n_grids = len(self.grids_dist_table)
        self.time = -1

        self.cal_neighbor()
    
    def cal_neighbor(self):
        self.neighbor_list = [None for _ in range(self.n_grids)]
        for g_ind in range(self.n_grids):
            self.neighbor_list[g_ind] = np.array(np.where(np.logical_and(self.grids_dist_table[g_ind]<=self.neighbor_dist,
                                                            self.grids_dist_table[g_ind]>0))).reshape(-1)
            # print(self.neighbor_list[g_ind])
    
    def cal_orders_stat(self, orders):
        stat = np.zeros(self.historical_orders_stat.shape[1], dtype=int)
        # order is [start_grid_id, end_grid_id, start_time_unit, duration_time_unit, price, waiting_time]
        for oo in orders:
            stat[oo[0]] += 1
        
        return stat
    
    def manage_with_distribution(self, time, idle_drivers_number_distribution, remain_orders):
        """
            idle_drivers_number_distribution has shape of [n_grids]
        """
        self.time = time

        management = []
        management_table = np.zeros(self.grids_dist_table.shape, dtype=int)
        idle_drivers_number_distribution = np.array(idle_drivers_number_distribution)

        stat = self.cal_orders_stat(remain_orders)

        if self.time+1 < self.n_time_unit:
            orders_number_pred = np.around(np.mean(self.historical_orders_stat[self.time+1:min(self.time+1+self.forecasting_time, self.n_time_unit)], axis=0), decimals=0)\
                                + stat
        
        for g_ind, n_drivers in enumerate(idle_drivers_number_distribution):
            if n_drivers > 0:
                orders_distribution = orders_number_pred[self.neighbor_list[g_ind]]
                # print(orders_distribution)
                assignment = np.around(orders_distribution * (n_drivers/np.sum(orders_distribution)), decimals=0).astype(int)
                sum_assignment = np.sum(assignment).astype(int)
                delta = n_drivers - sum_assignment

                if delta != 0:
                    assignment[np.argmax(orders_distribution)] += delta
                

                for i, n_ind in enumerate(self.neighbor_list[g_ind]):
                    # print('n_ind is', n_ind)
                    if assignment[i]>0:
                        management_table[g_ind][n_ind] = assignment[i]
                        management.append([g_ind, n_ind, assignment[i]])
                    
                
                # print(management_table)
        
        return management

    def manage_with_value_map(self, time, idle_drivers_number_distribution):
        """
            idle_drivers_number_distribution has shape of [n_grids]
        """
        self.time = time

        management = []
        management_table = np.zeros(self.grids_dist_table.shape, dtype=int)
        idle_drivers_number_distribution = np.array(idle_drivers_number_distribution)

        value_map_mean = np.mean(self.value_map[self.time:min(self.time+self.forecasting_time, self.n_time_unit)], axis=0)
        
        for g_ind, n_drivers in enumerate(idle_drivers_number_distribution):
            if n_drivers > 0:
                # print(type(self.value_map))
                value_distribution = value_map_mean[self.neighbor_list[g_ind]]
                if np.sum(value_distribution) == 0:
                    continue
                # print(orders_distribution)
                assignment = np.around(value_distribution * (n_drivers/np.sum(value_distribution)), decimals=0).astype(int)
                sum_assignment = np.sum(assignment).astype(int)
                delta = n_drivers - sum_assignment

                if delta != 0:
                    assignment[np.argmax(value_distribution)] += delta
                
                assignment = np.around(assignment * self.ratio, decimals=0).astype(int)
                    
                for i, n_ind in enumerate(self.neighbor_list[g_ind]):
                    # print('n_ind is', n_ind)
                    if assignment[i]>0:
                        management_table[g_ind][n_ind] = assignment[i]
                        management.append([g_ind, n_ind, assignment[i]])
                    
                
                # print(management_table)
        
        return management
                
                
        


def test():
    data_path = 'E:/dataset/didi/processed'
    data_file_name = 'processed_data'  # '.pkl' will be added for binary file
    historical_orders_stat_file_name = 'order_20161101_processed_number_stat'  # '.pkl' will be added for binary file

    with open(os.path.join(data_path, data_file_name+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(data_path, historical_orders_stat_file_name+'.pkl'), 'rb') as f:
        historical_orders_stat = pickle.load(f)

    grids_dist_table = np.array(data['dist_table'])

    fleet_management = FleetManagement(historical_orders_stat, grids_dist_table)


if __name__ == '__main__':
    test()

