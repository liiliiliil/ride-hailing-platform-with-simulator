
from collections import defaultdict
import sys
import traceback
import os, sys
import pickle

sys.path.append(os.path.join(sys.path[0], "../"))
from simulator.envs import *

data_path = 'E:/dataset/didi/processed'
data_name = 'processed_data.pkl'
n_time_units = 144
max_n_drivers = 160
l_max = 9

with open(os.path.join(data_path, data_name), 'rb') as f:
    data = pickle.load(f)

mapped_matrix_int = np.array(data['map'])  # in order to make every index greater than 0
order_real = np.array(data['orders'])
drivers_number_time_distribution = np.array(data['drivers_distribution']['time'])
drivers_number_grid_distribution = np.array(data['drivers_distribution']['grid']).reshape(-1)

M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = M * N
idle_driver_location_mat = np.zeros((144, num_valid_grid))

assert (mapped_matrix_int.reshape(-1) == (np.arange(num_valid_grid))).all()

for ii in np.arange(144):
    # each element is number of idle drivers at particular grid
    # print(max_n_drivers * drivers_number_time_distribution[ii])
    idle_driver_location_mat[ii, :] = np.round((max_n_drivers * drivers_number_time_distribution[ii]) * drivers_number_grid_distribution)
    # print(np.sum(idle_driver_location_mat[ii, :]))
    # print(idle_driver_location_mat[ii, :])
idle_driver_dist_time = np.stack([np.round(max_n_drivers * drivers_number_time_distribution), np.zeros(drivers_number_time_distribution.shape)], axis=1)

n_side = 6


############# NOT USED if you use real orders #############
order_time = [0.2, 0.2, 0.15,
                0.15,  0.1,  0.1,
                0.05, 0.04,  0.01]
order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                [15.02, 6.90],  # mean and std of order price when duration is 20 min
                [23.22, 11.63],
                [32.14, 16.20],
                [40.99, 20.69],
                [49.94, 25.61],
                [58.98, 31.69],
                [68.80, 37.25],
                [79.40, 44.39]]


onoff_driver_location_mat = np.zeros((n_time_units, num_valid_grid, 2))

print('shape of mapped_matrix_int is', mapped_matrix_int.shape)
print('shape of drivers_number_time_distribution is', drivers_number_time_distribution.shape)
print('shape of drivers_number_grid_distribution is', drivers_number_grid_distribution.shape)
print('shape of idle_driver_dist_time is', idle_driver_dist_time.shape)
print('shape of onoff_driver_location_mat is', onoff_driver_location_mat.shape)
print('len(order_real) is', len(order_real))

env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                order_time, order_price, l_max, M, N, n_side, 1, order_real, onoff_driver_location_mat)

state = env.reset_clean()
order_response_rates = []
T = 0
max_iter = n_time_units
while T < max_iter:
    # if T % 5 == 0:
    #     state = env.reset_clean(generate_order=2)
    dispatch_action = []
    state, reward, _ = env.step(dispatch_action, generate_order=2)  # generate_order: 1 for random, 2 for real data

    print("City time {}: Order response rate: {}".format(env.city_time-1, env.order_response_rate))
    order_response_rates.append(env.order_response_rate)

    print("idle driver: {} == {} total num of drivers: {} num of offline drivers {}".format(np.sum(state[0]),
                                                                    np.sum(env.get_observation_driver_state()),
                                                                    len(env.drivers.keys()),
                                                                    len(env.utility_collect_offline_drivers_id())))
    
    print("total number of orders: {} == {}".format(np.sum(state[1]),  # it is the state of next time unit, so the orders have not been dispatched
                                                                    len(env.day_orders[env.city_time % env.n_intervals])))

    assert np.sum(state[0]) == env.n_drivers

    T += 1
print(np.mean(order_response_rates))
