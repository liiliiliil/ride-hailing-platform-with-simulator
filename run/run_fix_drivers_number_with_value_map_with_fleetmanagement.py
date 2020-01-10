
from collections import defaultdict
import sys
import traceback
import os
import pickle

sys.path.append(os.path.join(sys.path[0], "../"))
from simulator.envs import *
from algorithm.od_and_fm import *

data_path = 'E:/dataset/didi/processed'
data_file_name = 'processed_data'  # '.pkl' will be added for binary file
value_file_map_name = 'value_map_20161101'  # '.pkl' will be added for binary file
historical_orders_stat_file_name = 'order_20161101_processed_number_stat'  # '.pkl' will be added for binary file
save_path = 'experiment'
save_file_name = '20161102_experiment_value_map_fm_0_33_150'
save_data = False
n_time_units = 144
max_n_drivers = 150
max_dispatch_dist = 6
gamma = 0.7 # 1 / math.e
forecasting_time = 1
neighbor_dist = 3
ratio = 0.33  # ratio of unmanned cars in total cars
l_max = 9

seed = 9102
np.random.seed(seed)


with open(os.path.join(data_path, data_file_name+'.pkl'), 'rb') as f:
    data = pickle.load(f)
with open(os.path.join(data_path, value_file_map_name+'.pkl'), 'rb') as f:
    value_map = pickle.load(f)
with open(os.path.join(data_path, historical_orders_stat_file_name+'.pkl'), 'rb') as f:
    historical_orders_stat = pickle.load(f)
historical_orders_stat = np.array(historical_orders_stat).reshape((n_time_units, -1))

mapped_matrix_int = np.array(data['map'])  # in order to make every index greater than 0
order_real = np.array(data['orders'])
drivers_number_time_distribution = np.array(data['drivers_distribution']['time'])
drivers_number_grid_distribution = np.array(data['drivers_distribution']['grid']).reshape(-1)
grids_dist_table = np.array(data['dist_table'])

M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = M * N
idle_driver_location_mat = np.zeros((144, num_valid_grid))

assert (mapped_matrix_int.reshape(-1) == (np.arange(num_valid_grid))).all()

extra_time_for_pickup = np.floor(np.arange(M+N)/3).astype(int)


for ii in np.arange(144):
    # each element is number of idle drivers at particular grid
    # print(max_n_drivers * drivers_number_time_distribution[ii])
    idle_driver_location_mat[ii, :] = np.round((max_n_drivers * drivers_number_time_distribution[ii]) * drivers_number_grid_distribution)
    # print(np.sum(idle_driver_location_mat[ii, :]))
    # print(idle_driver_location_mat[ii, :])

# fix drivers number
# idle_driver_location_mat[0, :] = idle_driver_location_mat[np.argmax(drivers_number_time_distribution), :]
drivers_number_distribution = [1]*max_n_drivers+[0]*(num_valid_grid-max_n_drivers)
np.random.shuffle(drivers_number_distribution)
idle_driver_location_mat[0, :] = drivers_number_distribution
print(drivers_number_distribution)
print(np.sum(idle_driver_location_mat[0, :]))
# print(idle_driver_location_mat[0, :])
# print(np.max(drivers_number_time_distribution))
# print(np.argmax(drivers_number_time_distribution))
# print(drivers_number_time_distribution[np.argmax(drivers_number_time_distribution)])
idle_driver_dist_time = np.stack([np.round(max_n_drivers * drivers_number_time_distribution), np.zeros(drivers_number_time_distribution.shape)], axis=1)

# fix drivers number
# idle_driver_dist_time[0] = idle_driver_dist_time[np.argmax(drivers_number_time_distribution)]
idle_driver_dist_time[0] = np.sum(idle_driver_location_mat[0, :])


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
                order_time, order_price, l_max, M, N, n_side, 1, order_real, onoff_driver_location_mat,
                extra_time_for_pickup)

od_fm = OrderDispatchAndFleetManagement(value_map, grids_dist_table, extra_time_for_pickup,
                                                max_dispatch_dist, gamma, historical_orders_stat,
                                                forecasting_time, neighbor_dist, ratio)

state, orders = env.reset_clean_with_extra_order_info()  # use real orders instead of randomly generated orders
moment = env.city_time % env.n_intervals

# print('moment is', moment)
# print('len(env.orders) is', len(orders))
# if len(orders) > 0:
#     match_order_to_grid, management = od_fm.step(moment, state[0].reshape(-1), orders)
#     print(match_order_to_grid)
#     print(management)


order_response_rates = []
T = 0
max_iter = n_time_units - 1  # n_time_units - 1
total_profit = []
managements = []

while T < max_iter:

    if T != 143:
        print('moment is', moment)
        print('len(env.orders) is', len(orders))
        if len(orders) > 0:
            match_order_to_grid, management = od_fm.step(moment, state[0].reshape(-1), orders)
            # print(match_order_to_grid)
            # print(management)
            managements.append(management)
    
    state, orders, profit = env.step_with_pickup_time_and_waiting_time(management, match_order_to_grid)  # generate_order: 1 for random, 2 for real data

    moment = env.city_time % env.n_intervals
    # print('shape of state is', state.shape)
    # print('shape of state[0] is', state[0].shape)
    

    order_response_rates.append(env.order_response_rate)
    total_profit.append(profit)

    # print(len(management))
    # print('%d == %d' % (np.sum(state[0]), env.n_drivers))
    assert np.sum(state[0]) == env.n_drivers

    T += 1

print('order response rate is', order_response_rates)
print('average order response rate is', np.mean(np.delete(order_response_rates, -1)))
print('total profit is', np.sum(total_profit))

if save_data:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with open(os.path.join(save_path, save_file_name+'_result.pkl'), 'wb') as f:
        pickle.dump({'response_rate': order_response_rates,
                    'profit': total_profit,
                    'managements': managements}, f)

