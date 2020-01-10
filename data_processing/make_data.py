import os
import time
import pickle

import math
import numpy as np
import linecache
import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import grid

data_path = 'E:/dataset/didi'
save_path = 'E:/dataset/didi/processed'
orders_file_name = 'order_20161101_sampled'  # '.pkl' will be added for binary file
orders_processed_file_name = 'order_20161101_processed'  # '.pkl' will be added for binary file
drivers_file_name = 'gps_20161101_only_reserve_presence_sampled'  # '.pkl' will be added for binary file
drivers_processed_file_name = 'driver_20161101_processed'  # '.pkl' will be added for binary file
earliest_time = '2016-11-01 00:00:00'  # include
latest_time = '2016-11-02 00:00:00'  # not include
random_seed = 1234
size_hexagon_to_edge = 0.0048
hexagon_size_factor_for_plot = 1
range_map_longitude = [103.96, 104.18]
range_map_latitude = [30.59, 30.77]
time_interval_min = 10  # min
max_order_waiting_time_min = 30  # min
NOT_IN_MAP_GRID_ID = -100
alpha = 2.0  # it is used for calculation of price
# mu = [104.07, 30.6642]  # it is used for drivers_number_grid_distribution; center of the distribution
# sigma = 0.0536  # it is used for drivers_number_grid_distribution; 2 * sigma is the distance from center to the edge
save_data = True
plot_fig = True


if not os.path.exists(save_path):
    os.mkdir(save_path)

size_hexagon = size_hexagon_to_edge * 2 / math.sqrt(3)  # length to the point
time_interval_sec = time_interval_min * 60
n_time_unit = math.ceil(1440/time_interval_min)
max_order_waiting_time_unit = max_order_waiting_time_min // time_interval_min

with open(os.path.join(data_path, orders_file_name+'.pkl'), 'rb') as f:
    orders = pickle.load(f)
with open(os.path.join(data_path, drivers_file_name+'.pkl'), 'rb') as f:
    drivers = pickle.load(f)

grids_dist = None  # it will become np.narray for convenience with shape of [n_grids, n_grids]

n_orders = len(orders)
orders_ts = []
orders_lo = []
orders_processed = []
grid_to_order_start = {}
grid_to_order_end = {}
orders_start_number_stat_over_time_and_grid = []  # it will become np.narray for convenience
orders_start_number_stat_over_grid = []  # it will become np.narray for convenience
order_start_time_stat = [0] * n_time_unit
order_end_time_stat = [0] * n_time_unit
order_duration_stat = [0] * n_time_unit  # duration
org_order_start_time_stat = []  # org means original, i.e. time unit is sec
org_order_end_time_stat = []
org_order_duration_stat = []  # duration

drivers_processed = []
grid_to_driver_on = {}
grid_to_driver_off = {}
drivers_number_stat = []  # list of which len is n_time_unit and it will become np.narray for convenience
drivers_on_number_stat = []
drivers_off_number_stat = []
org_drivers_number_stat = []  # it will become np.narray for convenience
org_drivers_on_number_stat = []
org_drivers_off_number_stat = []
drivers_number_time_distribution = None
drivers_number_time_distribution = None

def convert_time_stamp(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

def make_time_stamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")))

earliest_time_stamp = make_time_stamp(earliest_time)
latest_time_stamp = make_time_stamp(latest_time)
total_time_stamp = latest_time_stamp-earliest_time_stamp
org_order_start_time_stat = [0] * total_time_stamp
org_order_end_time_stat = [0] * total_time_stamp
org_order_duration_stat = [0] * total_time_stamp  # duration
org_drivers_number_stat = np.zeros(total_time_stamp)
org_drivers_on_number_stat = [0] * total_time_stamp
org_drivers_off_number_stat = [0] * total_time_stamp

print('earliest_time_stamp is', earliest_time_stamp)
print('latest_time_stamp is', latest_time_stamp)
print('time_interval_sec is', time_interval_sec)
print()

def time_is_valid(t):
    return earliest_time_stamp <= t and t < latest_time_stamp

def make_time_unit(ts):
    return ts // time_interval_sec

def cal_grids_dist_table(n_r, n_c):
    n_grids = n_r * n_c
    grid_dist_table = np.ones((n_grids, n_grids), dtype=int) * -1
    grid_index_mat = np.arange(n_grids).reshape(n_r, n_c)
    def get_hexagon_neighbors_indexes(ind):
        '''
                    0
                5       1
                  center
                4       2
                    3
        return indexes of neighbor 0-5 in the list
        '''
        nbs = [None] * 6
        i, j = np.unravel_index(ind, grid_index_mat.shape)
        if j % 2 == 0:
            if i - 1 >= 0:
                nbs[0] = grid_index_mat[i-1, j]
            if j + 1 < n_c:
                nbs[1] = grid_index_mat[i, j+1]
            if i + 1 < n_r and j + 1 < n_c:
                nbs[2] = grid_index_mat[i+1, j+1]
            if i + 1 < n_r:
                nbs[3] = grid_index_mat[i+1, j]
            if i + 1 < n_r and j - 1 >= 0:
                nbs[4] = grid_index_mat[i+1, j-1]
            if j - 1 >= 0:
                nbs[5] = grid_index_mat[i, j-1]
        elif j % 2 == 1:
            if i - 1 >= 0:
                nbs[0] = grid_index_mat[i-1, j]
            if i - 1 >= 0 and j + 1 < n_c:
                nbs[1] = grid_index_mat[i-1, j+1]
            if j + 1 < n_c:
                nbs[2] = grid_index_mat[i, j+1]
            if i + 1 < n_r:
                nbs[3] = grid_index_mat[i+1, j]
            if j - 1 >= 0:
                nbs[4] = grid_index_mat[i, j-1]
            if i - 1 >= 0 and j - 1 >= 0:
                nbs[5] = grid_index_mat[i-1, j-1]
        return np.array([n for n in nbs if n != None])

    neighbors_table = [None] * n_grids
    for i in range(n_grids):
        neighbors_table[i] = get_hexagon_neighbors_indexes(i)
    
    def get_neighbors_of_grids(grids_array):
        """ include themselves """
        return np.unique(np.concatenate([neighbors_table[g] for g in grids_array], axis=0))

    # grids themselves
    for i in range(n_grids):
        grid_dist_table[i, i] = 0
    # neighbors with distances of 1
    for i in range(n_grids):
        grid_dist_table[i][neighbors_table[i]] = 1
    
    for i in range(n_grids):
        cur_grids = neighbors_table[i]
        mask = grid_dist_table[i] == -1
        mask[np.arange(i+1)] = False
        mask[cur_grids] = False
        while mask.any():
            cur_grids_neighbors = get_neighbors_of_grids(cur_grids)
            new_dist = grid_dist_table[i][cur_grids[0]] + 1
            cur_grids = []
            for g in cur_grids_neighbors:
                if mask[g]:
                    grid_dist_table[i][g] = new_dist
                    mask[g] = False
                    cur_grids.append(g)
            cur_grids = np.array(cur_grids)
    
    # print(grid_dist_table)
    grid_dist_table_T = grid_dist_table.T.reshape(-1)
    grid_dist_table = grid_dist_table.reshape(-1)
    mask = grid_dist_table == -1
    grid_dist_table[mask] = grid_dist_table_T[mask]

    return grid_dist_table.reshape(n_grids, n_grids)



def cal_min_dist(p, mat):
    # print(mat.shape)
    x_dist_mat = mat[:, :, 0] - p[0]
    y_dist_mat = mat[:, :, 1] - p[1]
    dist_mat = (x_dist_mat**2 + y_dist_mat**2)
    ind_x, ind_y = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
    
    return ind_x, ind_y, dist_mat[ind_x, ind_y]**0.5

def reduce_tail_zero(l):
    i = len(l) - 1
    while l[i] == 0:
        i -= 1
    return l[:i+1]

for oo in orders:
    orders_ts.append(oo[1:3])
    orders_lo.append(oo[3:])  # location
    

assert len(orders_ts) == n_orders
assert len(orders_lo) == n_orders

orders_ts = np.array(orders_ts)
orders_lo = np.array(orders_lo)

print('orders_ts.shape is', orders_ts.shape)
print('orders_lo.shape is', orders_lo.shape)
print('examples: ')
print('time stamp')
print(orders_ts[0])
print(orders_ts[5])
print('location')
print(orders_lo[0])
print(orders_lo[5])
print()

range_start_time = [convert_time_stamp(np.min(orders_ts[:, 0])), convert_time_stamp(np.max(orders_ts[:, 0]))]
range_end_time = [convert_time_stamp(np.min(orders_ts[:, 1])), convert_time_stamp(np.max(orders_ts[:, 1]))]

range_start_longitude = [np.min(orders_lo[:, 0]), np.max(orders_lo[:, 0])]
range_start_latitude = [np.min(orders_lo[:, 1]), np.max(orders_lo[:, 1])]
range_end_longitude = [np.min(orders_lo[:, 2]), np.max(orders_lo[:, 2])]
range_end_latitude = [np.min(orders_lo[:, 3]), np.max(orders_lo[:, 3])]
range_longitude = [np.min([range_start_longitude[0], range_end_longitude[0]]), np.max([range_start_longitude[1], range_end_longitude[1]])]
range_latitude = [np.min([range_start_latitude[0], range_end_latitude[0]]), np.max([range_start_latitude[1], range_end_latitude[1]])]

print('range_start_time is', range_start_time)
print('range_end_time is', range_end_time)
print('range_start_longitude is', range_start_longitude)
print('range_start_latitude is', range_start_latitude)
print('range_end_longitude is', range_end_longitude)
print('range_end_latitude is', range_end_latitude)
print('range_longitude is', range_longitude)
print('range_latitude is', range_latitude)


# make hexagon
grid = grid.Hexagon(size_to_edge=size_hexagon_to_edge*hexagon_size_factor_for_plot)
grid_interval_lo = size_hexagon * 1.5
grid_interval_la = size_hexagon_to_edge * 2

grid_centers = []
for la in np.arange(range_map_latitude[1]-size_hexagon, range_map_latitude[0]-0.00001, -grid_interval_la):
    row = []
    count = 0
    for lo in np.arange(range_map_longitude[0], range_map_longitude[1]+0.00001, grid_interval_lo):
        if count % 2 == 0:
            row.append([lo, la])
        else:
            row.append([lo, la+size_hexagon_to_edge])
        count += 1
    grid_centers.append(row)

grid_centers_mat = np.array(grid_centers)
shape_grid_centers_mat = grid_centers_mat.shape
n_grids = shape_grid_centers_mat[0]*shape_grid_centers_mat[1]

grid_index_mat = np.arange(n_grids).reshape(shape_grid_centers_mat[:2])

print('shape of grids is', shape_grid_centers_mat)
print('number of grids is', n_grids)

grids_dist_table = cal_grids_dist_table(shape_grid_centers_mat[0], shape_grid_centers_mat[1])

orders_start_number_stat_over_time_and_grid = np.zeros((n_time_unit, shape_grid_centers_mat[0], shape_grid_centers_mat[1]))
orders_start_number_stat_over_grid = np.zeros((shape_grid_centers_mat[0], shape_grid_centers_mat[1]))
drivers_number_stat = np.zeros(n_time_unit)
drivers_on_number_stat = [0] * n_time_unit
drivers_off_number_stat = [0] * n_time_unit


def find_grid(lo, la):
    c = np.argmin(np.abs(grid_centers_mat[0, :, 0] - lo))
    if grid_centers_mat[0, c, 0] - lo > 0:
        r, bias_c, dist = cal_min_dist([lo, la], grid_centers_mat[:, max(c-1, 0):c+1])
        # if c is 0, bias_c must be 0, and then the result will be 0
        # if c is not 0 and bias_c is 0, then result will be c-1
        # if c is not 0 and bias_c is not 0, then result will be c
        c = max(0, c-1+bias_c)
    else:
        r, bias_c, dist = cal_min_dist([lo, la], grid_centers_mat[:, c:c+2])
        # if bias_c is 0, then result will be c
        # if bias_c is 1, then result will be c+1
        c += bias_c
    return r, c, dist

# orders
speed_list = []

### for construct the value map of each time unit ###
orders_processed_sort_by_time = [[] for _ in range(n_time_unit)]

for oo in orders:
    if time_is_valid(oo[1]) and time_is_valid(oo[2]):
        
        r_start, c_start, dist_start = find_grid(oo[3], oo[4])
        r_end, c_end, dist_end = find_grid(oo[5], oo[6])

        # dist_start > size_hexagon and dist_end > size_hexagon
        if dist_start > size_hexagon or dist_end > size_hexagon:
            continue

        if dist_start > size_hexagon:
            start_grid_id = NOT_IN_MAP_GRID_ID
        else:
            start_grid_id = grid_index_mat[r_start, c_start]
        if dist_end > size_hexagon:
            end_grid_id = NOT_IN_MAP_GRID_ID
        else:
            end_grid_id = grid_index_mat[r_end, c_end]

        if start_grid_id in grid_to_order_start:
            grid_to_order_start[start_grid_id].append([oo[3], oo[4]])
        else:
            grid_to_order_start[start_grid_id] = [[oo[3], oo[4]]]
        if end_grid_id in grid_to_order_end:
            grid_to_order_end[end_grid_id].append([oo[5], oo[6]])
        else:
            grid_to_order_end[end_grid_id] = [[oo[5], oo[6]]]
        
        start_time_stamp = oo[1]-earliest_time_stamp
        end_time_stamp = oo[2]-earliest_time_stamp

        start_time = make_time_unit(start_time_stamp)
        end_time = make_time_unit(end_time_stamp)
        duration = end_time - start_time
        if duration == 0:
            continue
        # duration = make_time_unit(oo[2]-oo[1])
        # assert start_time + duration == end_time

        price = grids_dist_table[start_grid_id][end_grid_id] + alpha * duration

        waiting_time = np.random.randint(0, min((max_order_waiting_time_unit, duration))+1)

        orders_start_number_stat_over_grid[r_start, c_start] += 1
        orders_start_number_stat_over_time_and_grid[start_time, r_start, c_start] += 1

        order_start_time_stat[start_time] += 1
        order_end_time_stat[end_time] += 1
        order_duration_stat[duration] += 1

        org_order_start_time_stat[start_time_stamp] += 1
        org_order_end_time_stat[end_time_stamp] += 1
        org_order_duration_stat[end_time_stamp-start_time_stamp] += 1

        speed = grids_dist_table[start_grid_id][end_grid_id] / duration
        speed_list.append(speed)

        # if end_time_stamp-start_time_stamp == 7690:
        #     print()
        #     print('find max_duration 7690:')
        #     print('time', convert_time_stamp(oo[1]),'from', str([oo[3], oo[4]]), 'to', str([oo[5], oo[6]]))
        #     print()

        # [start_grid_id, end_grid_id, start_time_unit, duration_time_unit, price, waiting_time]
        orders_processed.append([start_grid_id, end_grid_id,
                                start_time, duration, price, waiting_time])

        orders_processed_sort_by_time[start_time].append([start_grid_id, end_grid_id, duration, price])

print('average speed is', np.mean(speed_list))
print(np.bincount(np.round(speed_list).astype(int)))

order_duration_stat = reduce_tail_zero(order_duration_stat)
org_order_duration_stat = reduce_tail_zero(org_order_duration_stat)
assert sum(order_duration_stat) == sum(order_start_time_stat)
assert sum(order_duration_stat) == sum(order_end_time_stat)
assert sum(org_order_duration_stat) == sum(org_order_start_time_stat)
assert sum(org_order_duration_stat) == sum(org_order_end_time_stat)
# print(orders_start_number_stat_over_grid.shape, np.sum(orders_start_number_stat_over_grid), sum(order_start_time_stat))
assert np.sum(orders_start_number_stat_over_grid) == sum(order_start_time_stat)

print('len(orders_processed) is', len(orders_processed))
print('maximum duration is', (len(order_duration_stat)-1))
print('maximum duration in time stamp is', (len(org_order_duration_stat)-1))
print(order_duration_stat)


# drivers
range_on_driver_lo, range_on_driver_la = [100000, -100000], [100000, -100000]
range_off_driver_lo, range_off_driver_la = [100000, -100000], [100000, -100000]
# drivers is [driver id, on time stamp, off time stamp, on lo, on la, off lo, off la]
for dd in drivers:
    if time_is_valid(dd[1]) and time_is_valid(dd[2]):
        r_on, c_on, dist_on = find_grid(dd[3], dd[4])
        r_off, c_off, dist_off = find_grid(dd[5], dd[6])
        if dist_on > size_hexagon and dist_off > size_hexagon:
            continue
        
        on_grid_id = grid_index_mat[r_on, c_on]
        off_grid_id = grid_index_mat[r_off, c_off]

        on_time_stamp = dd[1]-earliest_time_stamp
        off_time_stamp = dd[2]-earliest_time_stamp
        on_time = make_time_unit(on_time_stamp)
        off_time = make_time_unit(off_time_stamp)

        if on_grid_id in grid_to_driver_on:
            grid_to_driver_on[on_grid_id].append([dd[3], dd[4]])
        else:
            grid_to_driver_on[on_grid_id] = [[dd[3], dd[4]]]
        if off_grid_id in grid_to_driver_off:
            grid_to_driver_off[off_grid_id].append([dd[5], dd[6]])
        else:
            grid_to_driver_off[off_grid_id] = [[dd[5], dd[6]]]

        drivers_on_number_stat[on_time] += 1
        drivers_off_number_stat[off_time] += 1
        drivers_number_stat[on_time:off_time+1] += 1
        org_drivers_on_number_stat[on_time_stamp] += 1
        org_drivers_off_number_stat[off_time_stamp] += 1
        org_drivers_number_stat[on_time_stamp:off_time_stamp+1] += 1

        range_on_driver_lo[0] = min(range_on_driver_lo[0], dd[3])
        range_on_driver_lo[1] = max(range_on_driver_lo[1], dd[3])
        range_on_driver_la[0] = min(range_on_driver_la[0], dd[4])
        range_on_driver_la[1] = max(range_on_driver_la[1], dd[4])
        range_off_driver_lo[0] = min(range_off_driver_lo[0], dd[5])
        range_off_driver_lo[1] = max(range_off_driver_lo[1], dd[5])
        range_off_driver_la[0] = min(range_off_driver_la[0], dd[6])
        range_off_driver_la[1] = max(range_off_driver_la[1], dd[6])

        # [on_time_unit, off_time_unit, on_grid_id, off_grid_id]
        drivers_processed.append([on_time, off_time, on_grid_id, off_grid_id])

assert sum(drivers_on_number_stat) == sum(org_drivers_on_number_stat)
assert sum(drivers_off_number_stat) == sum(org_drivers_off_number_stat)

print('len(drivers_processed) is', len(drivers_processed))
print('range_on_driver_lo is', range_on_driver_lo)
print('range_on_driver_la is', range_on_driver_la)
print('range_off_driver_lo is', range_off_driver_lo)
print('range_off_driver_la is', range_off_driver_la)
print('minimum number of drivers on this day is', np.min(drivers_number_stat))
print('maximum number of drivers on this day is', np.max(drivers_number_stat))
print('average number of drivers on this day is', np.mean(drivers_number_stat))
print('std of number of drivers on this day is', np.std(drivers_number_stat))
print('minimum number of drivers happens at', np.argmin(drivers_number_stat))
print('maximum number of drivers on this day is', np.argmax(drivers_number_stat))

drivers_number_time_distribution = drivers_number_stat / np.max(drivers_number_stat)

# # normal distribution
# drivers_number_grid_distribution = sigma * np.random.randn(shape_grid_centers_mat[0], shape_grid_centers_mat[1])\
#                                     + mu[0] * np.ones((shape_grid_centers_mat[0], shape_grid_centers_mat[1])) + mu[1] * np.ones(shape_grid_centers_mat[1])
# drivers_number_grid_distribution = drivers_number_grid_distribution / np.sum(drivers_number_grid_distribution)  # normalization

# orders distribution
drivers_number_grid_distribution = orders_start_number_stat_over_grid / np.sum(orders_start_number_stat_over_grid)

# print(np.sum(drivers_number_grid_distribution))


### for construct the value map of each time unit ###
# orders_processed_sort_by_time[start_time] is [start_grid_id, end_grid_id, duration, price]
gamma = 1 / math.e
extra_time_for_pickup = np.floor(np.arange(shape_grid_centers_mat[0]+shape_grid_centers_mat[0])/3).astype(int)
value_map = np.zeros((n_time_unit, n_grids))
n_order_start = np.zeros((n_time_unit, n_grids), dtype=int)
total_time, end_time, end_value = -1, -1, 0
for t in range(n_time_unit-1, -1, -1):
    for oo in orders_processed_sort_by_time[t]:
        for g_ind in range(n_grids):
            n_order_start[t][g_ind] += 1
            total_time = extra_time_for_pickup[grids_dist_table[g_ind][oo[0]]]+oo[2]
            end_time = t + total_time
            if end_time >= n_time_unit:
                end_value = 0
            else:
                # print(type(end_time), type(oo[1]))
                end_value = value_map[end_time][oo[1]]
            value_map[t][g_ind] +=\
                (gamma**(total_time)*end_value + oo[3] - value_map[t][g_ind]) / n_order_start[t][g_ind]
max_value = np.max(value_map)
min_value = np.min(value_map)
print('maximum value in value_map is', max_value)
print('minimum value in value_map is', min_value)
# value_map = (value_map - min_value) / max_value
# max_value = np.max(value_map)
# min_value = np.min(value_map)
# print('maximum value in value_map after normalization is', max_value)
# print('minimum value in value_map after normalization is', min_value)
        




if save_data:
    with open(os.path.join(save_path, orders_processed_file_name+'.pkl'), 'wb') as f:
        pickle.dump(orders_processed, f)
    with open(os.path.join(save_path, orders_processed_file_name+'_number_stat.pkl'), 'wb') as f:
        pickle.dump(orders_start_number_stat_over_time_and_grid.tolist(), f)
    # with open(os.path.join(save_path, orders_processed_file_name+'_grid_to_order_start.pkl'), 'wb') as f:
    #     pickle.dump(grid_to_order_start, f)
    # with open(os.path.join(save_path, orders_processed_file_name+'_grid_to_order_end.pkl'), 'wb') as f:
    #     pickle.dump(grid_to_order_end, f)
    with open(os.path.join(save_path, orders_processed_file_name), 'w') as f:
        for oo in orders_processed:
            f.write(str(oo)+'\n')

    with open(os.path.join(save_path, drivers_processed_file_name+'.pkl'), 'wb') as f:
        pickle.dump(drivers_processed, f)
    with open(os.path.join(save_path, drivers_processed_file_name+'_number_stat.pkl'), 'wb') as f:
        pickle.dump(drivers_number_stat.tolist(), f)
    with open(os.path.join(save_path, drivers_processed_file_name), 'w') as f:
        for dd in drivers_processed:
            f.write(str(dd)+'\n')
    
    data_for_simulator = {'map': grid_index_mat, 'orders': orders_processed,
                        'drivers_distribution': {'time': drivers_number_time_distribution.tolist(),
                                                'grid': drivers_number_grid_distribution.tolist()},
                        'dist_table': grids_dist_table}
    with open(os.path.join(save_path, 'processed_data.pkl'), 'wb') as f:
        pickle.dump(data_for_simulator, f)
    
    with open(os.path.join(save_path, 'value_map.pkl'), 'wb') as f:
        pickle.dump(value_map.tolist(), f)

    


# some test
def get_center_from_grid_id(ind):
    x, y = np.unravel_index(ind, grid_index_mat.shape)
    return grid_centers_mat[x, y]

def test_get_center_from_grid_id(ind):
    print('grid id:', ind, '\tcenter:', get_center_from_grid_id(ind).tolist())

# one order from order_processed has a duration of 13 from grid 309 to grid 334, is it correct?
test_get_center_from_grid_id(309)
test_get_center_from_grid_id(334)


if plot_fig:
    # plot
    corner_longitude = [range_longitude[0], range_longitude[0], range_longitude[1], range_longitude[1], range_longitude[0]]
    corner_latitude = [range_latitude[0], range_latitude[1], range_latitude[1], range_latitude[0], range_latitude[0]]

    plt.figure(0)

    plt.subplot(1, 2, 1)
    plt.title('order: pick-up')
    plt.plot(corner_longitude, corner_latitude)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)
            plt.text(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], str(grid_index_mat[i, j]))
    plt.scatter(orders_lo[:, 0], orders_lo[:, 1], s=1)

    plt.subplot(1, 2, 2)
    plt.title('order: drop-off')
    plt.plot(corner_longitude, corner_latitude)

    neighbors = np.where(np.logical_and(grids_dist_table[50]<=3,
                                        grids_dist_table[50]>0))
    print(neighbors)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)
            plt.text(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], grids_dist_table[300][grid_index_mat[i, j]])
    for ind in neighbors:
        i, j = ind // shape_grid_centers_mat[1], ind % shape_grid_centers_mat[1]
        plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], s=200)
    plt.scatter(orders_lo[:, 2], orders_lo[:, 3], s=1)

    

    fig = plt.figure(1, figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title('orders: pick-up')
    # plt.plot(corner_longitude, corner_latitude)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)

            ind = grid_index_mat[i, j]
            if ind in grid_to_order_start:
                x, y = zip(*grid_to_order_start[ind])
                plt.scatter(x, y, s=3)

    plt.subplot(1, 2, 2)
    plt.title('orders: drop-off')
    # plt.plot(corner_longitude, corner_latitude)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)
            
            ind = grid_index_mat[i, j]
            if ind in grid_to_order_end:
                x, y = zip(*grid_to_order_end[ind])
                plt.scatter(x, y, s=3)

    fig.savefig(os.path.join(save_path, 'orders_distribution.jpg'))

    fig = plt.figure(2, figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.title('order_start_time_stat')
    plt.plot(list(range(len(order_start_time_stat))), order_start_time_stat)
    plt.subplot(1, 3, 2)
    plt.title('order_end_time_stat')
    plt.plot(list(range(len(order_end_time_stat))), order_end_time_stat)
    plt.subplot(1, 3, 3)
    plt.title('order_duration_stat')
    plt.plot(list(range(len(order_duration_stat))), order_duration_stat)

    fig.savefig(os.path.join(save_path, 'orders_duration.jpg'))

    # plt.figure(3)
    # plt.subplot(1, 3, 1)
    # plt.title('org_order_start_time_stat')
    # plt.plot(list(range(len(org_order_start_time_stat))), org_order_start_time_stat)
    # plt.subplot(1, 3, 2)
    # plt.title('org_order_end_time_stat')
    # plt.plot(list(range(len(org_order_end_time_stat))), org_order_end_time_stat)
    # plt.subplot(1, 3, 3)
    # plt.title('org_order_duration_stat')
    # plt.plot(list(range(len(org_order_duration_stat))), org_order_duration_stat)

    fig = plt.figure(4, figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('drivers: first appear')
    # plt.plot(corner_longitude, corner_latitude)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)
            
            ind = grid_index_mat[i, j]
            if ind in grid_to_driver_on:
                x, y = zip(*grid_to_driver_on[ind])
                plt.scatter(x, y, s=3)

    plt.subplot(1, 2, 2)
    plt.title('drivers: last appear')
    # plt.plot(corner_longitude, corner_latitude)
    for i in range(len(grid_centers)):
        for j in range(len(grid_centers[0])):
            # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
            grid.set_center(grid_centers_mat[i, j])
            x, y = grid.get_x_y_for_plot()
            plt.plot(x, y, c='#ffff33', linewidth=1)
            
            ind = grid_index_mat[i, j]
            if ind in grid_to_driver_off:
                x, y = zip(*grid_to_driver_off[ind])
                plt.scatter(x, y, s=3)
    
    fig.savefig(os.path.join(save_path, 'drivers_distribution.jpg'))

    plt.figure(5)
    plt.subplot(1, 3, 1)
    plt.title('drivers_on_number_stat')
    plt.bar(list(range(len(drivers_on_number_stat))), drivers_on_number_stat)
    plt.subplot(1, 3, 2)
    plt.title('drivers_off_number_stat')
    plt.bar(list(range(len(drivers_off_number_stat))), drivers_off_number_stat)
    plt.subplot(1, 3, 3)
    plt.title('drivers_number_stat')
    plt.bar(list(range(len(drivers_number_stat))), drivers_number_stat)

    # plt.figure(6)
    # plt.subplot(1, 3, 1)
    # plt.title('org_drivers_on_number_stat')
    # plt.bar(list(range(len(org_drivers_on_number_stat))), org_drivers_on_number_stat)
    # plt.subplot(1, 3, 2)
    # plt.title('org_drivers_off_number_stat')
    # plt.bar(list(range(len(org_drivers_off_number_stat))), org_drivers_off_number_stat)
    # plt.subplot(1, 3, 3)
    # plt.title('org_drivers_number_stat')
    # plt.bar(list(range(len(org_drivers_number_stat))), org_drivers_number_stat)

    plt.figure(7)
    grid_centers_flat = grid_centers_mat.reshape(n_grids, 2)
    plt.subplot(1, 2, 1)
    plt.title('value map of time unit 48 (8:00-8:10)')
    # plt.plot(corner_longitude, corner_latitude)
    plt.bar(np.arange(n_grids), value_map[48])

    plt.subplot(1, 2, 2)
    plt.title('value map of time unit 105 (17:30-17:40)')
    # plt.plot(corner_longitude, corner_latitude)
    plt.bar(np.arange(n_grids), value_map[105])


    plt.figure(8)

    # norm = plt.Normalize(0, max_value)
    grid_centers_flat_T = grid_centers_mat.reshape(n_grids, 2).T
    plt.subplot(1, 2, 1)
    plt.title('value map of time unit 48 (8:00-8:10)')
    # plt.plot(corner_longitude, corner_latitude)
    plt.scatter(grid_centers_flat_T[0], grid_centers_flat_T[1], c=value_map[48], marker='H', s=200, alpha=0.5)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('value map of time unit 105 (17:30-17:40)')
    # plt.plot(corner_longitude, corner_latitude)
    plt.scatter(grid_centers_flat_T[0], grid_centers_flat_T[1], c=value_map[105], marker='H', s=200, alpha=0.5)
    plt.colorbar()


    plt.figure(9)
    plt.title('sum of value map at each time unit')
    # plt.plot(corner_longitude, corner_latitude)
    # print(value_map.shape, np.sum(value_map, axis=1).shape)
    plt.bar(np.arange(n_time_unit), np.sum(value_map, axis=1))

    # fig_path = os.path.join(save_path, orders_file_name+'_value_map_fig')
    # if not os.path.exists(fig_path):
    #     os.mkdir(fig_path)

            
    plt.show()