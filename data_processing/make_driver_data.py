import os
import time
import pickle

import math
import numpy as np
import linecache
import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import grid

path = 'E:\dataset\didi'
orders_file_name = 'gps_20161101_only_reserve_appearance_sampled'  # '.pkl' will be added for binary file
orders_processed_file_name = 'driver_20161101_processed'  # '.pkl' will be added for binary file
earliest_time = '2016-11-01 00:00:00'  # include
latest_time = '2016-11-02 00:00:00'  # not include
random_seed = 4321
size_hexagon_to_edge = 0.0048
hexagon_size_factor_for_plot = 1
range_map_longitude = [103.96, 104.18]
range_map_latitude = [30.59, 30.77]
time_interval_min = 10  # min
NOT_IN_MAP_GRID_ID = -100
save_data = True

size_hexagon = size_hexagon_to_edge * 2 / math.sqrt(3)  # length to the point
time_interval_sec = time_interval_min * 60

with open(os.path.join(path, orders_file_name+'.pkl'), 'rb') as f:
    orders = pickle.load(f)

n_orders = len(orders)
orders_ts = []
orders_lo = []
orders_processed = []
grid_to_order_start = {}
grid_to_order_end = {}
order_start_time_stat = [0] * (math.ceil(1440/time_interval_min))
order_end_time_stat = [0] * (math.ceil(1440/time_interval_min))
order_duration_stat = [0] * (math.ceil(1440/time_interval_min))  # duration
org_order_start_time_stat = []
org_order_end_time_stat = []
org_order_duration_stat = []  # duration

def convert_time_stamp(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

def make_time_stamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")))

earliest_time_stamp = make_time_stamp(earliest_time)
latest_time_stamp = make_time_stamp(latest_time)
org_order_start_time_stat = [0] * (latest_time_stamp-earliest_time_stamp)
org_order_end_time_stat = [0] * (latest_time_stamp-earliest_time_stamp)
org_order_duration_stat = [0] * (latest_time_stamp-earliest_time_stamp)  # duration

print('earliest_time_stamp is', earliest_time_stamp)
print('latest_time_stamp is', latest_time_stamp)
print('time_interval_sec is', time_interval_sec)
print()

def time_is_valid(t):
    return earliest_time_stamp <= t and t < latest_time_stamp

def make_time_unit(ts):
    return ts // time_interval_sec

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
    orders_lo.append(oo[3:])
    

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

for oo in orders:
    if time_is_valid(oo[1]) and time_is_valid(oo[2]):
        
        r_start, c_start, dist_start = find_grid(oo[3], oo[4])
        r_end, c_end, dist_end = find_grid(oo[5], oo[6])

        if dist_start > size_hexagon and dist_end > size_hexagon:
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
        # duration = make_time_unit(oo[2]-oo[1])
        # assert start_time + duration == end_time

        order_start_time_stat[start_time] += 1
        order_end_time_stat[end_time] += 1
        order_duration_stat[duration] += 1

        org_order_start_time_stat[start_time_stamp] += 1
        org_order_end_time_stat[end_time_stamp] += 1
        org_order_duration_stat[end_time_stamp-start_time_stamp] += 1

        if end_time_stamp-start_time_stamp == 7690:
            print()
            print('find max_duration 7690:')
            print('time', convert_time_stamp(oo[1]),'from', str([oo[3], oo[4]]), 'to', str([oo[5], oo[6]]))
            print()

        # [start_time_unit, duration_time_unit, start_grid_id, end_grid_id]
        orders_processed.append([start_time, duration,
                                start_grid_id, end_grid_id])

order_duration_stat = reduce_tail_zero(order_duration_stat)
org_order_duration_stat = reduce_tail_zero(org_order_duration_stat)
assert sum(order_duration_stat) == sum(order_start_time_stat)
assert sum(order_duration_stat) == sum(order_end_time_stat)
assert sum(org_order_duration_stat) == sum(org_order_start_time_stat)
assert sum(org_order_duration_stat) == sum(org_order_end_time_stat)

print('len(orders_processed) is', len(orders_processed))
print('maximum duration is', (len(order_duration_stat)-1))
print('maximum duration in time stamp is', (len(org_order_duration_stat)-1))
print(order_duration_stat)

if save_data:
    with open(os.path.join(path, orders_processed_file_name+'.pkl'), 'wb') as f:
        pickle.dump(orders_processed, f)
    with open(os.path.join(path, orders_processed_file_name+'_grid_to_order_start.pkl'), 'wb') as f:
        pickle.dump(grid_to_order_start, f)
    with open(os.path.join(path, orders_processed_file_name+'_grid_to_order_end.pkl'), 'wb') as f:
        pickle.dump(grid_to_order_end, f)

    with open(os.path.join(path, orders_processed_file_name), 'w') as f:
        for oo in orders_processed:
            f.write(str(oo)+'\n')

# some test
def get_center_from_grid_id(ind):
    x, y = np.unravel_index(ind, grid_index_mat.shape)
    return grid_centers_mat[x, y]

def test_get_center_from_grid_id(ind):
    print('grid id:', ind, '\tcenter:', get_center_from_grid_id(ind).tolist())

# one order from order_processed has a duration of 13 from grid 309 to grid 334, is it correct?
test_get_center_from_grid_id(309)
test_get_center_from_grid_id(334)

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
for i in range(len(grid_centers)):
    for j in range(len(grid_centers[0])):
        # plt.scatter(grid_centers_mat[i, j, 0], grid_centers_mat[i, j, 1], c='#ffff33')
        grid.set_center(grid_centers_mat[i, j])
        x, y = grid.get_x_y_for_plot()
        plt.plot(x, y, c='#ffff33', linewidth=1)
plt.scatter(orders_lo[:, 2], orders_lo[:, 3], s=1)


plt.figure(1)

plt.subplot(1, 2, 1)
plt.title('orders_processed: pick-up')
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
plt.title('orders_processed: drop-off')
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


plt.figure(2)
plt.subplot(1, 3, 1)
plt.title('order_start_time_stat')
plt.plot(list(range(len(order_start_time_stat))), order_start_time_stat)
plt.subplot(1, 3, 2)
plt.title('order_end_time_stat')
plt.plot(list(range(len(order_end_time_stat))), order_end_time_stat)
plt.subplot(1, 3, 3)
plt.title('order_duration_stat')
plt.plot(list(range(len(order_duration_stat))), order_duration_stat)

plt.figure(3)
plt.subplot(1, 3, 1)
plt.title('org_order_start_time_stat')
plt.plot(list(range(len(org_order_start_time_stat))), org_order_start_time_stat)
plt.subplot(1, 3, 2)
plt.title('org_order_end_time_stat')
plt.plot(list(range(len(org_order_end_time_stat))), org_order_end_time_stat)
plt.subplot(1, 3, 3)
plt.title('org_order_duration_stat')
plt.plot(list(range(len(org_order_duration_stat))), org_order_duration_stat)


plt.show()