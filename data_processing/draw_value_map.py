import os
import time
import pickle

import math
import numpy as np
import linecache
import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import grid

data_path = 'E:/dataset/didi/processed'
save_path = 'E:/dataset/didi/processed/order_20161101_sampled_value_map_fig'
data_file_name = 'processed_data'  # '.pkl' will be added for binary file
value_map_file_name = 'value_map'  # '.pkl' will be added for binary file

n_time_unit = 144
size_hexagon_to_edge = 0.0048
hexagon_size_factor_for_plot = 1
range_map_longitude = [103.96, 104.18]
range_map_latitude = [30.59, 30.77]

size_hexagon = size_hexagon_to_edge * 2 / math.sqrt(3)  # length to the point

if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(data_path, data_file_name+'.pkl'), 'rb') as f:
    data = pickle.load(f)
with open(os.path.join(data_path, value_map_file_name+'.pkl'), 'rb') as f:
    value_map = pickle.load(f)



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

grid_centers_flat_T = grid_centers_mat.reshape(n_grids, 2).T


max_value = np.max(value_map)
min_value = np.min(value_map)
print('maximum value in value_map is', max_value)
print('minimum value in value_map is', min_value)
# value_map = (value_map - min_value) / max_value
# max_value = np.max(value_map)
# min_value = np.min(value_map)
# print('maximum value in value_map after normalization is', max_value)
# print('minimum value in value_map after normalization is', min_value)
        


for t in range(n_time_unit):
    fig = plt.figure()
    plt.title('value map of time unit %d' % t)
    plt.scatter(grid_centers_flat_T[0], grid_centers_flat_T[1], c=value_map[t], marker='H', s=100, alpha=0.5)
    plt.colorbar()
    fig.savefig(os.path.join(save_path, '%d.jpg'%t))
    

