import os
import time
import pickle

import numpy as np
import linecache
import matplotlib.pyplot as plt

path = 'E:\dataset\didi'
orders_file_name = 'order_20161102'
n_samples = 10000
save_samples = True
random_seed = 2019

orders_file = os.path.join(path, orders_file_name)
count = 0
orders = []
orders_ts = []
orders_lo = []


def parse_order(order_str):
    order_list = order_str.split(',')
    return [order_list[0], int(order_list[1]), int(order_list[2]),  # id, ride start and stop time stamp
            float(order_list[3]), float(order_list[4]),  # pick-up location
            float(order_list[5]), float(order_list[6])]  # drop-off location

def convert_time_stamp(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

with open(orders_file, 'r') as f:
    for l in f.readlines():
        count += 1

with open(os.path.join(path, orders_file_name+'_sampled'), 'w') as f:
    np.random.seed(random_seed)
    for ind in np.random.choice(count, n_samples, False):
        order = linecache.getline(orders_file, ind)
        if save_samples:
            f.write(order)
        order_list = parse_order(order)
        orders.append(order_list)
        orders_ts.append(order_list[1:3])
        orders_lo.append(order_list[3:])

if save_samples:
    with open(os.path.join(path, orders_file_name+'_sampled.pkl'), 'wb') as f:
        pickle.dump(orders, f)

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

# plot

corner_longitude = [range_longitude[0], range_longitude[0], range_longitude[1], range_longitude[1], range_longitude[0]]
corner_latitude = [range_latitude[0], range_latitude[1], range_latitude[1], range_latitude[0], range_latitude[0]]

plt.figure()
plt.subplot(1, 2, 1)
plt.title('order: pick-up')
plt.plot(corner_longitude, corner_latitude)
plt.scatter(orders_lo[:, 0], orders_lo[:, 1])

plt.subplot(1, 2, 2)
plt.title('order: drop-off')
plt.plot(corner_longitude, corner_latitude)
plt.scatter(orders_lo[:, 2], orders_lo[:, 3])

plt.show()