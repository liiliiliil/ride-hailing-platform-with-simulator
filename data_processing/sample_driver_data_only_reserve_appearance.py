import os
import time
import pickle

import numpy as np
import linecache

path = 'E:\dataset\didi'
drivers_file_name = 'test'  # gps_20161101
n_samples = 2
save_samples = True
random_seed = 9102

drivers_file = os.path.join(path, drivers_file_name)
count = 0
drivers = []
drivers_str = []
drivers_sampled = []



def parse_driver_list(driver_list):
    return [driver_list[0], driver_list[1], int(driver_list[2]),  # driver id, order id, time stamp
            float(driver_list[3]), float(driver_list[4])]  # location

def convert_time_stamp(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

last_driver_id = ''
with open(drivers_file, 'rb') as f:
    with open(os.path.join(path, drivers_file_name+'_only_reserve_appearance'), 'w') as f_sampled:
        for l in f.readlines():
            ls = l.decode()
            ll = ls.split(',')
            if ll[0] != last_driver_id:
                s = ls.replace('\r', '')
                f_sampled.write(s)
                drivers_str.append(s)
                drivers.append(parse_driver_list(ll))
                last_driver_id = ll[0]
                count += 1

print('number of drivers is', count)


with open(os.path.join(path, drivers_file_name+'_only_reserve_appearance_sampled'), 'w') as f:
    np.random.seed(random_seed)
    for ind in np.random.choice(count, n_samples, False):
        drivers_sampled.append(drivers[ind])
        f.write(drivers_str[ind])
    

if save_samples:
    with open(os.path.join(path, drivers_file_name+'_only_reserve_appearance_sampled.pkl'), 'wb') as f:
        pickle.dump(drivers_sampled, f)

# orders_ts = np.array(orders_ts)
# orders_lo = np.array(orders_lo)

# print('orders_ts.shape is', orders_ts.shape)
# print('orders_lo.shape is', orders_lo.shape)
# print('examples: ')
# print('time stamp')
# print(orders_ts[0])
# print(orders_ts[5])
# print('location')
# print(orders_lo[0])
# print(orders_lo[5])
# print()

# range_start_time = [convert_time_stamp(np.min(orders_ts[:, 0])), convert_time_stamp(np.max(orders_ts[:, 0]))]
# range_end_time = [convert_time_stamp(np.min(orders_ts[:, 1])), convert_time_stamp(np.max(orders_ts[:, 1]))]

# range_start_longitude = [np.min(orders_lo[:, 0]), np.max(orders_lo[:, 0])]
# range_start_latitude = [np.min(orders_lo[:, 1]), np.max(orders_lo[:, 1])]
# range_end_longitude = [np.min(orders_lo[:, 2]), np.max(orders_lo[:, 2])]
# range_end_latitude = [np.min(orders_lo[:, 3]), np.max(orders_lo[:, 3])]
# range_longitude = [np.min([range_start_longitude[0], range_end_longitude[0]]), np.max([range_start_longitude[1], range_end_longitude[1]])]
# range_latitude = [np.min([range_start_latitude[0], range_end_latitude[0]]), np.max([range_start_latitude[1], range_end_latitude[1]])]

# print('range_start_time is', range_start_time)
# print('range_end_time is', range_end_time)
# print('range_start_longitude is', range_start_longitude)
# print('range_start_latitude is', range_start_latitude)
# print('range_end_longitude is', range_end_longitude)
# print('range_end_latitude is', range_end_latitude)
# print('range_longitude is', range_longitude)
# print('range_latitude is', range_latitude)

# # plot

# corner_longitude = [range_longitude[0], range_longitude[0], range_longitude[1], range_longitude[1], range_longitude[0]]
# corner_latitude = [range_latitude[0], range_latitude[1], range_latitude[1], range_latitude[0], range_latitude[0]]

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title('order: pick-up')
# plt.plot(corner_longitude, corner_latitude)
# plt.scatter(orders_lo[:, 0], orders_lo[:, 1])

# plt.subplot(1, 2, 2)
# plt.title('order: drop-off')
# plt.plot(corner_longitude, corner_latitude)
# plt.scatter(orders_lo[:, 2], orders_lo[:, 3])

# plt.show()