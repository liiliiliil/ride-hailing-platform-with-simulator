import numpy as np


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


dist_table = cal_grids_dist_table(5,6)


def test(r, c):
    print('%d to %d: %d' % (r, c, dist_table[r][c]))

test(0, 9)
test(9, 0)
test(11, 16)
test(25, 4)