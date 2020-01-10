import pickle
import numpy as np

INF = np.inf
NO_PATH = - np.inf
NOT_MATCH = -1

NEAR_ZERO = 1e-12

class KM:

    def __init__(self, graph):

        # weight of each edge
        self.graph = np.array(graph, dtype=float)
        self.min_value = np.min(self.graph)
        self.graph -= self.min_value
        # self.scale = scale

        # self.graph = (self.graph*self.scale).astype(int)
        
        
        self.has_transposed = False
        if self.graph.shape[0] > self.graph.shape[1]:
            self.graph = self.graph.T
            self.has_transposed = True
        
        self.n_x = self.graph.shape[0]
        self.n_y = self.graph.shape[1]

        # weight of each vertex
        self.w_x = np.zeros(self.n_x, dtype=int)
        self.w_y = np.zeros(self.n_y, dtype=int)
        self.init_w_of_v()

        # flag of wether vertex has been added in to path
        self.visited_x = np.zeros(self.n_x, dtype=bool)
        self.visited_y = np.zeros(self.n_y, dtype=bool)

        # match_x[i] is j means that vertex of index i in set X matches vertex of index j in set Y
        self.match_x = np.ones(self.n_x, dtype=int) * NOT_MATCH
        self.match_y = np.ones(self.n_y, dtype=int) * NOT_MATCH

        self.slack = np.ones(self.n_y) * INF
    
    def init_w_of_v(self):

        self.w_x = np.max(self.graph, axis=1)
        self.w_y = np.zeros(self.n_y)
    
    def init_path(self):
        # flag of wether vertex has been added in to path
        self.visited_x = np.zeros(self.n_x, dtype=bool)
        self.visited_y = np.zeros(self.n_y, dtype=bool)
    
    def find_path(self, u):
        """
            u: index of the beginning vertex (must in set X) in this path
        """
        self.visited_x[u] = True
        for v in range(self.n_y):
            if not self.visited_y[v] and self.graph[u][v] != np.inf:
                delta = self.w_x[u] + self.w_y[v] - self.graph[u][v]

                if delta < NEAR_ZERO:  # add v into path
                    self.visited_y[v] = True

                    # no conflict in v or path can be found
                    if self.match_y[v] == NOT_MATCH or self.find_path(self.match_y[v]):
                        self.match_x[u] = v
                        self.match_y[v] = u

                        return True
                
                elif delta > 0:  # delta is greater or equal to 0
                    self.slack[v] = min(self.slack[v], delta)
        
        return False
    
    def match(self):
        
        for u in range(self.n_x):
            self.slack = np.ones(self.n_y) * INF
            self.init_path()
            while not self.find_path(u):
                min_d = np.min(self.slack[np.logical_not(self.visited_y)])
                # print(u, min_d)

                self.w_x[self.visited_x] -= min_d
                self.w_y[self.visited_y] += min_d

                # because in these vertexes of set Y, weights of corresponding vertexes in set X
                # have been subtracted by min_d while weights of themselves and weights of corresponding
                # path have not been changed
                self.slack[np.logical_not(self.visited_y)] -= min_d

                self.init_path()
                
        return (np.sum(self.graph[np.arange(self.n_x), self.match_x]) + self.min_value*self.n_x)
    
    def get_match_result(self):
        if self.has_transposed:
            return self.match_y, self.match_x
        else:
            return self.match_x, self.match_y
    
    def set_graph(self, graph):
        self.__init__(graph)
    


def test():
    # graph = [[3, NO_PATH, 4],
    #         [2, 1, 3],
    #         [NO_PATH, NO_PATH, 5]]
    
    # # Not sure about correct result
    # graph = [[3, 4, 6, 4, 9],
    #         [6, 4, 5, 3, 8],
    #         [7, 5, 3, 4, 2],
    #         [6, 3, 2, 2, 5],
    #         [8, 4, 5, 4, 7]]

    # graph = [[1, 100],
    #         [NO_PATH, 1]]

    graph_1 = [[1, 100],
            [0, 1]]
    
    graph_2 = [[NO_PATH, 2, NO_PATH, NO_PATH,3],
            [7, NO_PATH, 23, NO_PATH, NO_PATH],
            [17, 24, NO_PATH, NO_PATH, NO_PATH],
            [NO_PATH, 6, 13, 20, NO_PATH]]
    
    km = KM(graph_2)
    res = km.match()
    match_r, match_c = km.get_match_result()

    print('match_r is', match_r)
    print('match_c is', match_c)
    print('maximum weight is', res)

    km.set_graph(graph_1)
    res = km.match()
    match_r, match_c = km.get_match_result()

    print('match_r is', match_r)
    print('match_c is', match_c)
    print('maximum weight is', res)


def test_with_given_graph(graph):
    km = KM(graph)
    res = km.match()
    match_r, match_c = km.get_match_result()

    print('match_r is', match_r)
    print('match_c is', match_c)
    print('maximum weight is', res)

    if len(match_c) >= len(match_r):
        print('Two match result is equal?', np.array([match_c[j] == i for i, j in enumerate(match_r)]).all())
        print('res is correct?', np.sum([graph[i][match_r[i]] for i in range(len(match_r))]) == res)
        print(np.sum([graph[i][match_r[i]] for i in range(len(match_r))]))
    else:
        print('Two match result is equal?', np.array([match_r[j] == i for i, j in enumerate(match_c)]).all())
        print('res is correct?', np.sum([graph[match_c[i]][i] for i in range(len(match_c))]) == res)
    
    


if __name__ == '__main__':
    with open(r'E:\Project\simulator_for_ride_hailing_platform\algorithm\graph_1.pkl', 'rb') as f:
        graph = pickle.load(f)
    
    test_with_given_graph(graph)
