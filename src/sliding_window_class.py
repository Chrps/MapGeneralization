import networkx as nx


class SlidingWindow:
    def __init__(self, win_x_size=2, win_y_size=2, unit='mm'):
        # win_x_size and win_y_size size of window in meters
        self.win_x_size = win_x_size
        self.win_y_size = win_y_size
        self.unit = unit
        self.WIN_X = None
        self.WIN_Y = None
        self.STRIDE_X = None
        self.STRIDE_Y = None
        self.set_window_size()

    def set_window_size(self):
        scale = 1000
        if self.unit == 'mm':
            scale = 1000
        elif self.unit == 'cm':
            scale = 100
        elif self.unit == 'm':
            scale = 1
        elif self.unit == 'km':
            scale = 0.001
        elif self.unit == 'in':
            scale = 39.37
        elif self.unit == 'ft':
            scale = 3.28
        elif self.unit == 'mi':
            scale = 0.000621
        else:
            print("Error incorrect unit specified please select mm, cm, m, km, in, ft or mi")
            print("Using m scale!")
        self.WIN_X = self.win_x_size * scale
        self.WIN_Y = self.win_y_size * scale
        self.STRIDE_X = self.WIN_X
        self.STRIDE_Y = self.WIN_Y

    @staticmethod
    def sample_window(positions, tl, br):
        nodes = []
        for i, position in enumerate(positions):
            if tl[0] <= position[0] <= br[0] and tl[1] >= position[1] >= br[1]:
                nodes.append(i)
        return nodes

    def perform_windowing(self, nxg, unit=None):
        if unit is not None:
            self.unit = unit
            self.set_window_size()
        positions = nx.get_node_attributes(nxg, 'pos')
        positions = list(map(list, positions.values()))
        max_pos = (max(positions, key=lambda x: x[0])[0], max(positions, key=lambda x: x[1])[1])
        min_pos = (min(positions, key=lambda x: x[0])[0], min(positions, key=lambda x: x[1])[1])
        win_pos = [min_pos[0] + self.WIN_X, max_pos[1] - self.WIN_Y]
        nxg_list = []
        while win_pos[1] > min_pos[1] + self.STRIDE_Y:
            while win_pos[0] < max_pos[0] - self.STRIDE_X:
                tl = (win_pos[0] - self.WIN_X, win_pos[1] + self.WIN_Y)
                br = (win_pos[0] + self.WIN_X, win_pos[1] - self.WIN_Y)
                nodes = self.sample_window(positions, tl, br)

                sub_nxg = nxg.subgraph(nodes)
                sub_nxg = nx.relabel.convert_node_labels_to_integers(sub_nxg)

                if sub_nxg.number_of_nodes() > 0:
                    nxg_list.append(sub_nxg)

                win_pos[0] += self.STRIDE_X
            win_pos[1] -= self.STRIDE_Y
            win_pos[0] = min_pos[0] + self.WIN_X
        return nxg_list
