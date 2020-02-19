import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import os
from src.dxf_utils import DxfReader

class Node:
    '''Class for keeping track of nodes in network'''
    def __init__(self, id, coordinate):
        self.id = id
        self.coordinate = coordinate
        self.connected_to = []



def main():
    directory_of_script = os.path.dirname(os.getcwd())
    dxf_file_path = 'data/dxf_files/MSP1-HoM-MA-XX+4-ET.dxf'
    dxf_file_path = directory_of_script + r'\\' + dxf_file_path

    dxf_obj = DxfReader(dxf_file_path=dxf_file_path)
    data = dxf_obj.extract_data()

    #list1 = [(0,0),(10,10),(20,20),(30,30)]
    #list2 = [(0,0), (2, 0)]
    #list3 = [(2, 0), (2, 2)]
    #list4 = [(2, 2), (0, 2)]
    #list5 = [(0, 2), (0, 0)]

    #data = [list1, list2, list3, list4, list5]

    # Create network graph
    G = nx.Graph()
    list_of_nodes = []
    current_id = 0
    point_iteration = 0
    bFoundSameNode = False


    for entity in data:
        connectivity_list = []
        for point in entity:
            # If we are beginning then list is empty (i.e. no nodes yet)
            if not list_of_nodes:
                list_of_nodes.append(Node(current_id, point))
                connectivity_list.append(current_id)
                current_id += 1
            else:
                bFoundSameNode = False
                for node in list_of_nodes:
                    # Check if the point in question is the same as a node
                    if point == node.coordinate:
                        connectivity_list.append(node.id)
                        bFoundSameNode = True
                        break
                if not bFoundSameNode:
                    list_of_nodes.append(Node(current_id, point))
                    connectivity_list.append(current_id)
                    current_id += 1
        for con_idx, connection in enumerate(connectivity_list[:-1]):
            for node in list_of_nodes:
                if node.id == connection:
                    node.connected_to.append(connectivity_list[con_idx+1])

    for node in list_of_nodes:
        G.add_node(node.id, pos=node.coordinate)
    for node in list_of_nodes:
        if not node.connected_to:
            pass
        else:
            for node_connection in node.connected_to:
                '''
                p0 = np.asarray(list_of_nodes[node.id].coordinate)
                p1 = np.asarray(list_of_nodes[node_connection].coordinate)
                weight = np.linalg.norm(p0 - p1)
                G.add_edge(node.id, node_connection, weight=weight)
                '''
                G.add_edge(node.id, node_connection)
    pos = nx.get_node_attributes(G, 'pos')
    w, h = figaspect(5 / 3)
    fig, ax = plt.subplots(figsize=(w, h))
    nx.draw(G, pos, node_size=20, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()

