import matplotlib.pyplot as plt
import networkx as nx
import random


def generate_graph():
    G = nx.Graph()

    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0),
             (3, 1), (3, 2), (3, 3)]
    positions = [list(pos) for pos in positions]

    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    edges = [(0, 4), (0, 1), (1, 5), (1, 2), (2, 6), (2, 3),
             (3, 7), (4, 8), (4, 5), (5, 9), (5, 6), (6, 10),
             (6, 7), (7, 11), (8, 12), (8, 9), (9, 13), (9, 10),
             (10, 14), (10, 11), (11, 15), (12, 13), (13, 14), (14, 15)]


    rand_int_1_x = random.randint(0, 2)
    rand_int_2_x = random.randint(2, 4)
    rand_int_3_x = random.randint(4, 6)
    rand_int_4_x = random.randint(6, 8)

    rand_int_1_y = random.randint(0, 2)
    rand_int_2_y = random.randint(2, 4)
    rand_int_3_y = random.randint(4, 6)
    rand_int_4_y = random.randint(6, 8)

    for node in nodes:
        # Add some randomness to x coordinates
        if positions[node][0] == 0:
            positions[node][0] += rand_int_1_x
        elif positions[node][0] == 1:
            positions[node][0] += rand_int_2_x
        elif positions[node][0] == 2:
            positions[node][0] += rand_int_3_x
        elif positions[node][0] == 3:
            positions[node][0] += rand_int_4_x
        # Add some randomness to y coordinates
        if positions[node][1] == 0:
            positions[node][1] += rand_int_1_y
        elif positions[node][1] == 1:
            positions[node][1] += rand_int_2_y
        elif positions[node][1] == 2:
            positions[node][1] += rand_int_3_y
        elif positions[node][1] == 3:
            positions[node][1] += rand_int_4_y
        G.add_node(node, pos=tuple(positions[node]), label=0)

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    #num_diamonds = random.randint(2, 5)
    num_diamonds = 4
    # Get a random edge
    idxs = random.sample(range(0, (len(edges)-1)), num_diamonds)

    counter = 0
    for idx in idxs:
        G.remove_edge(edges[idx][0], edges[idx][1])

        # Drawing diamond
        node1 = edges[idx][0]
        node2 = edges[idx][1]
        pos1 = positions[node1]
        pos2 = positions[node2]
        center_point = ((pos1[0]+pos2[0])/2, (pos1[1]+pos2[1])/2)

        # Randomize diamond
        x_ran = (random.randint(2, 4))/10
        y_ran = (random.randint(2, 4))/10

        # If same x coordinate draw diamond certain way
        if pos1[0] == pos2[0]:
            new_node_1 = tuple((center_point[0], center_point[1] - y_ran))
            new_node_2 = tuple((center_point[0], center_point[1] + y_ran))
            new_node_3 = tuple((center_point[0] - x_ran, center_point[1]))
            new_node_4 = tuple((center_point[0] + x_ran, center_point[1]))
        else:
            new_node_1 = tuple((center_point[0] - x_ran, center_point[1]))
            new_node_2 = tuple((center_point[0] + x_ran, center_point[1]))
            new_node_3 = tuple((center_point[0], center_point[1] - y_ran))
            new_node_4 = tuple((center_point[0], center_point[1] + y_ran))

        node_name_1 = 16 + counter
        node_name_2 = 17 + counter
        node_name_3 = 18 + counter
        node_name_4 = 19 + counter

        G.add_node(node_name_1, pos=new_node_1, label=1)
        G.add_node(node_name_2, pos=new_node_2, label=1)
        G.add_node(node_name_3, pos=new_node_3, label=1)
        G.add_node(node_name_4, pos=new_node_4, label=1)

        G.add_edge(node1, node_name_1)
        G.add_edge(node_name_1, node_name_3)
        G.add_edge(node_name_1, node_name_4)
        G.add_edge(node_name_4, node_name_2)
        G.add_edge(node_name_3, node_name_2)
        G.add_edge(node_name_2, node2)
        counter += 4

    pos = nx.get_node_attributes(G, 'pos')

    return G, pos
def main():
    NUM_GRAPHS_TRAIN = 5
    NUM_GRAPHS_TEST = 2

    train_file = open('../data/synth_graphs/train_file_list.txt', "w+")
    test_file = open('../data/synth_graphs/test_file_list.txt', "w+")

    name = 'graph_'
    for graph_idx in range(NUM_GRAPHS_TRAIN):
        G, pos = generate_graph()
        PATH = '../data/synth_graphs/training/' + name + str(graph_idx) + '.pickle'
        nx.write_gpickle(G, PATH)
        train_file.write(name + str(graph_idx) + '.pickle' + "\n")

    for graph_idx in range(NUM_GRAPHS_TEST):
        G, pos = generate_graph()
        PATH = '../data/synth_graphs/testing/' + name + str(graph_idx) + '.pickle'
        nx.write_gpickle(G, PATH)
        test_file.write(name + str(graph_idx) + '.pickle' + "\n")

    train_file.close()
    test_file.close()
    nx.draw(G, pos, node_size = 30)
    plt.show()
    print("done")

if __name__ == "__main__":
    main()
