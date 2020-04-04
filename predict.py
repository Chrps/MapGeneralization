import numpy as np
import torch
import time
import src.graph_utils as graph_utils
import src.models as models
import matplotlib.pyplot as plt
import os
import networkx as nx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--predict-path', type=str, default='data/predict_file_list.txt')
parser.add_argument('--model_name', type=str, default='test_model')

args = parser.parse_args()

def load_model_txt(model_name):
    model_txt = 'models/' + model_name + '/' + model_name + '.txt'
    data = [line.rstrip() for line in open(model_txt)]

    # network train on ()
    net = data[0]

    # Number of features per node
    n_features = int(data[1])

    # Number of classes
    n_classes = int(data[2])

    return net, n_features, n_classes

def draw(results, ax, nx_G, positions):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'

    colors = []
    for v in range(len(nx_G)):
        cls = results[v]
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Results')
    nx.draw_networkx(nx_G.to_undirected(), positions, node_color=colors,
            with_labels=False, node_size=25, ax=ax)

def predict(predict_path, model_name):
    # Read the parameters of the trained model
    net, n_features, n_classes = load_model_txt(model_name)

    # Load the trained model
    trained_net, config = models.get_model_and_config(net)
    model = trained_net(n_features,
                n_classes,
                *config['extra_args'])
    model_path = 'models/' + model_name + '/' + model_name + '.pth'
    model.load_state_dict(torch.load(model_path))
    print(model)

    # Get the list of files for prediction
    data_path = 'data/'
    folder = 'graph_annotations'
    pred_files = [os.path.join(data_path, folder, line.rstrip()) for line in open(predict_path)]
    for file in pred_files:
        # Convert the gpickle file to a dgl graph
        dgl_g = graph_utils.convert_gpickle_to_dgl_graph(file)
        # Get the features from the given graph
        features = graph_utils.get_features(file)

        model.eval()
        with torch.no_grad():
            logits = model(dgl_g, features)
            _, predictions = torch.max(logits, dim=1)
            predictions = predictions.numpy()

        # % Plot the results
        # Get positions
        nxg = nx.read_gpickle(file)
        positions = nx.get_node_attributes(nxg, 'pos')
        positions = list(positions.values())
        # Plot
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        draw(predictions, ax, nxg, positions)  # draw the results

        plt.show()


if __name__ == '__main__':
    predict_path = args.predict_path
    model_name = args.model_name

    predict(predict_path, model_name)