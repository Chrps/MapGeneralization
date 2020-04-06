import random
from io import open
from src.deepwalk import graph, walks as serialized_walks
from gensim.models import Word2Vec
from src.deepwalk.skipgram import Skipgram
import os
import numpy as np
import torch


class DeepWalk:

    def __init__(self, number_walks=4, walk_length=5, representation_size=2):
        self.undirected = True  # Treat graph as undirected.
        self.number_walks = number_walks  # Number of random walks to start at each node
        self.walk_length = walk_length  # Length of the random walk started at each node
        self.max_memory_data_size = 1000000000  # Size to start dumping walks to disk, instead of keeping them in memory
        self.seed = 0  # Seed for random walk generator
        self.representation_size = representation_size  # Number of latent dimensions to learn for each node
        self.window_size = 10  # Window size of skipgram model
        self.workers = 1  # Number of parallel processes
        self.vertex_freq_degree = False  # Use vertex degree to estimate the frequency of nodes in the random walks. This option is faster than calculating the vocabulary


    def create_embeddings(self, nx_graph, file_path):
        print('---EXECUTING DEEPWALK---')
        print('Creating embedding for: ' + file_path)

        # Where and what the embedding file will be saved
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output = 'data/embeddings/' + file_name + '_nw_' + str(self.number_walks) + '_wl_' + str(self.walk_length) + '_rs_' + str(self.representation_size)

        # If the embedding file already exists, retrieve it
        if os.path.exists(output):
            print('Embedding already exists.')
            pass
        else:
            G = graph.from_networkx(nx_graph, undirected=self.undirected)

            print("Number of nodes: {}".format(len(G.nodes())))

            num_walks = len(G.nodes()) * self.number_walks

            print("Number of walks: {}".format(num_walks))

            data_size = num_walks * self.walk_length

            print("Data size (walks*length): {}".format(data_size))

            if data_size < self.max_memory_data_size:
                print("Walking...")
                walks = graph.build_deepwalk_corpus(G, num_paths=self.number_walks,
                                                    path_length=self.walk_length, alpha=0, rand=random.Random(self.seed))
                print("Training...")
                model = Word2Vec(walks, size=self.representation_size, window=self.window_size, min_count=0, sg=1, hs=1,
                                 workers=self.workers)
            else:
                print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
                    data_size,
                    self.max_memory_data_size))
                print("Walking...")

                walks_filebase = output + ".walks"
                walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=self.number_walks,
                                                                  path_length=self.walk_length, alpha=0,
                                                                  rand=random.Random(self.seed),
                                                                  num_workers=self.workers)

                print("Counting vertex frequency...")
                if not self.vertex_freq_degree:
                    vertex_counts = serialized_walks.count_textfiles(walk_files, self.workers)
                else:
                    # use degree distribution for frequency in tree
                    vertex_counts = G.degree(nodes=G.iterkeys())

                print("Training DeepWalk...")
                walks_corpus = serialized_walks.WalksCorpus(walk_files)
                model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                                 size=self.representation_size,
                                 window=self.window_size, min_count=0, trim_rule=None, workers=self.workers)

            model.wv.save_word2vec_format(output)

    def read_embeddings(self, file_path):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file = 'data/embeddings/' + file_name + '_nw_' + str(self.number_walks) + '_wl_' + str(self.walk_length) + '_rs_' + str(self.representation_size)
        with open(file) as f:
            list_string_feat = f.readlines()
            list_string_feat.pop(0)  # First line is not used
            embedding_feat = []
            for string_node_feat in list_string_feat:
                node_feat = np.fromstring(string_node_feat, dtype=float, sep=' ')
                embedding_feat.append(node_feat)
            embedding_feat = sorted(embedding_feat, key=lambda x: x[0])
            for idx, row in enumerate(embedding_feat):
                embedding_feat[idx] = np.delete(row, 0)
            all_embedding_feats = np.concatenate(embedding_feat, axis=0)
            max_emb = all_embedding_feats.max()
            min_emb = all_embedding_feats.min()
            for idx_l, list in enumerate(embedding_feat):
                for idx_e, element in enumerate(list):
                    embedding_feat[idx_l][idx_e] = (element + abs(min_emb)) / (max_emb + abs(min_emb))

            embedding_feat = torch.FloatTensor(embedding_feat)

        return embedding_feat

