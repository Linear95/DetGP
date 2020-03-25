import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import random
import networkx as nx

from tensorflow.contrib import learn
from tools import edges_to_undirect

import config

class DataLoader():
    def __init__(self, text_path, graph_path, edge_split_ratio):

        self.graph, self.train_graph, self.test_graph = self.load_graph(graph_path, edge_split_ratio)
        self.text, self.num_vocab, self.num_nodes = self.load_text(text_path)
        self.train_edges = edges_to_undirect(self.num_nodes, self.train_graph.edges())
        self.test_edges = edges_to_undirect(self.num_nodes, self.test_graph.edges())

    
    def get_adj_list(self,edges, node_num):
        adj_list = []
        for node in range(node_num):
                nbr = [node]
                for edge in edges:
                        if node in edge:
                                nbr += edge
                adj_list.append(list(set(nbr)))
        return adj_list


    def load_graph(self, graph_path, edge_split_ratio):
        total_graph = nx.Graph()
        train_graph = nx.Graph()
        test_graph = nx.Graph()
        np.random.seed(config.random_seed)
        graph_file = open(graph_path, 'rb').readlines()
        for line in graph_file:
            # print(line)
            edge = map(int, line.strip().split('\t'))
            total_graph.add_edge(edge[0],edge[1])
            
            if np.random.uniform(0.0, 1.0) <= edge_split_ratio:
                train_graph.add_edge(edge[0],edge[1])
            else:
                test_graph.add_edge(edge[0],edge[1])
                
        return total_graph, train_graph, test_graph

    
    def load_text(self, text_path):
        text_file = open(text_path, 'rb').readlines()
        vocab = learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
        text = np.array(list(vocab.fit_transform(text_file)))
        num_vocab = len(vocab.vocabulary_)
        num_nodes = len(text_file)
        return text, num_vocab, num_nodes

    
    def subgraph_edges(self, node_list):
        subg_edges = self.train_graph.subgraph(node_list).edges()
        return edges_to_undirect(self.num_nodes, subg_edges)
    
    def negative_sampling(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        #np.random.seed(config.random_seed)
        for i in range(len(edges)):
            neg_node = np.random.randint(self.num_nodes)
            while neg_node in self.graph.neighbors(node1[i]):  # total or train?
                # warning: can not deal with fully connected node
                neg_node = np.random.randint(self.num_nodes)
            sample_edges.append([node1[i], node2[i], neg_node])
        return sample_edges

    
    def generate_batches(self, mode=None):
        num_batch = len(self.train_edges)//config.batch_size
        edges = self.train_edges
        random.shuffle(edges)
        sample_edges = edges[:num_batch*config.batch_size]
        sample_edges = self.negative_sampling(sample_edges)
        batches = [sample_edges[i*config.batch_size:(i+1)*config.batch_size] for i in range(num_batch) ]
        return batches
