import numpy as np 
import tensorflow as tf 
import config
import pdb
from tensorflow.contrib import layers
from tensorflow.sparse import sparse_dense_matmul as sd_matmul



def RBF_Kernel(feature_a, feature_b, gamma = 1000.):
    f_a = tf.expand_dims(feature_a, 1)  #[node_num_a, 1, f_dim]
    f_b = tf.expand_dims(feature_b, 0)  #[1, node_num_b, f_dim]
    k_ab = tf.reduce_sum(tf.square(f_a-f_b), axis = 2)
    return tf.exp(-k_ab / gamma)

def Linear_Kernel(feature_a, feature_b):
    k_ab = tf.matmul(feature_a, feature_b, transpose_b = True)
    return k_ab + 1.


def Polynomial_Kernel(feature_a, feature_b):
    return Linear_Kernel(feature_a, feature_b)**2.


def regularized_adj(adj):
    return adj/tf.sparse.reduce_sum(adj, axis = 1, keepdims = True)


def get_transition_matrix(Adj): # a sparse adjacency matrix
    reg_Adj = tf.sparse.add(Adj, tf.sparse.eye(tf.shape(Adj)[0]))
    return reg_Adj/tf.sparse.reduce_sum(reg_Adj,axis = 1, keepdims = True)


def norm_trans(adj_mat):
    return adj_mat / tf.sparse.reduce_sum(adj_mat, axis = 1, keepdims = True)
    

class DetGP():
    def __init__(self, vocab_size, num_nodes, text_data, edges):
        self.num_nodes = num_nodes
        self.kernel = Linear_Kernel
        self.inducing_num = config.inducing_num
        self.whiten = False
        self.embedding_dim = config.embed_size/2
        self.trans_order = config.trans_order

        with tf.name_scope('read_inputs') as scope:
            self.text_all = tf.constant(text_data, dtype = tf.int32)
            self.node_a_ids = tf.placeholder(tf.int32, [None], name = 'a_ids')
            self.node_b_ids = tf.placeholder(tf.int32, [None], name = 'b_ids')
            self.node_n_ids = tf.placeholder(tf.int32, [None], name = 'n_ids')

            self.edges = tf.constant(edges, dtype = tf.int64)
            self.adj_mat = tf.sparse.SparseTensor(
                            self.edges, tf.ones(tf.shape(self.edges)[0]),
                            dense_shape=[self.num_nodes, self.num_nodes])
            self.trans_mat = get_transition_matrix(self.adj_mat)
            #self.initial_inducing_points = tf.placeholder(tf.float32, [None, config.embed_size/2], name = 'inducing')

        with tf.variable_scope('ggp') as scope:
            self.inducing_points = tf.get_variable(name = 'inducing_points', shape = [self.inducing_num, self.embedding_dim])
            self.q_mu = tf.get_variable(name = 'embedding_mu', shape = [self.inducing_num, self.embedding_dim], initializer = tf.initializers.constant(0.1))
            self.alpha = tf.get_variable(name = 'alpha', shape = [config.trans_order+1], initializer = tf.initializers.constant(0.))
            self.al = tf.nn.softmax(self.alpha)

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed=tf.Variable(tf.truncated_normal([vocab_size,config.embed_size/2],stddev=0.3),name = 'word_embedding')
            # self.node_embed=tf.Variable(tf.truncated_normal([num_nodes,config.embed_size/2],stddev=0.3))
            # self.node_embed=tf.clip_by_norm(self.node_embed,clip_norm=1,axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.text_emb_lookup = tf.nn.embedding_lookup(self.text_embed, self.text_all)
            self.text_feature = tf.reduce_mean(self.text_emb_lookup, axis = 1)                #

        self._build_model()


    def load_inducing_points(self, inducing_points):
        self.inducing_points = tf.assign(self.inducing_points, inducing_points)
        
        
    def ggp_proces (self, features, z_features):
        delta = 0.01
        Kxz = self.kernel(features, z_features)
        Kzz = self.kernel(z_features, z_features) +tf.eye(self.inducing_num) * delta
        Kzz = tf.stop_gradient(Kzz)

        if self.trans_order == 3:
            Kxz_0 = Kxz
            Kxz_1 = sd_matmul(self.trans_mat, Kxz)
            Kxz_2 = sd_matmul(self.trans_mat, sd_matmul(self.trans_mat, Kxz))
            Kxz_3 = sd_matmul(self.trans_mat, Kxz_2)
            Kxz = self.al[0]*Kxz_0 + self.al[1]*Kxz_1 + self.al[2] * Kxz_2 + self.al[3]*Kxz_3
            
        Lz = tf.cholesky(Kzz)
        # Compute the projection matrix A
        A = tf.matrix_triangular_solve(Lz, tf.transpose(Kxz), lower=True)
        if not self.whiten:
            A = tf.matrix_triangular_solve(tf.transpose(Lz), A, lower=False)
        # construct the conditional mean
        fmean = tf.matmul(A, self.q_mu, transpose_a=True)        
        return fmean


    def compute_loss(self, emb_a, emb_b, emb_n):
        positive_loss = tf.reduce_sum(emb_a * emb_b, axis = 1)
        negative_loss = -tf.reduce_sum(emb_a * emb_n, axis = 1)
        p_likeli = positive_loss - tf.math.softplus(positive_loss)     #log(sigmoid(x))
        n_likeli = negative_loss - tf.math.softplus(negative_loss)

        total_loss = -tf.reduce_mean(p_likeli + n_likeli)
        return total_loss


    def get_distance(self, feature, ind_points):
        feature_prime = tf.expand_dims(feature, axis = 1)
        ind_points_prime = tf.expand_dims(ind_points, axis = 0)
        dist = tf.reduce_mean(tf.square(feature_prime - ind_points_prime), axis = 2) #[feature_num, induc_num]
        return tf.reduce_max(tf.sqrt(dist))

    
    def dmte_embeddings(self, text_features):
        first_order_emb = text_features
        second_orde_emb = sd_matmul(self.trans_mat,self.text_feature)

        return first_order_emb + second_orde_emb

    
    def _build_model(self):
        self.text_emb = self.dmte_embeddings(self.text_feature)
        self.struct_emb = self.ggp_proces(self.text_emb, self.inducing_points)

        self.text_emb_a = tf.nn.embedding_lookup(self.text_emb, self.node_a_ids)
        self.text_emb_b = tf.nn.embedding_lookup(self.text_emb, self.node_b_ids)
        self.text_emb_n = tf.nn.embedding_lookup(self.text_emb, self.node_n_ids)
        self.text_loss = self.compute_loss(self.text_emb_a, self.text_emb_b, self.text_emb_n)

        self.struct_emb_a = tf.nn.embedding_lookup(self.struct_emb, self.node_a_ids)
        self.struct_emb_b = tf.nn.embedding_lookup(self.struct_emb, self.node_b_ids)
        self.struct_emb_n = tf.nn.embedding_lookup(self.struct_emb, self.node_n_ids)
        self.struct_loss = self.compute_loss(self.struct_emb_a, self.struct_emb_b, self.struct_emb_n)
        
        self.total_loss = self.text_loss + 0.3 * self.struct_loss

