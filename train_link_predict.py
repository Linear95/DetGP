import numpy as np
import tensorflow as tf 
import random
import scipy.cluster
from sklearn.model_selection import train_test_split

from model import DetGP 
from dataloader import DataLoader
from tools import Logger
import config
from evaluation import evaluate as svm_classify

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU_ID)

#load data
dataset_choice = config.dataset_choice
graph_path = './datasets/%s/graph.txt' % dataset_choice  # train_
text_path = './datasets/%s/data.txt' % dataset_choice
group_path = './datasets/%s/group.txt' % dataset_choice

model_name = 'DetGP'
log_save_path = './logs/'
log = Logger(log_save_path+model_name+'.txt')
data = DataLoader(text_path, graph_path, config.edge_split_ratio)

    
def eval_link_prediction(sess, model, model_name):
    text_emb, struct_emb = sess.run([model.text_emb_a, model.struct_emb_a], feed_dict = {
            # model.edges: data.edges,
            # model.text_all: data.text,
            model.node_a_ids: np.arange(data.num_nodes) } )
    node_emb = np.concatenate((text_emb,struct_emb), axis =1)
    seen_nodes = data.train_graph.nodes()
    
    auc_scores = []
    negative_samples = negative_sampling(data.graph, data.test_edges)
    for node_a, node_b, node_n in negative_samples:
        if node_a in seen_nodes and node_b in seen_nodes: 
            dot1 = np.dot(node_emb[node_a], node_emb[node_b])
            dot2 = np.dot(node_emb[node_a], node_emb[node_n])
            if dot1 > dot2:
                auc_scores.append(1.)
            elif dot1 == dot2:
                auc_scores.append(0.5)
            else:
                auc_scores.append(0.)
    return np.mean(auc_scores)

def get_initial_inducing(sess, model, inducing_num):
    text_feature = sess.run(model.text_feature)
    inducing_points = scipy.cluster.vq.kmeans2(text_feature, inducing_num, minit='points')[0]
    return inducing_points

# start session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.GPU_USAGE)
with tf.Graph().as_default():
    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():
        model = DetGP(data.num_vocab, data.num_nodes, data.text, data.train_edges)
        opt = tf.train.AdamOptimizer(config.lr)
        train_op = opt.minimize(model.total_loss)

        sess.run(tf.global_variables_initializer())
        inducing_points = get_initial_inducing(sess, model, config.inducing_num)
        model.load_inducing_points(inducing_points)
	    #training
        log.write('start training : {0}'.format(model_name))
        auc_best = 0.

        for epoch in range(config.num_epoch):
            loss_epoch=0
            batches=data.generate_batches()
            num_batch=len(batches)
            for i in range(num_batch):
                batch=batches[i]
                node1, node2, node3=zip(*batch)    
                node1, node2, node3=np.array(node1),np.array(node2),np.array(node3)
                feed_dict={
                    model.node_a_ids: node1,
                    model.node_b_ids: node2,
                    model.node_n_ids: node3}
                # run the model
                _, loss_batch, al  = sess.run([train_op,model.total_loss, model.al],feed_dict=feed_dict)
                loss_epoch += loss_batch
                    
            log.write('epoch: {0} loss: {1}'.format(epoch+1,loss_epoch))
            if (epoch+1) % config.eval_epoch_num == 0:
                auc = eval_link_prediction(sess, model, model_name)
                log.write('test AUC {0}'.format(auc))
 
