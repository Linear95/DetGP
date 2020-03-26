MAX_LEN=300
neg_table_size = 1000000
NEG_SAMPLE_POWER = 0.75

batch_size = 128
num_epoch = 500

report_epoch_num = 1
eval_epoch_num = 10 * report_epoch_num

GPU_USAGE = 0.5
GPU_ID = 0

embed_size = 200
lr = 1e-3

dataset_choice = 'cora' #'zhihu' 'cora' 'DBLP'

node_split_ratio = 0.5    # proportion of nodes
edge_split_ratio = 0.35   # proportion of edgesused in training

random_seed = 1855

if dataset_choice == 'cora':
    inducing_num = 20
elif dataset_choice == 'zhihu':
    inducing_num = 100
elif dataset_choice == 'DBLP':
    inducing_num = 300
elif dataset_choice == 'HepTh':
    inducing_num = 20

    
encoder_type = 'wavg'  # 'dmte' 'wavg'
kernel_type = 'linear'
trans_order = 3
