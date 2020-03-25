def edges_to_undirect(node_num, edges):
    # when given edge [i,j] build undirect graph with edge [i,j] and [j,i], each edge only appear once.
    edge_hash = []
    for edge in edges:
        edge_hash.append(edge[0]*node_num+edge[1])
        edge_hash.append(edge[1]*node_num+edge[0])
    edge_hash = list(set(edge_hash))
    ud_edges = [[v // node_num, v % node_num] for v in edge_hash]
    return ud_edges


class Logger():
    def __init__(self, log_file_path):
        self.log_lines = []
        self.path = log_file_path
        
    def __del__(self):
        with open(self.path, 'w') as f:
            for line in self.log_lines:
                f.write("%s\n" % line)

    def write(self, text):
        self.log_lines.append(text)
        print(self.log_lines[-1])

    
