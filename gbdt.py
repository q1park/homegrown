import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from graphviz import dot

maxdepth = 3

class decisiontree(nx.DiGraph):
    def __init__(self):
        super(decisiontree, self).__init__()
        self.add_node('root', nodetype = 'fork', feature = None, splitval = None, weight = None)
        
    def _splitnode(self, node):
        if node is 'root':
            lname, rname = 'l', 'r'
        else:
            lname, rname = node + 'l', node + 'r'
        self.add_node(lname, nodetype = None, feature = None, splitval = None, weight = None)
        self.add_node(rname, nodetype = None, feature = None, splitval = None, weight = None)
        self.add_edge(node, lname)
        self.add_edge(node, rname)
        
    def listnodes(self):
        return self.nodes
        
    def build(self, node, depth):
        assert len(list(self.successors(node) ) ) == 0
        
        if depth > maxdepth:
            self.nodes[node]['nodetype'] = 'leaf'
            self.nodes[node]['weight'] = np.random.uniform(-1, 0)
            return
        else:
            cointoss = np.random.uniform(0, 1)

            if cointoss > 0.8:
                self.nodes[node]['nodetype'] = 'leaf'
                self.nodes[node]['weight'] = np.random.uniform(-1, 0)
            else:
                self.nodes[node]['nodetype'] = 'fork'
                self.nodes[node]['splitval'] = np.random.uniform(0, 1)
                self.nodes[node]['feature'] = np.random.randint(0,8)
                self._splitnode(node)
                self.build(list(self.successors(node) )[0], depth = depth + 1)
                self.build(list(self.successors(node) )[1], depth = depth + 1 )
            
    def run(self, x, node):
        if self.nodes[node]['nodetype'] is 'leaf':
            return self.nodes[node]['weight']
        else:
            if x[self.nodes[node]['feature']] <= self.nodes[node]['splitval']:
                return self.run(x, list(self.successors(node) )[0])
            else:
                return self.run(x, list(self.successors(node) )[1])
    
