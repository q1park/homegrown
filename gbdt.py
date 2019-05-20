import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from graphviz import dot

maxloss = 1.0e15

## Quantum Relative Entropy
def qrelative_entropy(y, output):
    assert y.shape == output.shape
    eps = 1e-16
    y = np.piecewise(y,[y<eps, y>=eps], [lambda x: eps, lambda x: x])
    
    if output is None:
        output = np.random.uniform(eps, 1-eps, size = y.shape)
    else:
        output = np.piecewise(output,[output<eps, output>=eps], [lambda x: eps, lambda x: x])
    
    S = (1/len(y))*np.sum([y[i]*np.log(y[i]/output[i]) for i in range(len(y))])
    dS = -(1/len(y))*np.array([y[i]/output[i] for i in range(len(y))]).reshape(len(y), 1)
    ddS = (1/len(y))*np.array([y[i]/output[i]**2 for i in range(len(y))]).reshape(len(y), 1)
    
    return S, dS, ddS

## Mean-squared Error
def mse(y, output):
    assert y.shape == output.shape
    ### minus sign in ddl
    l = np.array([(y[i] - output[i])**2 for i in range(len(output))])
    dl = np.array([2 * (y[i] - output[i]) for i in range(len(output))])
    ddl = np.full(len(output), 2)
    
    return l, dl, ddl

class decisiontree(nx.DiGraph):
    def __init__(self):
        super(decisiontree, self).__init__()
        self.add_node('root', nodetype = 'fork', feature = None, splitval = None, weight = None)
        
    def listnodes(self):
        return self.nodes
        
    def _splitnode(self, node):
        if node is 'root':
            lname, rname = 'L', 'R'
        else:
            lname, rname = node + 'L', node + 'R'
        self.add_node(lname, nodetype = None, feature = None, splitval = None, weight = None)
        self.add_node(rname, nodetype = None, feature = None, splitval = None, weight = None)
        self.add_edge(node, lname)
        self.add_edge(node, rname)
        
    def _lossredux(self, dlsum, ddlsum, dlsum_L, ddlsum_L, dlsum_R, ddlsum_R, lamb):
        term1 = np.square(dlsum_L) / (ddlsum_L + lamb)
        term2 = np.square(dlsum_R) / (ddlsum_R + lamb)
        term3 = np.square(dlsum) / (ddlsum + lamb)
        return term1 + term2 - term3
    
    def _leafweight(self, dl, ddl, lamb):
        return np.sum(dl) / (np.sum(ddl) + lamb)
        
    def build(self, node, X, dl, ddl, shrinkrate, depth, params):
        assert len(list(self.successors(node) ) ) == 0
        
        if depth > params['maxdepth']:
            self.nodes[node]['nodetype'] = 'leaf'
            self.nodes[node]['weight'] = self._leafweight(
                dl, ddl, params['lambda'])*shrinkrate
            return
            
        dlsum, ddlsum = np.sum(dl), np.sum(ddl)
        
        best_lossredux = 0.
        best_splitval = 0.
        best_feature = None
        best_idx_L, best_idx_R = None, None
        
        for feature in X.columns:
            dlsum_L, ddlsum_L = 0., 0.
            sorted_idx = X[feature].argsort()

            for i, idx in enumerate(sorted_idx):
                dlsum_L += dl[idx]
                ddlsum_L += ddl[idx]
                dlsum_R, ddlsum_R = dlsum - dlsum_L, ddlsum - ddlsum_L
                
                lossredux = self._lossredux(dlsum, ddlsum, dlsum_L, ddlsum_L, \
                                            dlsum_R, ddlsum_R, params['lambda'])

                if lossredux > best_lossredux:
                    best_lossredux = lossredux
                    best_feature = feature
                    best_splitval = X[feature].iloc[idx]
                    best_idx_L = sorted_idx[:i+1]
                    best_idx_R = sorted_idx[i+1:]

        if best_lossredux < params['min_lossredux']:
            self.nodes[node]['nodetype'] = 'leaf'
            self.nodes[node]['weight'] = self._leafweight(
                dl, ddl, params['lambda'])*shrinkrate
        else:
            self.nodes[node]['nodetype'] = 'fork'
            self.nodes[node]['splitval'] = best_splitval
            self.nodes[node]['feature'] = best_feature
            
            self._splitnode(node)
    
            self.build(list(self.successors(node) )[0], X.iloc[best_idx_L], \
                       dl[best_idx_L], ddl[best_idx_L], shrinkrate, \
                       depth + 1, params)
            self.build(list(self.successors(node) )[1], X.iloc[best_idx_R], \
                       dl[best_idx_R], ddl[best_idx_R], shrinkrate, \
                       depth + 1, params)
            
    def run(self, x, node):
        if self.nodes[node]['nodetype'] is 'leaf':
            return self.nodes[node]['weight']
        else:
            if x[self.nodes[node]['feature']] <= self.nodes[node]['splitval']:
                return self.run(x, list(self.successors(node) )[0])
            else:
                return self.run(x, list(self.successors(node) )[1])
            

class xgbdt:
    def __init__(self, params):
        self.params = params
        self.trees = list()
        self.best_run = None

    def _newtree(self, Xtrain, dl, ddl, shrinkrate):
        newtree = decisiontree()
        newtree.build('root', Xtrain, dl, ddl, shrinkrate, 0, self.params)
        return newtree
    
    def _scores(self, X):
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self._predict(X.iloc[i])
        return scores
    
    def _predict(self, x):
        assert len(self.trees) > 0
        return np.sum([tree.run(x, 'root') for tree in self.trees])

    def _loss(self, y, output, losstype = 'mse'):
        y = y.values.reshape(output.shape)
        if losstype is 'mse':
            l, dl, dll = mse(y, output)
        return l, dl, dll

    def cleartrees(self):
        self.trees.clear()
    
    def treeplot(self, i):
        assert i <= len(self.trees) - 1
        plt.title('draw_networkx')
        pos=graphviz_layout(self.trees[i], prog='dot')
        return nx.draw(self.trees[i], pos, with_labels=True, arrows=True, \
                       node_size=1000, node_color='gray')
    
    def train(self, train_set, valid_set, features, labels, nboost):
        print('Starting training')
        start_time = time.time()

        shrinkrate = self.params['shrinkrate']
        v_bestloss = maxloss
        best_run = None
        
        Xtrain, Ytrain = train_set[features], train_set[labels]
        Xvalid, Yvalid = valid_set[features], valid_set[labels]
        
        for iboost in range(nboost):
            boost_time = time.time()
            
            if iboost == 0:
#                 rand = np.random.RandomState(1234)
                dloss = np.random.uniform(size = train_set.shape[0])
                ddloss = np.full(dloss.shape[0], 2)
                
            newtree = self._newtree(Xtrain, dloss, ddloss, shrinkrate)
            self.trees.append(newtree)
            
            if iboost > 0:
                shrinkrate *= self.params['learnrate']
            
            scores = self._scores(train_set)
            loss, dloss, ddloss = self._loss(Ytrain, scores)
            
            v_scores = self._scores(valid_set)
            v_loss, _, _ = self._loss(Yvalid, v_scores)
            
            loss = np.mean(loss)
            v_loss = np.mean(v_loss)
            
            print("Iter {:>3}, Train Loss: {:.10f}, Valid Loss: {:.5f}, Time: {:.2f} secs"
                  .format(iboost, loss, v_loss, time.time() - boost_time))
            
            if v_loss < v_bestloss and v_bestloss - v_loss >= self.params['earlystop_precision']:
                v_bestloss = v_loss
                best_run = iboost
                
            if best_run is not None and iboost - best_run >= self.params['earlystop_rounds']:
                print("Stopped early with best run:")
                print("Iter {:>3}, Valid Loss: {:.10f}".format(best_run, v_bestloss))
                break

        self.best_run = best_run
        print("Training finished after {:.2f} secs".format(time.time() - start_time))