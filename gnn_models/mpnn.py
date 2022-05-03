import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dgl.nn.pytorch import NNConv, Set2Set

from util import MC_dropout
from sklearn.metrics import mean_absolute_error


class nmrMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, embed_mode,
                 node_feats = 128,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 hidden_feats = 512, prob_dropout = 0.1):
        
        super(nmrMPNN, self).__init__()

        self.embed_mode = embed_mode
        
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_feats), nn.ReLU(),
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, node_feats * node_feats)
        )
        
        self.gnn_layer = NNConv(
            in_feats = node_feats,
            out_feats = node_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.gru = nn.GRU(node_feats, node_feats)
        
        self.readout = Set2Set(input_dim = node_feats + node_in_feats,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)
                               
        self.predict = nn.Sequential(
            nn.Linear(node_feats * 3 + node_in_feats * 3, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )                           

        self.predict_naive = nn.Sequential(
            nn.Linear(node_feats + node_in_feats, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )                      

    def forward(self, g, n_nodes, masks):
        
        def embed(g):
            
            node_feats_org = g.ndata['node_attr']
            edge_feats = g.edata['edge_attr']
            
            node_feats = self.project_node_feats(node_feats_org)
            hidden_feats = node_feats.unsqueeze(0)
            
            node_aggr = [node_feats_org]
            for _ in range(self.num_step_message_passing):
                node_feats = torch.relu(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
                node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
                node_feats = node_feats.squeeze(0)
                
            node_aggr.append(node_feats)
            node_aggr = torch.cat(node_aggr, 1)
            
            return node_aggr

        node_embed_feats = embed(g)

        if self.embed_mode == 'naive':
            out = self.predict_naive(node_embed_feats[masks])


        elif self.embed_mode == 'gconcat':
            graph_embed_feats = self.readout(g, node_embed_feats)        
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

            out = self.predict(torch.hstack([node_embed_feats, graph_embed_feats])[masks])

        return out[:,0]