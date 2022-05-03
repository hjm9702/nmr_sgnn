import torch
import torch.nn as nn

from dgl.nn.pytorch import NNConv, Set2Set


class nmr_mpnn_BASELINE(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, readout_mode,
                 node_feats = 128,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 hidden_feats = 512, prob_dropout = 0.1):
        
        super(nmr_mpnn_BASELINE, self).__init__()

        self.readout_mode = readout_mode
        
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
        
        self.readout_g = Set2Set(input_dim = node_feats + node_in_feats,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)
                               
        self.readout_n = nn.Sequential(
            nn.Linear(node_feats * 3 + node_in_feats * 3, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.ReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )                           

        self.readout_n_naive = nn.Sequential(
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

        if self.readout_mode == 'baseline':
            out = self.readout_n_naive(node_embed_feats[masks])


        elif self.readout_mode == 'proposed':
            graph_embed_feats = self.readout_g(g, node_embed_feats)        
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

            out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[masks])

        return out[:,0]