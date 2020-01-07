import numpy as np
import torch
from torch import nn, cuda
from torch.nn import functional as F


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        lstm_hidden_size = 120
        gru_hidden_size = 60
        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(gru_hidden_size * 6, 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(20, 1)
        
    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out

# max_features = 100000
# embed_size = 200

# class SpatialDropout(nn.Dropout2d):
#     def forward(self, x):
#         x = x.unsqueeze(2)    # (N, T, 1, K)
#         x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
#         x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
#         x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
#         x = x.squeeze(2)  # (N, T, K)
#         return x
    
# class NeuralNet(nn.Module):
#     def __init__(self, embedding_matrix):
#         super(NeuralNet, self).__init__()
#         embed_size = embedding_matrix.shape[1]
        
#         self.embedding = nn.Embedding(max_features, embed_size)
#         self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
#         self.embedding.weight.requires_grad = False
#         self.embedding_dropout = SpatialDropout(0.3)
        
#         self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
            
#         self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
#         self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
#         self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, lengths=None):
#         h_embedding = self.embedding(x.long())
#         h_embedding = self.embedding_dropout(h_embedding)
        
#         h_lstm1, _ = self.lstm1(h_embedding)
#         h_lstm2, _ = self.lstm2(h_lstm1)
        
#         # global average pooling
#         avg_pool = torch.mean(h_lstm2, 1)
#         # global max pooling
#         max_pool, _ = torch.max(h_lstm2, 1)
        
#         h_conc = torch.cat((max_pool, avg_pool), 1)
#         h_conc_linear1  = F.relu(self.linear1(h_conc))
#         h_conc_linear2  = F.relu(self.linear2(h_conc))
                
# #         hidden = h_conc + h_conc_linear1 + h_conc_linear2 + normal_linear
#         hidden = h_conc + h_conc_linear1 + h_conc_linear2 

#         hidden = self.dropout(hidden)
#         result = self.linear_out(hidden)
        
#         return result