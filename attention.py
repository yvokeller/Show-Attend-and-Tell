import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, embedding_size):
        super(Attention, self).__init__()
        self.U = nn.Linear(embedding_size, embedding_size)
        self.W = nn.Linear(encoder_dim, embedding_size)
        self.v = nn.Linear(embedding_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha
    