import torch
from torch import nn
import torch.nn.functional as F
import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMTagger(nn.Module):
    def __init__(self,vocab_size, target_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = 256
        self.num_layers = 1
        self.embedding_dim = 256
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim = self.embedding_dim).to(device)
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
        )
        self.hidden2tag = nn.Linear(4*self.hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).to(device)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out.reshape(len(sentence), -1)).to(device)
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 128, self.hidden_dim).to(device),
                    torch.zeros(self.num_layers, 128,self.hidden_dim).to(device))
