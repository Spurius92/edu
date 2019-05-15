class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        hidden_size = 64
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.embedding_dropout = nn.Dropout2d(0.2) 
        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(hidden_size*2, maxlen)
        
        self.out = nn.Linear(384, 1)

        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding.transpose(1,2).unsqueeze(-1)).squeeze().transpose(1,2)

        h_lstm, _ = self.lstm(h_embedding)
        h_lstm_atten = self.lstm_attention(h_lstm)

        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        
        conc = torch.cat((h_lstm_atten, avg_pool, max_pool), 1)
        out = self.out(conc)
        
        return out