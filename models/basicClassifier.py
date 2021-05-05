from torch import nn


class BasicTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super(BasicTextClassifier, self).__init__()
        # Gets mean(/sum/max) of embeddings for each text position
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                         sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        """Weight initialization, not essential, but better than random distribution"""
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offset):
        embed = self.embedding(text, offset)      # Get text embedding
        return self.fc(embed)                     # Embedding passed to FC layer
