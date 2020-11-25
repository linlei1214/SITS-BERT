import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class SBERT(nn.Module):

    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-BERT model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(num_features, int(hidden/2))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, doy, mask):
        mask = (mask > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
