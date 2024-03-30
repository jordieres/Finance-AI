import torch
from torch import nn
from tqdm import tqdm

from EncoderTransformer import TransformerEncoder as Encoder
from ModelTransformer.DecoderTransformer import TransformerDecoder as Decoder


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_model: int,
            num_attention_heads: int,
            units_hidden_layer: int,
            dropout: float,
            n_ftrs: int,
            activation: nn.Module = nn.ReLU(),
            mask: bool = True):

        super().__init__()

        self.encoder = Encoder (
            num_layers=num_encoder_layers,
            d_model=dim_model,
            num_heads=num_attention_heads,
            units_hidden_layer=units_hidden_layer,
            dropout=dropout,
            mask=mask,
            activation=activation)

        self.decoder = Decoder (
            num_layers=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_attention_heads,
            units_hidden_layer=units_hidden_layer,
            dropout=dropout,
            activation=activation,
            mask=mask)
        
        self.output_layer = nn.Linear(dim_model, n_ftrs) # Linear layer for predicting the single feature

    def forward(self, source, target):
        outputs = []

        for src, tgt_seq in tqdm(zip(source, target), total=len(source)):
            src = torch.unsqueeze(src, dim=0)
            encoder_output = self.encoder(src)
            batch_output = []
            
            for tgt_step in tgt_seq:
                tgt_step = torch.unsqueeze(tgt_step, dim=0)
                decoder_output = self.decoder(src=tgt_step, memory=encoder_output)
                batch_output.append(decoder_output)

            # Concatenate along the time dimension and apply the linear layer
            output = torch.cat(batch_output, dim=0)
            output = self.output_layer(output)
            outputs.append(output)

        return torch.cat(outputs, dim=0)