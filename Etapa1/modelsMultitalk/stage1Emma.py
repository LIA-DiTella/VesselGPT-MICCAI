
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, GroupedResidualVQ, LFQ
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim] ← this is the key change
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, depth=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True, dropout=0.1)
            for _ in range(depth)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class VQAutoEncoder(nn.Module): 
    def __init__(self, args, input_dim=39, hidden_dim=512, codebook_size=256): #emma codebook size 2048 ---llevar hidden dim a 1024?
        super().__init__()

        # Input projection
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_enc = PositionalEncoding(dim=hidden_dim)

        self.encoder_transformer = TransformerBlock(dim=hidden_dim)

        self.vq = GroupedResidualVQ(
                                dim=hidden_dim,               # Total latent dimension (e.g., 512); input and output will match this shape
                                codebook_size=codebook_size,  # Number of entries in each codebook (per quantizer, per group); higher = more expressive
                                groups=4, #emma 32   pau64         # Split the input vector into 16 equal parts (e.g., 512 → eight 32D sub-vectors)#ANDUVO CON 8, PRUEBO CON 4
                                num_quantizers=4, #emma 4    # For each sub-vector (group), apply 4 quantizers sequentially (residual refinement)
                                commitment_weight=0.1,        # Weight for commitment loss: encourages encoder outputs to match selected codes
                                decay=0.97,                   # EMA decay for updating codebook entries; higher = slower update (more stable)
                                use_cosine_sim=False,         # Use L2 distance for nearest neighbor search (more stable than cosine for reconstruction)
                                rotation_trick=False          # Disable rotation trick (orthogonal transform); better stability and interpretability
                            )

        self.decoder_transformer = TransformerBlock(dim=hidden_dim)

        # Output projection
        self.decoder_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, blendshapes, mask=None):
        # blendshapes: [B, T, 58]
        # mask:        [B, T] with 1 for valid tokens, 0 for padded

        x = self.encoder_proj(blendshapes)

        x = self.pos_enc(x)

        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, _, vq_loss = self.vq(x)

        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)

        decoded = self.decoder_proj(x)

        #aca agregar sigmoide

        return decoded, vq_loss


    def get_quant(self, blendshapes, mask=None):
        x = self.encoder_proj(blendshapes)            # [B, T, hidden_dim]

        x = self.pos_enc(x)
        
        x = self.encoder_transformer(x, src_key_padding_mask=~mask if mask is not None else None)

        quantized, indices, _ = self.vq(x)            # indices: [B, T]

        return quantized, indices


    def decode(self, quantized, mask=None):

        x = self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)

        decoded = self.decoder_proj(x)

        return decoded
    
    
    
    def sample_step(self, x, mask=None):
        quantized, indices = self.get_quant(x, mask)

        x_sample_det = self.decoder_proj(
            self.decoder_transformer(quantized, src_key_padding_mask=~mask if mask is not None else None)
        )

        recovered_codes = self.vq.get_codes_from_indices(indices)  # [G, B, 1, T, D]

        # Remove the singleton dim (dim=2)
        recovered_codes = recovered_codes.squeeze(2)  # Now: [G, B, T, D]
        assert recovered_codes.dim() == 4, f"Expected 4D after squeezing, got {recovered_codes.shape}"

        # Rearrange to [B, T, G, D]
        recovered_codes = recovered_codes.permute(1, 2, 0, 3).contiguous()
        B, T, G, D = recovered_codes.shape
        recovered_codes = recovered_codes.view(B, T, G * D)  # [B, T, G*D]


        x_sample_check = self.decoder_proj(
            self.decoder_transformer(recovered_codes, src_key_padding_mask=~mask if mask is not None else None)
        )

        return x_sample_det, x_sample_check

    def entry_to_feature(self, indices, zshape):
        # indices shape might be something like [batch, seq_len, groups, quantizers] or similar
        quant_z = self.vq.get_output_from_indices(indices)  # get reconstructed vectors
        quant_z = quant_z.reshape(zshape)                   # reshape to original latent shape
        return quant_z


