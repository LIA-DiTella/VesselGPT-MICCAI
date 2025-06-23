import torch
import torch.nn as nn
import torch.nn.functional as F

from Stage1.modelsMultitalk.lib.quantizer import VectorQuantizer
from Stage1.modelsMultitalk.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from Stage1.base import BaseModel


class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """

    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args, args.in_dim)
        self.quantize = VectorQuantizer(args.n_embed,
                                        args.zquant_dim,
                                        beta=0.25)
        self.args = args



    def encode(self, x, x_a=None):
        h = self.encoder(x) ## x --> z'
        h = h.view(x.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim)
        h = h.view(x.shape[0], -1, self.args.zquant_dim)
        quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        
        return quant, emb_loss, info
        #return h


    def decode2(self, quant):
        #BCL
        quant = quant.permute(0,2,1)
        #quant = quant.view(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        #quant = quant.reshape(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        quant = quant.reshape(quant.shape[0], self.args.zquant_dim, self.args.face_quan_num).contiguous()
        quant = quant.view(quant.shape[0], -1,  self.args.face_quan_num*self.args.zquant_dim).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = self.decoder(quant) ## z' --> x
        return dec
    
    def decode(self, quant):
        # Assuming quant has the shape (batch_size, num_tokens, zquant_dim)
        # Step 1: Reshape or permute the input tensor as required by your model
        # Change the shape to (B, num_tokens, zquant_dim) if needed
        quant = quant.permute(0, 2, 1)  # Change shape to (batch_size, zquant_dim, num_tokens)

        # Step 2: Reshape for the decoder input
        quant = quant.reshape(quant.shape[0], -1, self.args.face_quan_num * self.args.zquant_dim)

        # Step 3: Ensure the shape matches the expected input of the decoder
        # The decoder might expect the shape to be (batch_size, num_features, seq_len)
        quant = quant.permute(0, 2, 1)  # Change back to (batch_size, features, seq_len)

        # Step 4: Pass through the decoder
        dec = self.decoder(quant)  # z' --> x

        return dec

    def forward(self, x):
        #template = template.unsqueeze(1) #B,V*3 -> B, 1, V*3
        #x = x - template

        ###x.shape: [B, L C]
        # Assuming your model has named layers

        quant, emb_loss, info = self.encode(x)
        #quant = self.encode(x)
        ### quant [B, C, L]
        dec = self.decode(quant)
        #dec = dec + template
        return dec, emb_loss, info
        #return dec


    def sample_step(self, x, x_a=None):
        
        quant_z, _, info = self.encode(x, x_a)
        
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check
    
    def sample_step_wc(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        
        return x_sample_det
    
    @torch.no_grad()
    def get_quant_no_grad(self, x, x_a=None):

        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices
    
    def get_quant(self, x, x_a=None):

        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x
    
    @torch.no_grad()
    def decode_quant(self, quant_z, zshape):
        #quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        #quant_z = quant_z.permute(0,2,1)
        #print("quant_z", quant_z.shape)
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:

            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)

        else:
            ix = logits
        #ix = torch.reshape(ix, (-1,1))
 
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0
        ##########
        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args):
    super().__init__()
    self.args = args
    size = self.args.in_dim
    dim = self.args.hidden_size
    self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(self.args.neg, True))
    if args.quant_factor == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
        for _ in range(1, args.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.args.neg, True),
                        nn.InstanceNorm1d(dim, affine=args.INaffine),
                        nn.MaxPool1d(2)
                        )]
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
                self.args.num_hidden_layers,
        num_attention_heads=\
                self.args.num_attention_heads,
        intermediate_size=\
                self.args.intermediate_size)
    self.encoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.encoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    #print("inputs", inputs.device)
    inputs = self.vertice_mapping(inputs)
    #print(f"After vertice_mapping: {inputs}")
    for param in self.vertice_mapping.parameters():
        if torch.isnan(param).any():
            print("NaN detected in model vertice mapping parameters!")
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]
    #print(f"After squasher: {inputs}")
    for param in self.squasher.parameters():
        if torch.isnan(param).any():
            print("NaN detected in model squasher parameters!")
    encoder_features = self.encoder_linear_embedding(inputs)
    #print(f"After encoder_linear_embedding: {encoder_features}")

    encoder_features = self.encoder_pos_embedding(encoder_features)
    #print(f"After encoder_pos_embedding: {encoder_features}")

    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
    #print(f"After encoder_transformer: {encoder_features}")


    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args, out_dim, is_audio=False):
    super().__init__()
    self.args = args
    size=self.args.hidden_size
    dim=self.args.hidden_size
    self.expander = nn.ModuleList()
    if args.quant_factor == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))                      
        num_layers = args.quant_factor+2 \
            if is_audio else args.quant_factor

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.args.neg, True),
                                nn.InstanceNorm1d(dim, affine=args.INaffine),
                                ))
    self.decoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
            self.args.num_hidden_layers,
        num_attention_heads=\
            self.args.num_attention_heads,
        intermediate_size=\
            self.args.intermediate_size)
    self.decoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.decoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

    self.vertice_map_reverse = nn.Linear(args.hidden_size,out_dim)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)
    inputs = inputs.permute(0,2,1) #BLC
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.vertice_map_reverse(decoder_features)
    return pred_recon
  
def validate(val_loader, model, loss_fn, epoch, cfg, device):
    accumulated_loss = 0
    accumulated_rec = 0
    accumulated_quant = 0
    model.eval()

    with torch.no_grad():
        for inputs in val_loader:
            
            inputs = inputs.to(device)
            out, quant_loss, info = model(inputs)
            
            # LOSS
            loss, loss_details = loss_fn(out, inputs, quant_loss, quant_loss_weight=cfg.quant_loss_weight)
            accumulated_loss += loss
            accumulated_rec += loss_details[0].item()
            accumulated_quant += loss_details[1].item()
        

        # Calculate the average loss over the entire dataset
        avg_loss = accumulated_loss / len(val_loader)
        rec_loss = accumulated_rec / len(val_loader)
        quant_loss = accumulated_quant / len(val_loader)

            

    return rec_loss, quant_loss