import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import numpy as np

from .pos_emb import PosEmbFactory
from .vit import Transformer

"""
    Cited and modified from
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
"""


class DMaeV5Encoder(nn.Module):
    def __init__(self, *,
                 patch_num,  # joint num
                 patch_dim,  # the number of sampling frame * joint coordinate shape
                 num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.,
                 window_size=15):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_dim = patch_dim
        self.to_patch_embedding = PosEmbFactory(emb_type="fourier", d_pos=dim)

        # self.pos_embedding = nn.Parameter(torch.randn(1, patch_num, dim))
        frame_idx = window_size
        joint_idx = patch_num // window_size
        pos_idx = torch.from_numpy(np.array([[(x, y) for x in range(frame_idx) for y in range(joint_idx)]])).to(
            torch.float32)
        pos_emb_factory = PosEmbFactory(emb_type="fourier", d_in=2, d_pos=dim)
        self.pos_embedding = pos_emb_factory(pos_idx)
        self.pos_embedding = self.pos_embedding.permute(0, 2, 1).cuda()

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )  # extract hidden space on the first row to get classification

    def forward(self, skel_data):
        x = self.to_patch_embedding(skel_data)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class DMaeV5(nn.Module):
    def __init__(self,
                 *,
                 encoder,
                 decoder_dim,
                 masking_ratio=0.75,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # encoder parameters
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.patch_to_emb = encoder.to_patch_embedding

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_prediction = nn.Linear(decoder_dim, self.encoder.patch_dim)

    def load_checkpoint(self, load_dir, tag=None):
        load_path = os.path.join(
            load_dir,
            str(tag) + ".pth",
        )
        client_states = torch.load(load_path)
        state_dict = client_states['model']
        self.load_state_dict(state_dict, strict=True)
        return load_path, client_states

    def forward(self, skel_data, force_rand_idx=None):
        device = skel_data.device
        batch, num_patches, *_ = skel_data.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(skel_data)
        tokens = tokens.permute(0, 2, 1)  # TO: batch, patch_size, hidden_dim
        tokens = tokens + self.encoder.pos_embedding

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        if force_rand_idx is not None:
            masked_indices = force_rand_idx[0]
            unmasked_indices = force_rand_idx[1]
            num_masked = masked_indices.shape[1]
        else:
            num_masked = int(self.masking_ratio * num_patches)
            rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        skel_masked = skel_data[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(unmasked_tokens)

        # project encoder to decoder dimensions, if they are not equal
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        skel_masked_pred = self.to_prediction(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(skel_masked_pred, skel_masked)
        return skel_masked_pred, skel_masked, masked_indices, recon_loss

    def change_ratio(self, ratio):
        self.masking_ratio = ratio


class DMaeV5_ft(nn.Module):
    def __init__(self,
                 *,
                 encoder,
                 decoder_dim,
                 masking_ratio=0.75,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # encoder parameters
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.patch_to_emb = encoder.to_patch_embedding

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(210, decoder_dim)
        self.decoder_pos_emb_ft = nn.Embedding(num_patches, decoder_dim)
        self.to_prediction = nn.Linear(decoder_dim, self.encoder.patch_dim)

    def load_checkpoint(self, load_dir, tag=None):
        load_path = os.path.join(
            load_dir,
            str(tag) + ".pth",
        )
        client_states = torch.load(load_path)
        state_dict = client_states['model']
        self.load_state_dict(state_dict, strict=True)
        return load_path, client_states

    def forward(self, skel_data, force_rand_idx=None):
        device = skel_data.device
        batch, num_patches, *_ = skel_data.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(skel_data)
        tokens = tokens.permute(0, 2, 1)  # TO: batch, patch_size, hidden_dim
        tokens = tokens + self.encoder.pos_embedding

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        if force_rand_idx is not None:
            masked_indices = force_rand_idx[0]
            unmasked_indices = force_rand_idx[1]
            num_masked = masked_indices.shape[1]
        else:
            num_masked = int(self.masking_ratio * num_patches)
            rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        skel_masked = skel_data[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(unmasked_tokens)

        # project encoder to decoder dimensions, if they are not equal
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb_ft(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb_ft(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        skel_masked_pred = self.to_prediction(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(skel_masked_pred, skel_masked)
        return skel_masked_pred, skel_masked, masked_indices, recon_loss

    def change_ratio(self, ratio):
        self.masking_ratio = ratio


def dmae_v5(patch_num, patch_dim, window_size, mask_ratio, ft=False):
    v = DMaeV5Encoder(
        patch_num=patch_num,  # joint num
        patch_dim=patch_dim,  # the number of sampling frame
        num_classes=patch_dim,
        window_size=window_size,
        dim=256,  # hidden space
        depth=6,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0., emb_dropout=0.
    )
    if ft:
        model = DMaeV5_ft(
            encoder=v,
            decoder_dim=128,
            masking_ratio=mask_ratio,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64)
    else:
        model = DMaeV5(
            encoder=v,
            decoder_dim=128,
            masking_ratio=mask_ratio,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64)
    return model
