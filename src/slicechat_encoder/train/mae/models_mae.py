# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn


from ...models.mamba import make_mamba_encoder_from_config


class MaskedAutoencoderMamba(nn.Module):
    """
    Masked Autoencoder with Vision Mamba backbone for pre-extracted WSI features.
    
    Unlike the standard MAE which operates on images:
    - No patch_embed: features are already extracted (e.g., from UNI, CONCH, Virchow)
    - Reconstruction target: original features (not pixels)
    - Handles variable-length sequences with padding
    - Encoder uses bidirectional Mamba with optional hybrid transformer blocks
    - Decoder can be Mamba-based or a simple transformer
    """

    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        feature_dim: int = 768,
        norm_feature_loss: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.norm_feature_loss = norm_feature_loss
        
        # Encoder: Vision Mamba
        self.encoder = make_mamba_encoder_from_config(encoder_config)
        self.encoder_embed_dim = encoder_config.get('encoder_embed_dim', feature_dim)
        
        # Decoder: another Mamba encoder (lighter weight)
        self.decoder_embed_dim = decoder_config.get('encoder_embed_dim', 512)
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        
        self.decoder = make_mamba_encoder_from_config(decoder_config)
        
        # Prediction head: reconstruct original features
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, feature_dim, bias=True)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize only MAE-specific weights (decoder_embed, decoder_pred, mask_token).
        
        NOTE: We do NOT use self.apply() here because it would recursively re-initialize
        the Mamba encoder/decoder weights, overwriting their careful initialization.
        """
        # Initialize mask_token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize only the MAE projection layers
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        if self.decoder_embed.bias is not None:
            nn.init.constant_(self.decoder_embed.bias, 0)
            
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            nn.init.constant_(self.decoder_pred.bias, 0)
        
    def random_masking(self, x, coords, padding_mask, mask_ratio):
        """
        Perform per-sample random masking while respecting padding.
        
        Each sample keeps (1 - mask_ratio) of its own valid tokens, so samples
        with different sequence lengths will keep different numbers of tokens.
        The visible set is padded to max_len_keep across the batch.
        
        Args:
            x: [N, L, D] - input features
            coords: [N, L, 2] - coordinates for each token
            padding_mask: [N, L] - 1 for padded tokens, 0 for valid tokens
            mask_ratio: fraction of valid tokens to mask
            
        Returns:
            x_visible: [N, max_len_keep, D] - visible (kept) tokens (padded per sample)
            coords_visible: [N, max_len_keep, 2] - coordinates for visible tokens
            padding_mask_visible: [N, max_len_keep] - padding mask for visible set
            mask: [N, L] - binary mask (0=keep, 1=remove) for all original positions
            ids_restore: [N, L] - indices to restore original order
            coords_full: [N, L, 2] - coordinates in shuffled order for decoder
        """
        N, L, D = x.shape
        device = x.device
        
        # Create valid mask (1 for valid, 0 for padded)
        valid_mask = 1 - padding_mask.float()  # [N, L]
        
        # Count valid tokens per sample
        num_valid = valid_mask.sum(dim=1).long()  # [N]
        
        # Calculate per-sample number of tokens to keep
        len_keep = (num_valid.float() * (1 - mask_ratio)).long().clamp(min=1)  # [N]
        max_len_keep = len_keep.max().item()
        
        # For each sample, we want to keep (1 - mask_ratio) of valid tokens
        # Assign large noise values to padded tokens so they're always "removed"
        noise = torch.rand(N, L, device=device)
        noise = noise + padding_mask.float() * 2.0
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascending: small noise = keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first max_len_keep tokens (per-sample actual keep count varies)
        ids_keep = ids_shuffle[:, :max_len_keep]
        
        # Gather visible tokens and their coordinates
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        coords_visible = torch.gather(coords, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, 2))
        
        # Create padding mask for visible tokens (per-sample: padded if position >= len_keep[i])
        vis_positions = torch.arange(max_len_keep, device=device).unsqueeze(0)  # [1, max_len_keep]
        padding_mask_visible = (vis_positions >= len_keep.unsqueeze(1)).to(padding_mask.dtype)  # [N, max_len_keep]
        
        # Generate binary mask: 0 is keep, 1 is remove (in original order)
        # In shuffled order: positions < len_keep[i] are kept
        positions = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)  # [N, L]
        mask_shuffled = (positions >= len_keep.unsqueeze(1)).float()  # [N, L]
        mask = torch.gather(mask_shuffled, dim=1, index=ids_restore)  # [N, L]
        
        # Track coordinates in shuffled order (needed for decoder)
        coords_shuffled = torch.gather(coords, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, 2))
        
        return x_visible, coords_visible, padding_mask_visible, mask, ids_restore, coords_shuffled
        
    def forward_encoder(self, features, coords, padding_mask, mask_ratio):
        """
        Encode only the visible (unmasked) tokens.
        
        Args:
            features: [N, L, D] - input features
            coords: [N, L, 2] - coordinates
            padding_mask: [N, L] - padding mask
            mask_ratio: masking ratio
            
        Returns:
            latent: [N, max_len_keep, D_enc] - encoded visible tokens
            mask: [N, L] - binary mask for original positions
            ids_restore: [N, L] - indices to restore order
            coords_shuffled: [N, L, 2] - coordinates in shuffled order for decoder
            padding_mask_visible: [N, max_len_keep] - padding mask for visible set
        """
        # Apply masking - only keep visible tokens
        x_visible, coords_visible, padding_mask_visible, mask, ids_restore, coords_shuffled = \
            self.random_masking(features, coords, padding_mask, mask_ratio)
        
        # Encode with Mamba (only visible tokens)
        encoder_out = self.encoder(
            token_embeddings=x_visible,
            positions=coords_visible,
            encoder_padding_mask=padding_mask_visible,
            return_pooled_output=False,
            retain_order=True,
        )
        
        latent = encoder_out['encoder_out']
        
        return latent, mask, ids_restore, coords_shuffled, padding_mask_visible
        
    def forward_decoder(self, x, ids_restore, coords_shuffled, padding_mask, padding_mask_visible):
        """
        Decode by combining encoded visible tokens with mask tokens, 
        then reconstruct all features.
        
        Args:
            x: [N, max_len_keep, D_enc] - encoded visible tokens
            ids_restore: [N, L] - indices to restore original order
            coords_shuffled: [N, L, 2] - coordinates in shuffled order
            padding_mask: [N, L] - original padding mask
            padding_mask_visible: [N, max_len_keep] - padding mask for visible set
                (marks positions that are padding due to per-sample variable keep counts)
            
        Returns:
            pred: [N, L, D_feat] - predicted features for all positions
        """
        N = x.shape[0]
        L = ids_restore.shape[1]
        
        # Project encoder output to decoder dimension
        x = self.decoder_embed(x)
        
        # Replace padded visible positions with mask tokens.
        # These are positions beyond each sample's len_keep that were included
        # only for batching (they are actually masked tokens, not truly visible).
        if padding_mask_visible.any():
            mask_token_expanded = self.mask_token.expand_as(x)
            x = torch.where(padding_mask_visible.unsqueeze(-1).bool(), mask_token_expanded, x)
        
        # Append mask tokens for remaining positions
        num_mask_tokens = L - x.shape[1]
        mask_tokens = self.mask_token.expand(N, num_mask_tokens, -1)
        
        # Concatenate visible tokens and mask tokens (in shuffled order)
        x_full = torch.cat([x, mask_tokens], dim=1)  # [N, L, D_dec]
        
        # Unshuffle to restore original order
        x_full = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2])
        )
        
        # Restore coordinates to original order
        coords_restored = torch.gather(
            coords_shuffled,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, 2)
        )
        
        # Decode with Mamba
        decoder_out = self.decoder(
            token_embeddings=x_full,
            positions=coords_restored,
            encoder_padding_mask=padding_mask,
            return_pooled_output=False,
            retain_order=True,
        )
        
        x_decoded = decoder_out['encoder_out']
        
        # Project to feature dimension
        pred = self.decoder_pred(x_decoded)
        
        return pred
        
    def forward_loss(self, features, pred, mask, padding_mask):
        """
        Compute reconstruction loss on masked (removed) tokens only.
        
        Args:
            features: [N, L, D] - original features (target)
            pred: [N, L, D] - predicted features
            mask: [N, L] - binary mask (0=keep, 1=remove)
            padding_mask: [N, L] - padding mask (1=padded, 0=valid)
            
        Returns:
            loss: scalar reconstruction loss
        """
        target = features
        
        if self.norm_feature_loss:
            # Normalize features per-token
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
            
        # MSE loss per token
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per token
        
        # Only compute loss on masked (removed) AND valid (non-padded) tokens
        valid_mask = mask * (1 - padding_mask.float())
        
        # Avoid division by zero
        num_valid = valid_mask.sum()
        if num_valid > 0:
            loss = (loss * valid_mask).sum() / num_valid
        else:
            loss = (loss * valid_mask).sum()
        
        return loss
        
    def forward(self, features, coords, padding_masks, mask_ratio=0.75):
        """
        Forward pass for MAE pretraining.
        
        Args:
            features: [N, L, D] - input features from WSI patches
            coords: [N, L, 2] - coordinates of each patch
            padding_masks: [N, L] - padding mask (1=padded, 0=valid)
            mask_ratio: fraction of valid tokens to mask
            
        Returns:
            loss: reconstruction loss
            pred: [N, L, D] - predicted features
            mask: [N, L] - binary mask showing which tokens were masked
        """
        # Encode only visible tokens
        latent, mask, ids_restore, coords_shuffled, padding_mask_visible = self.forward_encoder(
            features, coords, padding_masks, mask_ratio
        )
        
        # Decode all tokens (visible + masked)
        pred = self.forward_decoder(latent, ids_restore, coords_shuffled, padding_masks, padding_mask_visible)
        
        # Compute loss on masked tokens only
        loss = self.forward_loss(features, pred, mask, padding_masks)
        
        return loss, pred, mask

