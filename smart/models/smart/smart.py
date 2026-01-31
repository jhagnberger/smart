"""SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model

This module contains the implementation of the encoder and decoder blocks used in SMART, as well as the complete SMART model.
Designed for simulating time-independent PDEs over complex 3D geometries, SMART leverages a Transformer-based architecture
to perform simulations using solely inexpensive geometry meshes, eliminating the need for costly surface and volumetric
meshing during inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulatedPositionalEmbedding(nn.Module):
    """Embedding layer that applies a modulated sine-cosine positional embedding to the spatial positions. This means 
    that the sine-cosine functions are shifted and scaled based on learned parameters from an MLP for each position.
    This allows the model to adaptively adjust the positional embeddings based on the data to emphasize or suppress
    high-frequency variations.

    The implementation follows the original Transformer positional embedding (https://arxiv.org/abs/1706.03762) and
    the class is based on the PositionalEncoding class from the PyTorch tutorial 'Language Modeling with nn.Transformer and torchtext'.
    
    Args:
        dim: Dimensionality of the embedded positions.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original Transformer paper.
    """

    def __init__(self, dim, spatial_dim=3, max_seq_length=10000):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        self.dim_per_spatial_dim = dim_per_spatial_dim
        
        # Compute the total padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)
        
        # Modulation MLP
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim_per_spatial_dim * spatial_dim * 2))

    def compute_embedding(self, pos, shift_sin=None, scale_sin=None, shift_cos=None, scale_cos=None):
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            sin_cos_arg = pos[..., None] @ self.div_term[None, ...]
            
            embedding = torch.zeros((*sin_cos_arg.shape[:-1], self.dim_per_spatial_dim), device=sin_cos_arg.device, dtype=sin_cos_arg.dtype)
            # Apply shift and scale to embedding if provided
            if shift_sin is not None and scale_sin is not None and shift_cos is not None and scale_cos is not None:
                embedding[..., 0::2] = scale_sin * torch.sin(sin_cos_arg + shift_sin)
                embedding[..., 1::2] = scale_cos * torch.cos(sin_cos_arg + shift_cos)
            else:
                embedding[..., 0::2] = torch.sin(sin_cos_arg)
                embedding[..., 1::2] = torch.cos(sin_cos_arg)
            
        # Rearrange spatial dimensions
        embedding = rearrange(embedding, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Apply padding if necessary
        if self.total_padding > 0: embedding = torch.concat([embedding, self.padding.expand(*embedding.shape[:-1], -1)], dim=-1)
        
        return embedding
        
    def forward(self, pos):
        """Embeds the positions, normalized to [0, max_seq_length], using modulated sine-cosine positional embeddings.

        Args:
            pos: Normalized positions with shape (batch size, number points, spatial_dim).

        Returns:
            Embedded positions with shape (batch size, number points, dim).
        """
        initial_embedding = self.compute_embedding(pos)
        
        # Apply modulation MLP for shift and scaling
        shift_scale = self.mlp(initial_embedding)
        shift_sin, scale_sin, shift_cos, scale_cos = torch.unbind(rearrange(shift_scale, "b n (d shift_scale spatial_dim) -> b n spatial_dim d shift_scale", shift_scale=4, spatial_dim=self.spatial_dim), -1)
        
        embedding = self.compute_embedding(pos, shift_sin=shift_sin, scale_sin=scale_sin, shift_cos=shift_cos, scale_cos=scale_cos)
        
        return embedding


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) for spatial positions.
    
    Args:
        dim: Dimensionality of the features to be embedded.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original RoPE paper.
    """
    
    def __init__(self, dim, spatial_dim, max_seq_length=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embeddings"
        
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        
        # Compute the padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding // 2))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):
        """Applies RoPE to the features x based on the positions pos.
        
        Args:
            x: Features to apply RoPE to with shape (batch size, number points, dim).
            pos: Normalized positions with shape (batch size, number points, spatial_dim).
            
        Returns:
            Features with RoPE applied to with shape (batch size, number points, dim).
        """
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            theta = pos[..., None] @ self.div_term[None, ...]
        
        theta = rearrange(theta, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Add padding
        theta = torch.concat([theta, self.padding.expand(*theta.shape[:-1], -1)], dim=-1)
        
        # Apply rotation matrix in complex space following Llama 3 implementation
        # (https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L65)
        rotation = torch.polar(torch.ones_like(theta), theta)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        embedded = torch.view_as_real(x_complex * rotation[:, None, ...]).flatten(3)
        
        return embedded.type_as(x)
    

class CrossAttention(nn.Module):
    """Computes multi-head cross-attention (https://arxiv.org/abs/1706.03762) between the query and key/value sequences. It
    optionally applies Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) to both the query and key features.

    Args:
        dim: Dimensionality of the query and key/value features.
        num_heads: Number of attention heads. Defaults to 8.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        dropout: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Pre-layer normalization
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        
        # Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)
        
        self.dropout = dropout

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        """Applies pre-norm and computes cross-attention between q and kv.

        Args:
            q: Queries with shape (batch size, number query tokens, dim).
            kv: Key/value with shape (batch size, number key/value tokens, dim).
            q_pos (optional): Positions for the queries with shape (batch size, number query tokens, spatial_dim).
            kv_pos (optional): Positions for the key/value with shape (batch size, num key/value tokens, spatial_dim).

        Returns:
            Updated queries that attend to kv with shape (batch size, number query tokens, dim).
        """
        # Apply layer normalization
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        # Linear projections
        q = self.q(q)
        kv = self.kv(kv)
            
        # Split heads and keys/values
        q_heads = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(rearrange(kv, "b kv (h d) -> b h kv d", h=2*self.num_heads, d=self.head_dim), 2, dim=1)
        
        # Apply RoPE if positions are provided
        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)
            
        # Compute attention using PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=(self.dropout if self.training else 0.0))
        
        # Merge heads and output projection
        x = rearrange(x, "b h q d -> b q (h d)")
        x = self.out_proj(x)
        
        return x


class Modulator(nn.Module):
    """Modulator module for FiLM-like modulation (https://github.com/ethanjperez/film) of features based on conditioning parameters.
    
    Args:
        dim: Dimensionality of the features to be modulated.
        cond_dim: Dimensionality of the conditioning parameters.
        hidden_dim: Dimensionality of the hidden layer in the modulation MLP. Default is 128.
        use_residual: If true, use residual connection in modulation. Default is False.
    """
    
    def __init__(self, dim, cond_dim, hidden_dim=128, use_residual=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim * 2))
        self.use_residual = use_residual

    def forward(self, x, params):
        """Modulates the features x based on the conditioning parameters params.
        
        Args:
            x: Features to be modulated with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Modulated features with shape (batch size, number points, dim).
        """
        scale, shift = torch.tensor_split(self.mlp(params), 2, dim=-1)
        x = scale * x + shift if not self.use_residual else x + scale * x + shift
        return x


class SimulationParamModulatedMLP(nn.Module):
    """MLP with FiLM-like modulation (https://github.com/ethanjperez/film) based on simulation parameters.
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        cond_dim: Dimensionality of the conditioning parameters.
        dropout: Dropout rate. Default is 0.1.
        use_residual: If true, use residual connection in modulation. Default is False.
    """

    def __init__(self, dim, hidden_dim, cond_dim, dropout=0.1, use_residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.modulator = Modulator(hidden_dim, cond_dim, use_residual=use_residual)
        

    def forward(self, x, params):        
        """Processes the features x with an MLP, modulated based on the conditioning parameters params.
        
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.modulator(self.non_linearity(self.linear1(self.norm(x))), params)
        x = self.linear2(x)
        
        return x


class PlainMLP(nn.Module):
    """Plain multi-layer perceptron (MLP) **without** modulation
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        dropout: Dropout rate. Default is 0.1.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        

    def forward(self, x, params):
        """Processes the features x with an MLP without modulation.
            
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters will be ignored.
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.linear2(self.non_linearity(self.linear1(self.norm(x))))
        
        return x


class EncoderBlock(nn.Module):
    """The encoder block updates the latent geometry by attending to a cross-attended version of itself. This means that
    first, the latent geometry attends to a subsampled version of the input geometry to integrate geometric information,
    and then it attends to this cross-attended version of itself to refine the latent representation.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2):
        super().__init__()
        self.geo_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
       
    def forward(self, latent_geometry, subsampled_geometry, params, latent_geometry_pos=None, subsampled_geometry_pos=None):
        """Updates the latent geometry by attending to a cross-attended version of itself that first attends to a subsampled
        version of the input geometry.

        Args:
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            subsampled_geometry: Subsampled input geometry with shape (batch size, number subsampled points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
            subsampled_geometry_pos (optional): Positions of the subsampled input geometry for the positional embeddings with shape (batch size, number subsampled points, spatial_dim) . Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Updated latent geometry with shape (batch size, number latent points, dim).
                - Latent geometry after geometry cross-attention and before cross-attention and MLP with shape (batch size, number latent points, dim).
        """
        # First cross-attention with the subsampled geometry
        latent_geometry_cross = latent_geometry + self.attn_dropout(self.geo_attn(q=latent_geometry, kv=subsampled_geometry, q_pos=latent_geometry_pos, kv_pos=subsampled_geometry_pos))

        # Update the initial latent geometry by attending to the cross-attended version of itself
        latent_geometry_self = latent_geometry + self.attn_dropout(self.cross_attn(q=latent_geometry, kv=latent_geometry_cross, q_pos=latent_geometry_pos, kv_pos=latent_geometry_pos))
        
        # Pointwise MLP
        latent_geometry_mlp = latent_geometry_self + self.mlp(latent_geometry_self, params)
        
        return latent_geometry_mlp, latent_geometry_cross


class DecoderBlock(nn.Module):
    """The decoder block attends to the latent geometry of the corresponding encoder block to produce predictions
    of physical quantities at query positions.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        shared_attn: Shared cross-attention module from the corresponding encoder block. Defaults to None.
        shared_mlp: Shared MLP module from the corresponding encoder block. Defaults to None.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_attn=None, shared_mlp=None):
        super().__init__()
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout) if shared_mlp is None else shared_mlp
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout) if shared_mlp is None else shared_mlp
       
    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None):
        """Updates the queries by attending to the latent geometry of the corresponding encoder block.

        Args:
            queries: Features of the query positions with shape (batch size, number query points, dim).
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            queries_pos (optional): Positions of the query positions for the positional embeddings with shape (batch size, number query points, spatial_dim). Defaults to None.
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
        
        Returns:
            Updated queries with shape (batch size, number query points, dim).
        """
        # Cross-attention with the latent geometry
        queries = queries + self.attn_dropout(self.attn(q=queries, kv=latent_geometry, q_pos=queries_pos, kv_pos=latent_geometry_pos))
        
        # Pointwise MLP
        queries = queries + self.mlp(queries, params)
        
        return queries


def sample_geometry(geometry, num_samples):
    """Samples points from the input geometry.

    Args:
        geometry: Input geometry with shape (batch size, number points, spatial_dim).
        num_samples: Number of points to sample.
    
    Returns:
        Sampled input geometry with shape (batch size, num_samples, spatial_dim).
    """
    idx = torch.randperm(geometry.shape[1], device=geometry.device)[:num_samples]
    sampled_geometry = geometry[:, idx, :]
    return sampled_geometry

   
class SMART(nn.Module):
    """SMART model for simulating time-independent PDEs over complex 3D geometries.
    
    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 2.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of points of the latent geometry. Default is 4096.
        subsampled_geometry_points: Number of points in the subsampled geometry for geometry cross-attention. Default is 16384.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions to use more/less of the dynamic range of the positional embedding. Default is 1000.
        dropout: Dropout rate. Default is 0.0.
        subregion_size: Number of query points to process in each subregion during sequential inference. Default is 262144.
    """

    def __init__(self, spatial_dim=3,
                 surface_channels=1,
                 volume_channels=3,
                 parameter_channels=2,
                 latent_dim=256,
                 latent_geometry_points=4096,
                 subsampled_geometry_points=16384,
                 num_encoder_decoder_blocks=8, 
                 num_heads=8,
                 pos_scale_factor=1000,
                 dropout=0.0,
                 subregion_size=262144):
        super(SMART, self).__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."
        
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.pos_scale_factor = pos_scale_factor
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        
        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for i in range(num_encoder_decoder_blocks)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels, shared_attn=self.encoder_blocks[i].cross_attn, shared_mlp=self.encoder_blocks[i].mlp) for i in range(num_encoder_decoder_blocks)])
        
        # Final MLP
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, surface_channels+volume_channels))
        
        # Subregion size for inference
        self.subregion_size = subregion_size
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    # Weight initialization from Transolver
    # (https://github.com/thuml/Transolver/blob/a11be9c4f7db1885e4b08c68432bc31799492ec9/Car-Design-ShapeNetCar/models/Transolver.py#L168)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def encode(self, geo, params):
        # Prepare positions by scaling
        geo = geo * self.pos_scale_factor
        
        # Sample the initial latent geometry
        latent_geo_pos = sample_geometry(geo, self.num_geo)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            # Subsample the geometry for geometry cross-attention
            sub_geo_pos = sample_geometry(geo, self.subsampled_geometry_points)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            
            # Apply encoder block
            latent_geo_emb, e_ca = block(latent_geo_emb, sub_geo_emb, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        return intermediate_latent_geometries, latent_geo_pos
    
    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos):
        # Prepare positions by scaling
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        
        # Apply decoder blocks
        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(query_emb, e_ca, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos)
        
        # Final MLP
        pred = self.mlp(query_emb)
        
        return pred
    
    def forward(self, geo, surf_query_pos, vol_query_pos, params):
        """Forward method for SMART model.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(geo, params)
        
        # Prepare query positions by concatenating surface and volume query positions
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        
        # Decode
        pred = self.decode(intermediate_latent_geometries, latent_geo_pos, params, query_pos)
        
        # Split surface and volume predictions
        pred_surf = pred[:, :surf_query_pos.shape[1], 0:self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1]:, self.surface_channels:]
        
        return pred_surf, pred_vol
    
    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params):
        """Sequential inference method to handle large number of query points that may not fit into GPU memory.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(geo, params)
        
        # Surface predictions sequentially
        N_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, N_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i+self.subregion_size, :]
            y_surf_subregion = self.decode(intermediate_latent_geometries, latent_geo_pos, params, surf_subregion)
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        # Volume predictions sequentially
        N_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, N_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i+self.subregion_size, :]
            y_vol_subregion = self.decode(intermediate_latent_geometries, latent_geo_pos, params, vol_subregion)
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)
        
        # Split surface and volume predictions
        pred_surf = y_hat_surf[:, :, 0:self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels:]
        
        return pred_surf, pred_vol
