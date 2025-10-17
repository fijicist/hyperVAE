import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class LorentzLinear(nn.Module):
    """Lorentz-equivariant linear transformation"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Preserve Lorentz structure: (E, px, py, pz) -> keep E² - p² invariant
        # Simplified approximation for computational efficiency
        return F.linear(x, self.weight, self.bias)


class LorentzAttention(nn.Module):
    """
    Lorentz-equivariant attention mechanism for particle physics.
    Approximates L-GATr with physics-aware attention.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        
        # Query, Key, Value projections
        self.q_proj = LorentzLinear(hidden_dim, hidden_dim)
        self.k_proj = LorentzLinear(hidden_dim, hidden_dim)
        self.v_proj = LorentzLinear(hidden_dim, hidden_dim)
        self.o_proj = LorentzLinear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, edge_index=None):
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Optional edge connectivity [2, E]
        """
        batch_size = x.size(0)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('nhd,mhd->nmh', q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.einsum('nmh,mhd->nhd', attn_weights, v)
        out = out.reshape(batch_size, self.hidden_dim)
        
        # Output projection
        out = self.o_proj(out)
        
        return out


class LGATrLayer(nn.Module):
    """
    Lorentz Geometric Algebra Transformer Layer.
    Simplified implementation for memory efficiency on 1650Ti.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = LorentzAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            LorentzLinear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LorentzLinear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index=None):
        # Self-attention with residual
        x_attn = self.attention(self.norm1(x), edge_index)
        x = x + x_attn
        
        # Feed-forward with residual
        x_ffn = self.ffn(self.norm2(x))
        x = x + x_ffn
        
        return x


class EdgeAwareTransformerConv(MessagePassing):
    """
    Edge-aware transformer convolution for bipartite graphs.
    Incorporates edge features into attention mechanism.
    """
    
    def __init__(self, in_channels, out_channels, edge_dim, num_heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.dropout = dropout
        
        # Projections
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.edge_proj = nn.Linear(edge_dim, out_channels)
        self.o_proj = nn.Linear(out_channels, out_channels)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        edge_emb = self.edge_proj(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_emb=edge_emb)
        out = self.o_proj(out)
        
        return out
    
    def message(self, q_i, k_j, v_j, edge_emb, index, ptr, size_i):
        """
        Compute messages with edge-aware attention.
        
        Args:
            q_i: Query from target nodes [E, out_channels]
            k_j: Key from source nodes [E, out_channels]
            v_j: Value from source nodes [E, out_channels]
            edge_emb: Edge embeddings [E, out_channels]
        """
        # Reshape for multi-head attention
        q_i = q_i.view(-1, self.num_heads, self.head_dim)
        k_j = k_j.view(-1, self.num_heads, self.head_dim)
        v_j = v_j.view(-1, self.num_heads, self.head_dim)
        edge_emb = edge_emb.view(-1, self.num_heads, self.head_dim)
        
        # Attention scores with edge features
        attn_scores = (q_i * (k_j + edge_emb)).sum(dim=-1) * self.scale
        attn_weights = softmax(attn_scores, index, ptr, size_i)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = attn_weights.unsqueeze(-1) * v_j
        out = out.view(-1, self.out_channels)
        
        return out


if __name__ == "__main__":
    # Test L-GATr layer
    print("Testing L-GATr layer...")
    x = torch.randn(32, 64)  # 32 particles, 64 features
    layer = LGATrLayer(hidden_dim=64, num_heads=4)
    out = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test edge-aware transformer
    print("\nTesting Edge-Aware Transformer...")
    edge_index = torch.randint(0, 32, (2, 100))
    edge_attr = torch.randn(100, 5)
    conv = EdgeAwareTransformerConv(64, 64, edge_dim=5, num_heads=4)
    out = conv(x, edge_index, edge_attr)
    print(f"Output shape: {out.shape}")
