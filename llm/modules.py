import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) layer with additional possibility
    to add shift.
    
    Args:
        emb_dim (int): Embedding dimension.
        eps (float): Additional small value added to variance to avoid zero division.
        bias (bool): Defines if shift is applied or not.

    Returns: 
        torch.Tensor: Normalized batch of token embeddings 
            with size [batch_size, seq_len, emb_dim].
    """
    def __init__(
            self, 
            emb_dim: int, 
            eps: float = 1e-6,
            bias: bool = False
            ) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift:
            norm_x = norm_x + self.shift
        
        return norm_x
    
