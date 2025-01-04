# import models_pvt
# from attention import MultiModalTransformer
# from torch import nn
#
#
# class PVTSimCLR(nn.Module):
#
#     def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True):
#         super(PVTSimCLR, self).__init__()
#
#         self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
#         num_ftrs = self.backbone.head.in_features
#
#         self.proj = nn.Linear(num_ftrs, out_dim)
#
#         self.proj_context = nn.Linear(context_dim, out_dim)
#
#         # attention
#         dim_head = out_dim // num_head
#         self.mm_transformer = MultiModalTransformer(out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout)
#
#         self.norm1 = nn.LayerNorm(context_dim)
#
#     def forward(self, x, context=None):
#         h = self.backbone.forward_features(x)  # shape = B, N, D
#         h = h.squeeze()
#
#         # project to targeted dim
#         x = self.proj(h)
#         # Check context before applying LayerNorm
#
#         context = self.proj_context(self.norm1(context))
#
#         # multi-modal attention
#         x = self.mm_transformer(x, context=context)
#
#         # return the classification token
#         return x[:, 0]
# import models_pvt
# from attention import MultiModalTransformer
# from torch import nn
#
#
# class PVTSimCLR(nn.Module):
#
#     def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True):
#         super(PVTSimCLR, self).__init__()
#
#         self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
#         num_ftrs = self.backbone.head.in_features
#
#         self.proj = nn.Linear(num_ftrs, out_dim)
#
#         self.proj_context = nn.Linear(context_dim, out_dim)
#
#         # attention
#         dim_head = out_dim // num_head
#         self.mm_transformer = MultiModalTransformer(out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout)
#
#         self.norm1 = nn.LayerNorm(context_dim)
#
#     #
#     def forward(self, x, context=None):
#         # Forward through the PVT backbone
#         h = self.backbone.forward_features(x)  # Expected shape: [B, N, D]
#
#         # Debug output shape
#         print(f"Shape of h after backbone processing: {h.shape}")
#
#         # Remove unnecessary squeeze operation
#         if h.ndim == 3 and h.shape[1] == 1:
#             h = h.view(h.shape[0], -1)
#
#         # Project to targeted dimension
#         x = self.proj(h)
#
#         # Handle the context (if provided)
#         if context is None:
#             print("Warning: `context` is None in PVTSimCLR forward method.")
#             return x  # Skip further processing if context is critical
#
#         # Debug context shape
#         print(f"`context` shape before normalization: {context.shape}")
#
#         context = self.proj_context(self.norm1(context))
#
#         # Multi-modal attention
#         x = self.mm_transformer(x, context=context)
#
#         # Return the classification token
#         return x[:, 0]
#
import models_pvt
from attention import MultiModalTransformer
from torch import nn


class PVTSimCLR(nn.Module):
    def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True):
        super(PVTSimCLR, self).__init__()

        # Load the backbone from models_pvt
        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        # Linear projection layers
        self.proj = nn.Linear(num_ftrs, out_dim)
        self.proj_context = nn.Linear(context_dim, out_dim)

        # Attention module
        dim_head = out_dim // num_head
        self.mm_transformer = MultiModalTransformer(
            out_dim, mm_depth, num_head, dim_head, context_dim=out_dim, dropout=dropout
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(context_dim)
        self.additional_norm = nn.LayerNorm(out_dim)  # Added normalization for compatibility

    def forward(self, x, context=None):
        """
        Forward pass for PVTSimCLR.

        Parameters:
        - x: Input tensor of shape (B, C, H, W).
        - context: Context tensor of shape (B, Context_Dim).

        Returns:
        - Tensor of shape (B, Out_Dim).
        """
        # Extract features using the PVT backbone
        h = self.backbone.forward_features(x)  # Expected shape: (B, N, D)
        h = h.squeeze()

        # Project to the target dimension
        x = self.proj(h)

        # Normalize and process the context if provided
        if context is not None:
            context = self.proj_context(self.norm1(context))
        else:
            raise ValueError("Context input is required for PVTSimCLR.")

        # Multi-modal attention
        x = self.mm_transformer(x, context=context)
        x = self.additional_norm(x)  # Added normalization

        # Return the classification token
        return x[:, 0]


if __name__ == "__main__":
    # Example usage for testing
    import torch
    from attention import MultiModalTransformer  # Ensure `attention.py` is present in the module

    # Input example
    x = torch.randn((8, 3, 224, 224))  # Batch of 8, 3 channels, 224x224 image size
    context = torch.randn((8, 9))      # Context tensor with 9 dimensions

    # Initialize the model
    model = PVTSimCLR(base_model="pvt_tiny", out_dim=512, context_dim=9, pretrained=False)
    output = model(x, context=context)

    # Print output shape
    print(f"Output shape: {output.shape}")
