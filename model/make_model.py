import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

for param in clip_model.parameters():
    param.requires_grad = True

for param in clip_model.transformer.parameters():
    param.requires_grad = False

for param in clip_model.token_embedding.parameters():
    param.requires_grad = False

clip_model.positional_embedding.requires_grad = False

for param in clip_model.ln_final.parameters():
    param.requires_grad = False

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, features) 형태 사용
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        residual = x
        
        out, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            need_weights=False
        )
        
        out = self.dropout(out)
        out = self.layer_norm(residual + out)  # residual connection + layer normalization
        
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model=768, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_seq, kv_seq, key_padding_mask=None, attn_mask=None):
        """
        q_seq: (B, Nq, d)  - 쿼리(예: 텍스트 토큰 시퀀스)
        kv_seq: (B, Nk, d) - 키/밸류(예: 이미지 패치/토큰 시퀀스)
        key_padding_mask: (B, Nk)  - True=패딩(무시)
        attn_mask: (Nq, Nk) 또는 (B*nhead, Nq, Nk)
        """
        
        residual = q_seq  # residual connection 추가
        
        q = self.ln_q(q_seq)
        kv = self.ln_kv(kv_seq)
        
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        h = self.dropout(attn_out)
        
        return residual + h  # residual connection 추가


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)  # LayerNorm을 residual connection 후에 적용

    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        h = residual + h  # residual connection
        return self.layer_norm(h)  # LayerNorm을 마지막에 적용


class ContextualFeatureFusion(nn.Module):
    def __init__(self, d_model=768, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.CA = CrossAttention(d_model, nhead, mlp_ratio, dropout)

        self.SA1 = SelfAttention(d_model, nhead, dropout)
        self.FFN1 = FeedForward(d_model, mlp_ratio, dropout)
        self.SA2 = SelfAttention(d_model, nhead, dropout)
        self.FFN2 = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, i, t):
        x = self.CA(t, i)
        x = self.SA1(x)
        x = self.FFN1(x)
        x = self.SA2(x)
        x = self.FFN2(x)

        return x


class build_model(nn.Module):
    def __init__(self, cfg):
        super(build_model, self).__init__()

        self.text_projection = nn.Linear(512, 768)

        # cfg에서 매개변수 추출하거나 기본값 사용
        d_model = cfg.get('d_model', 768)
        nhead = cfg.get('nhead', 8)
        mlp_ratio = cfg.get('mlp_ratio', 4.0)
        dropout = cfg.get('dropout', 0.1)
        
        self.CFF = ContextualFeatureFusion(d_model, nhead, mlp_ratio, dropout)
       
    def forward(self, images, text_ids):
        # 이미지와 텍스트 인코딩
        image_features = self.visualEncoder(images)
        text_features = self.textEncoder(text_ids)

        text_feature_seq_proj = self.text_projection(text_features)
        
        # Contextual Feature Fusion
        fused_features = self.CFF(image_features, text_features)
        
        return fused_features

    def textEncoder(self, text_ids):
        x = clip_model.token_embedding(text_ids).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = clip_model.transformer(x)

        return x

    def visualEncoder(self, images):
        visual = clip_model.visual
        assert hasattr(visual, "conv1")

        B = images.size(0)
        x = visual.conv1(images.type(clip_model.dtype))  # [B, D, H/patch, W/patch]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)  # [B, L_patch, D]
        cls = visual.class_embedding.to(images.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=images.dtype, device=images.device)
        x = torch.cat([cls, x], dim=1)                     # [B, 1 + L_patch, D]
        x = x + visual.positional_embedding.to(images.dtype)
        x = visual.ln_pre(x)

        if getattr(visual.transformer, 'batch_first', False):
            x = visual.transformer(x)                      # [B, L, D]
        else:
            x = x.permute(1, 0, 2)                         # [L, B, D]
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                         # [B, L, D]
            
        return x


def make_model(cfg, num_class, camera_view, view_num):
    model = build_model(cfg)
    return model