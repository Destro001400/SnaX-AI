import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """RoPE (Rotary Position Embeddings) implementation."""
    def __init__(self, dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Criar buffer para cache dos senos e cossenos
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotaciona metade das dimensões do tensor."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica RoPE em queries e keys."""
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )

class EfficientAttention(nn.Module):
    """Implementação de Grouped-Query Attention com RoPE."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim)
        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Projeções Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape e transpose para atenção
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Aplicar RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repetir K, V para num_heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape e projeção final
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    """SwiGLU Activation Function - melhor que ReLU/GELU."""
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.w2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        return self.w2(F.silu(x1) * x2)

class TransformerBlock(nn.Module):
    """Bloco Transformer com GQA e SwiGLU."""
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.attention = EfficientAttention(
            config["hidden_size"],
            config["num_heads"],
            config["num_kv_heads"]
        )
        self.ffn = SwiGLU(config["hidden_size"])
        self.ln1 = nn.LayerNorm(config["hidden_size"])
        self.ln2 = nn.LayerNorm(config["hidden_size"])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class SnaXIA_v3(nn.Module):
    """Modelo principal SnaX IA v3."""
    def __init__(self, config: dict = None) -> None:
        super().__init__()
        self.config = config or {
            "vocab_size": 8192,
            "hidden_size": 384,
            "num_layers": 6,
            "num_heads": 6,
            "num_kv_heads": 2,
            "max_seq_len": 512,
        }
        
        # Token embeddings com weight tying
        self.tok_embeddings = nn.Embedding(
            self.config["vocab_size"],
            self.config["hidden_size"]
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.config)
            for _ in range(self.config["num_layers"])
        ])
        
        # Layer norm final
        self.ln_out = nn.LayerNorm(self.config["hidden_size"])
        
        # LM head com weight tying
        self.lm_head = nn.Linear(
            self.config["hidden_size"],
            self.config["vocab_size"],
            bias=False
        )
        self.lm_head.weight = self.tok_embeddings.weight
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_out(x)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, max_new_tokens: int = 50,
                temperature: float = 0.8, top_k: int = 40) -> str:
        """Gera texto usando top-k sampling."""
        self.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # Truncar para max_seq_len
            if input_ids.shape[1] > self.config["max_seq_len"]:
                input_ids = input_ids[:, -self.config["max_seq_len"]:]
            
            # Forward pass
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            # Amostragem
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Parar se encontrar EOS
            if next_token.item() == tokenizer.eos_id():
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    # Teste rápido do modelo
    config = {
        "vocab_size": 8192,
        "hidden_size": 384,
        "num_layers": 6,
        "num_heads": 6,
        "num_kv_heads": 2,
        "max_seq_len": 512,
    }
    
    model = SnaXIA_v3(config)
    x = torch.randint(0, config["vocab_size"], (1, 32))
    y = model(x)
    assert y.shape == (1, 32, config["vocab_size"])
    print("✅ Teste do modelo passou!")