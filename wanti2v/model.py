import math
import torch
import torch.nn as nn
from attention import flash_attention

def sinusoidal_embedding_1d(dim, position):
    """
    # Klasik yöntem
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    Bu kod, matematiksel formülü "Pythonic" veya "HuggingFace-style" değil de, doğrudan bir "Lineer Cebir" problemi olarak çözüyor.
    """
    assert dim % 2 == 0
    ## Frekans hazırlanması
    half = dim // 2
    position = position.type(torch.float64)

    # [0, 1, 2, ..., half-1]
    x = -torch.arange(half).to(position)
    # [0/half, 1/half, ..., half-1/half]
    y = x.div(half)
    # [10k**0, ]
    z = torch.pow(10_000, y)

    # outer( [1 2 3], [4, 5, 6] ) 
    # - soldakini N sütun olarak yaz
    # - sagdaki her n. degeri, n. sütun ile çarp
    # 1, 1, 1     4, 5, 6      4,  5,  6
    # 2, 2, 2  x  4, 5, 6 ->   8, 10, 12
    # 3, 3, 3     4, 5, 6     12, 15, 18
    # Bu işlem sonucunda her bir pozisyon için farklı dalga boylarına sahip bir açılar matrisi elde edersin. 
    # Düşük frekanslı dalgalar büyük değişimleri, yüksek frekanslılar ise küçük detayları temsil eder.
    sinusoid = torch.outer(position, z)

    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10_000):
    """
    Rotary Positional Embedding (RoPE) için döndürme (rotation) parametrelerini hesaplar.
    
    Bu fonksiyon, her bir dizi pozisyonu ve boyut çifti için karmaşık düzlemde 
    bir 'döndürme operatörü' oluşturur. Bu operatörler, Transformer modelinin 
    tokenlar arasındaki göreceli (relative) mesafeyi anlamasını sağlar.

    Args:
        max_seq_len (int): İşlenecek maksimum dizi uzunluğu (örneğin video kare sayısı).
        dim (int): Modelin embedding boyutu (hidden_dim).
        theta (int): Frekans hesaplamasında kullanılan taban değer (standart: 10,000).

    Returns:
        torch.Tensor: (max_seq_len, dim // 2) boyutunda, torch.complex128 tipinde 
                      birim çember üzerindeki döndürme katsayıları.
    """
    assert dim % 2 == 0
    range_ = torch.arange(max_seq_len)
    
    # Frekans ölçeklendirme: Boyutun yarısı kadar (dim/2) farklı frekans hesaplanır.
    # Her bir boyut çifti farklı bir hızda 'döner'.
    range_even = torch.arange(0, dim, 2).to(torch.float64)
    range_even_div = range_even.div(dim)
    range_2 = 1.0 / torch.pow(theta, range_even_div)
    
    # Her pozisyon (range_) ile her frekansı (range_2) çarparak açıları (theta) oluşturur.
    # freqs[pos, i] = pos * (theta ^ (-2i/dim))
    freqs = torch.outer(range_, range_2)

    # torch.polar(abs, angle) fonksiyonu z = abs * (cos(angle) + i*sin(angle)) üretir.
    # Burada abs=1 (ones_like) verilerek vektörün boyunu değiştirmeyen, 
    # sadece açısını (freqs) kullanarak 'döndüren' bir birim karmaşık sayı elde edilir.
    # Bu, ileride veriyi (hidden states) karmaşık düzlemde döndürmek için kullanılacak.
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs


def rope_apply(x, grid_sizes, freqs):
    """
    f: frame
    h: height
    w: width
    """
    # x = [2, 300, 8, 64]
    # grid_sizes = [2, 3] -> [[4, 8, 8], [4, 8, 8]]
    # freqs = [16, 32]

    n = x.size(2)
    c = x.size(3) // 2
    
    # frame, height, width olarak 3 parçaya böler
    # frame: zaman boyunca döndürme
    # height: dikey eksende döndürme
    # width: yatay eksende döndürme
    # [16, 32] -> ([16, 12], [16, 10], [16, 10])
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        X = x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        x_i = torch.view_as_complex(X)

        # ilk parçanın 'f' kadar olan kısmını sadece f yap
        F = freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1) # [16, 12]
        H = freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1) # [16, 10]
        W = freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1) # [16, 10]
        # 3 parçadan da gelen döndürme değerlerini birleştir
        freqs_i = torch.cat([F, H, W], dim=-1).reshape(seq_len, 1, -1)

        # döndür
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # seq_len'in sağında kalan token kısımlarını da birleştir
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    
    return torch.stack(output).float()

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        x = [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight
    
    def _norm(self, x):
        return x * torch.rqsrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        """
        x = [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
            x: Shape [B, L, num_heads, C / num_heads]
            seq_lens: Shape [B]
            grid_sizes: Shape [B, 3], the second dimension contains (F, H, W)
            freqs: Rope freqs, shape [1024, C / num_heads / 2]
        """
        # b = x.shape[0]
        # s = x.shape[1]
        b, s, n, d = b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        )

        x = x.flatten(2)
        x = self.o(x)
        return x

class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        """
            x: [B, L1, C]
            context: [B, L2, C]
            context_lens: [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        self.norm2 = WanLayerNorm(dim, eps)

        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        """
        Args:
            x: Shape [B, L, C]
            e: Shape [B, L1, 6, C]
            seq_lens: Shape [B], length of each sequence in batch
            grid_sizes: Shape [B, 3], the second dimension contains (F, H, W)
            freqs: Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0)+e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self attn
        y = self.self_attn(
            self.norm1(x).float() * (1+e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)
        
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1+e[4].squeeze(2)) + e[3].squeeze(2)
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(path_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim)/dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x

class WanModel:
    def __init__(self, patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1), qk_norm=True, cross_attn_norm=True, eps=1e-6):
        """
        Initialize the diffusion model backbone.

        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # emb
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim*6)
        )

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d-4*(d//6)),
            rope_params(1024, 2*(d//6)),
            rope_params(1024, 2*(d//6))
        ], dim=1)

        self.init_weights()

    def forward(self, x, t, context, seq_len, y=None):
        """
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # emb
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert = seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # time emb
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                ) for u in context
            ])
        )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
    
    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)