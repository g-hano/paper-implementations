"""
DiffusionGemma — saf PyTorch model bileşenleri (öğrenme amaçlı).

Bu dosya HuggingFace `modeling_diffusion_gemma.py` dosyasının bağımlılıksız
yeniden yazımıdır. Tüm sınıf ve fonksiyon docstring'leri Türkçedir; öğrencilerin
mimariyi adım adım takip edebilmesi hedeflenmiştir.

DiffusionGemma nedir?
  Otoregresif (AR) modeller token'ı token üretir; DiffusionGemma ise metni
  blok blok difüzyon ile üretir. Encoder prompt'u KV önbelleğine yazar;
  decoder tuvaldeki gürültülü tokenları iteratif olarak temizler.

Mimari özeti:
  - Encoder: Gemma4 benzeri transformer + MoE; KV cache üretir
  - Decoder: Aynı ağırlıkları paylaşır; encoder KV'yi salt okunur okur
  - Self-conditioning: Önceki adım logits'leri decoder girdisine eklenir
  - Blok difüzyon: Tuval parça parça gürültüden arındırılır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGemmaTextxRotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) katmanı.

    DiffusionGemma'da farklı katman tipleri (full_attention, sliding_attention)
    farklı head boyutları ve frekans parametreleri kullanabilir. Bu sınıf her
    katman tipi için ayrı `inv_freq` tamponu tutar.

    RoPE fikri: pozisyon bilgisini vektöre eklemek yerine Q/K vektörlerini
    pozisyona göre karmaşık düzlemde döndürmek. Böylece attention, tokenlar
    arası göreceli mesafeyi doğal biçimde öğrenir.
    """
    inv_freq: torch.Tensor

    def __init__(self, config, device, layer_type):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # modeldeki benzersiz katman tipleri: örn. {"full_attention", "sliding_attention"}
        self.layer_types = set(config.layer_types)
        self.rope_init_fns = {}
        self.rope_type = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            rope_init_fn = self.compute_default_rope_parameters
            rope_type = rope_params.get("rope_type", "default")

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            # global (full) attention katmanlarında head boyutu daha büyük olabilir
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            # her katman tipi için ayrı frekans tamponu kaydet
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config=None,
        device=None,
        seq_len=None,
        layer_type=None,
    ):
        """
        Klasik RoPE ters frekanslarını hesaplar.

        Formül: inv_freq[i] = 1 / (theta ^ (2i / dim))
        Burada theta = rope_theta (genelde 10_000), dim = head boyutu.

        Returns:
            inv_freq: [dim/2] — her boyut çifti için bir frekans
            attention_factor: bu RoPE tipinde 1.0 (ek ölçekleme yok)
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0  # bu RoPE tipinde kullanılmıyor
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type=None):
        """
        Verilen pozisyonlar için cos/sin embedding üretir.

        Args:
            x: referans tensör (cihaz ve dtype için)
            position_ids: [batch, seq_len] — her tokenın mutlak pozisyonu
            layer_type: hangi katman tipinin frekanslarını kullanacağımız

        Returns:
            cos, sin: [batch, seq_len, head_dim] — Q/K'ya uygulanacak döndürme katsayıları
        """
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # inv_freq: [dim/2] -> [batch, dim/2, 1]
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # matmul: her (batch, pozisyon) çifti için açı = pozisyon * inv_freq
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # sin/cos için boyutu ikiye katla: [batch, seq, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DiffusionGemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    LayerNorm'dan farkı: ortalamayı çıkarmaz, sadece L2 normuna göre ölçekler.
    Gemma ailesinde standart normalizasyon yöntemidir.

    Formül: x / sqrt(mean(x²) + eps) * weight
    """
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states):
        """
        RMSNorm uygular; isteğe bağlı öğrenilebilir weight ile ölçekler.

        Args:
            hidden_states: [..., dim]

        Returns:
            Normalize edilmiş tensör, girdi ile aynı şekil ve dtype
        """
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output *= self.weight.float()
        return normed_output.type_as(hidden_states)


class DiffusionGemmaClippableLiner(nn.Module):
    """
    Giriş/çıkış değerlerini belirli aralıkta tutan doğrusal katman.

    Eğitim stabilitesi için aktivasyonları clamp eder. DiffusionGemma'nın
    orijinal implementasyonunda `use_clipped_linears` bayrağı ile açılır.
    """
    def __init__(
        self,
        config,
        in_features,
        out_features,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states):
        """
        İsteğe bağlı giriş/çıkış clamp ile doğrusal dönüşüm uygular.

        Args:
            hidden_states: [..., in_features]

        Returns:
            [..., out_features]
        """
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


def rotate_half(x):
    """
    RoPE'nin temel döndürme adımı.

    Vektörün ilk yarısı ile ikinci yarısını yer değiştirip işaret değiştirir.
    Bu, karmaşık çarpım (cos + i·sin) · (x1 + i·x2) işleminin vektör karşılığıdır.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """
    Q veya K tensörüne RoPE uygular.

    Formül: x_rotated = x * cos + rotate_half(x) * sin

    unsqueeze_dim: cos/sin'in Q/K ile broadcast edilebilmesi için hangi eksene
    boyut ekleneceğini belirler. Attention'da Q/K şekli [batch, heads, seq, dim]
    olduğundan genelde unsqueeze_dim=2 kullanılır.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states, n_rep):
    """
    Grouped Query Attention (GQA) için K/V head'lerini çoğaltır.

    Az sayıda K/V head'i (num_key_value_heads) vardır; Q head sayısı (num_attention_heads)
    daha fazladır. Her K/V head'i, ona karşılık gelen Q head grubu kadar tekrarlanır.

    torch.repeat_interleave(x, dim=1, repeats=n_rep) ile aynı işi yapar,
    ama expand+reshape ile daha verimli.

    Girdi : [batch, num_kv_heads, seq_len, head_dim]
    Çıktı : [batch, num_attention_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    b, nkvh, sl, hd = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, nkvh, n_rep, sl, hd)
    return hidden_states.reshape(b, nkvh * n_rep, sl, hd)


def eager_attention_forward(
    module: nn.Module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    softcap=None,
    **kwargs
):
    """
    Standart scaled dot-product attention (saf PyTorch, Flash/SDPA yok).

    Adımlar:
      1. K/V head'lerini GQA için tekrarla
      2. attn_scores = Q @ K^T * scaling
      3. (opsiyonel) softcap ile skorları sınırla
      4. attention_mask ekle (causal/sliding mask burada uygulanır)
      5. softmax → dropout → @ V

    DiffusionGemma'da scaling=1.0 kullanılır (klasik 1/sqrt(d) değil).

    Returns:
        attn_output: [batch, seq_len, num_heads, head_dim]
        attn_weights: [batch, num_heads, seq_len, seq_len]
    """
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # query, key_states: [batch, heads, seq, head_dim]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        # Gemma softcapping: aşırı büyük logitleri tanh ile yumuşatır
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        # mask genelde -inf ile engellenen pozisyonları içerir (additive mask)
        attn_weights += attention_mask

    # softmax fp32'de hesaplanır (sayısal kararlılık), sonra orijinal dtype'a döner
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class DiffusionGemmaEncoderTextAttention(nn.Module):
    """
    DiffusionGemma encoder (denoiser) attention katmanı.

    Gemma4 attention'a benzer; farklar:
      - Paylaşımlı KV cache mantığı yok (difüzyon modelinde kullanılmıyor)
      - `full_attention` katmanlarında V projeksiyonu yok; V = K (weight tying)
      - `sliding_attention` katmanlarında ayrı V projeksiyonu var
      - Global (full) katmanlarda head_dim=512, sliding'de head_dim=256

    is_causal:
      - use_bidirectional_attention="all" ise False → tüm tokenlar birbirini görür
      - aksi halde True → causal (geleceği görme) mask uygulanır
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_causal = config.use_bidirectional_attention != 'all'

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # full attention katmanları daha büyük head kullanır (global context)
        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0  # Gemma scaling; sqrt(head_dim) ile bölünmez
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        # sliding katmanlarda ayrı V; full katmanlarda V = K (v_proj=None)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias
        )

        # Q/K/V üzerinde head-bazlı RMSNorm (Gemma 2/3/4 mimarisi)
        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        **kwargs
    ):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple — RoPE katsayıları
            attention_mask: additive mask veya None
            past_key_values: önceki adımlardan KV cache (encoder tarafında)

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: attention ağırlıkları (eager modda)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # --- Query yolu ---
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)  # [batch, heads, seq, dim]

        # --- Key / Value yolu ---
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # encoder KV cache'e yeni K/V ekle (varsa)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # saf PyTorch attention — HF'deki ALL_ATTENTION_FUNCTIONS yerine doğrudan eager
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        # head'leri birleştir ve çıkış projeksiyonu
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffusionGemmaDecoderTextAttention(nn.Module):
    """
    DiffusionGemma decoder attention katmanı — tuval self-attention + encoder çapraz dikkat.

    DiffusionGemma mimarisinde encoder, kullanıcı prompt'unu işleyerek her katmanda
    Key/Value (K/V) tensörlerini bir önbelleğe (KV cache) yazar. Decoder ise difüzyon
    tuvalindeki (canvas) gürültülü tokenları adım adım temizlerken bu önbelleği
    *salt okunur* olarak kullanır; decoder kendi K/V'sini önbelleğe eklemez.

    Bu katmanın görevi, tuval tokenlarından üretilen Query'lerin hem tuvalin kendi
    K/V'sine hem de encoder'ın önbelleğindeki K/V'ye bakmasını sağlamaktır. Pratikte
    attention şu şekilde çalışır:

      1. Tuval gizli durumlarından Q, K, V projeksiyonları hesaplanır (RoPE uygulanır).
      2. Encoder önbelleğindeki K/V, mevcut tuval K/V ile seq ekseninde birleştirilir
         (torch.cat). past_key_values.update() *çağrılmaz* — önbellek değişmez.
      3. Birleşik K/V üzerinde iki yönlü (bidirectional) scaled dot-product attention
         uygulanır (is_causal=False).

    Encoder attention'dan üç kritik fark:
      - Paylaşımlı KV katmanı mantığı yok (PLE kullanılmıyor).
      - KV önbelleği güncellenmez; decoder encoder çıktısını sadece okur.
      - is_causal her zaman False; config.use_bidirectional_attention yalnızca encoder'ı
        etkiler, decoder attention her zaman çift yönlüdür.

    Öğrenci notu: Klasik encoder-decoder modellerde decoder, encoder'ın son gizli
    durumlarını cross-attention ile okur. DiffusionGemma'da ise decoder doğrudan
    encoder'ın *tüm ara katman K/V önbelleğini* okur; bu sayede tuval tokenları
    prompt bağlamına her katmanda erişebilir.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_causal = False  # decoder'da attention her zaman iki yönlüdür

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        **kwargs
    ):
        """
        Tuval tokenları için çapraz dikkat + self-attention hesaplar.

        Encoder KV önbelleği salt okunur: K/V birleştirilir, önbellek güncellenmez.

        Args:
            hidden_states: [batch, canvas_len, hidden_size] — tuval gizli durumları
            position_embeddings: (cos, sin) — tuval pozisyonları için RoPE katsayıları
            attention_mask: encoder önbelleği + tuval için additive mask
            past_key_values: encoder'ın yazdığı salt okunur KV önbelleği

        Returns:
            attn_output: [batch, canvas_len, hidden_size]
            attn_weights: attention ağırlıkları (eager modda)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # --- Query yolu (tuval tokenları) ---
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)  # [batch, heads, seq, dim]

        # --- Key / Value yolu (tuval tokenları) ---
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        # Encoder KV önbelleğini oku ve tuval K/V ile birleştir (önbelleği güncelleme!)
        if past_key_values is not None:
            encoder_key_states = past_key_values.layers[self.layer_idx].keys
            encoder_value_states = past_key_values.layers[self.layer_idx].values
            key_states = torch.cat([encoder_key_states, key_states], dim=2)
            value_states = torch.cat([encoder_value_states, value_states], dim=2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffusionGemmaText4MLP(nn.Module):
    """
    Gemma4 tarzı kapılı feed-forward ağı (SwiGLU benzeri).

    Her token bağımsız işlenir. gate_proj ve up_proj paralel çalışır; aktivasyon
    gate üzerinden uygulanır, up ile çarpılır ve down_proj ile gizli boyuta geri
    projekte edilir.

    Formül: down_proj( act_fn(gate_proj(x)) * up_proj(x) )
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = config.hidden_activation

    def forward(self, x):
        """
        Kapılı MLP ileri geçişi.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, hidden_size]
        """
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        act = self.act_fn(gate)
        down = self.down_proj(act * up)
        return down


class DiffusionGemmaTextRouter(nn.Module):
    """
    Mixture-of-Experts (MoE) yönlendirici (router) katmanı.

    Her token için hangi uzmanların (expert) devreye gireceğine karar verir.
    DiffusionGemma'da encoder ve decoder katmanları hem sabit MLP hem de MoE
    yolunu paralel çalıştırır; router yalnızca MoE kolundaki token yönlendirmesinden
    sorumludur.

    Çalışma adımları:
      1. Girdi RMSNorm ile normalize edilir, hidden_size^-0.5 ve öğrenilebilir scale
         ile ölçeklenir.
      2. Linear projeksiyon tüm uzmanlara ham skor üretir.
      3. Softmax (fp32) ile olasılık dağılımı hesaplanır.
      4. top_k_experts kadar uzman seçilir; ağırlıklar token başına 1'e normalize edilir.
      5. per_expert_scale ile uzman bazlı ek ölçekleme uygulanır.

    Çıktılar:
      - router_probabilities: tüm uzman olasılıkları (eğitim/izleme için)
      - top_k_weights: seçilen uzmanların normalize ağırlıkları
      - top_k_index: seçilen uzman indeksleri
    """
    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = DiffusionGemmaRMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Token başına uzman seçimi ve ağırlıklandırma yapar.

        Args:
            hidden_states: [batch*seq, hidden_size] veya [batch, seq, hidden_size]
                düzleştirilmiş token temsilleri

        Returns:
            router_probabilities: [..., num_experts] — tüm uzman softmax olasılıkları
            top_k_weights: [..., top_k] — seçilen uzmanların normalize ağırlıkları
            top_k_index: [..., top_k] — seçilen uzman indeksleri
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1, dtype=torch.float32)

        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # ikisi de [B*S, K]

        # top-k ağırlıkları token başına 1'e normalize et
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # uzman bazlı ölçeklemeyi doğrudan ağırlıklara uygula
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class DiffusionGemmaTextExperts(nn.Module):
    """
    MoE uzman ağırlıklarının 3D tensör olarak saklandığı koleksiyon.

    Her uzman, bağımsız bir kapılı FFN'dir. Ağırlıklar nn.Parameter olarak
    [num_experts, ...] şeklinde tutulur; bu sayede tüm uzmanlar tek modülde
    vektörize edilebilir.

    İleri geçişte yalnızca router'ın seçtiği uzmanlar çalıştırılır (seyrek aktivasyon).
    Her token, top_k uzmanın çıktısının ağırlıklı toplamını alır.
    """
    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Seçilen uzmanları token bazında çalıştırır ve ağırlıklı toplam döner.

        Yalnızca en az bir token tarafından seçilen uzmanlar döngüye alınır;
        bu seyrek yürütme MoE'nin hesaplama maliyetini düşürür.

        Args:
            hidden_states: [num_tokens, hidden_dim] — düzleştirilmiş token girdileri
            top_k_index: [num_tokens, top_k] — her token için seçilen uzman indeksleri
            top_k_weights: [num_tokens, top_k] — uzman ağırlıkları

        Returns:
            [num_tokens, hidden_dim] — uzman çıktılarının ağırlıklı toplamı
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DiffusionGemmaEncoderTextLayer(GradientCheckpointingLayer):
    """
    DiffusionGemma encoder (denoiser bağlam encoder'ı) transformer katmanı.

    Standart transformer bloğuna ek olarak *çift feed-forward* yapısı vardır:
    sabit MLP ve MoE kolu paralel çalışır, çıktıları toplanır.

    Katman akışı (pre-norm + residual):
      1. input_layernorm → self-attention → post_attention_layernorm → residual
      2. pre_feedforward_layernorm → MLP → post_feedforward_layernorm_1  (MLP kolu)
      3. Paralel MoE kolu: residual (MLP öncesi) → router → experts → post_norm_2
      4. hidden = MLP_kolu + MoE_kolu → post_feedforward_layernorm → residual
      5. layer_scalar ile çıkış ölçeklenir

    Gemma4TextDecoderLayer'dan farklar:
      - PLE (paylaşımlı KV) kod yolu yok
      - shared_kv_states taşınmıyor
      - EncoderTextAttention kullanır (KV önbelleği güncellenir)
    """
    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaEncoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encoder katmanının tam ileri geçişi: attention + çift FFN (MLP + MoE).

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) RoPE katsayıları
            attention_mask: additive attention mask
            position_ids: token pozisyon kimlikleri
            past_key_values: KV önbelleği (encoder tarafında güncellenir)

        Returns:
            [batch, seq_len, hidden_size]
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # MoE yönlendirmesi için MLP öncesi residual durumunu kullan
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # MLP ve MoE çıktılarını birleştir
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaDecoderTextLayer(GradientCheckpointingLayer):
    """
    DiffusionGemma decoder transformer katmanı.

    Encoder katmanıyla aynı çift FFN (MLP + MoE) yapısına sahiptir; tek kritik
    fark attention modülüdür: DiffusionGemmaDecoderTextAttention kullanır.

    Decoder attention özellikleri:
      - Encoder KV önbelleğini salt okunur okur (çapraz dikkat)
      - Tuval tokenları arasında iki yönlü self-attention
      - past_key_values güncellenmez

    Gemma4TextDecoderLayer'dan farklar:
      - PLE kod yolu yok
      - shared_kv_states taşınmıyor
      - DecoderTextAttention ile encoder önbelleğine salt okunur erişim
    """
    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaDecoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Decoder katmanının tam ileri geçişi: çapraz dikkat + çift FFN (MLP + MoE).

        Args:
            hidden_states: [batch, canvas_len, hidden_size]
            position_embeddings: (cos, sin) RoPE katsayıları
            attention_mask: encoder önbelleği + tuval için mask
            position_ids: tuval token pozisyon kimlikleri
            past_key_values: encoder'ın salt okunur KV önbelleği

        Returns:
            [batch, canvas_len, hidden_size]
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # MoE yönlendirmesi için MLP öncesi residual durumunu kullan
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # MLP ve MoE çıktılarını birleştir
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaTextScaledWordEmbedding(nn.Embedding):
    """
    Öğrenilen gömme vektörlerini sqrt(hidden_size) ile ölçekleyen word embedding.

    Gemma ailesinde embedding çıkışı hidden_size^0.5 ile çarpılır; bu sayede
    gömme büyüklüğü transformer katmanlarıyla uyumlu kalır. nn.Embedding.forward
    sonucuna embed_scale tamponu uygulanır.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        """
        Token ID'lerini ölçeklenmiş gömme vektörlerine dönüştürür.

        Args:
            input_ids: [batch, seq_len] — token kimlikleri

        Returns:
            [batch, seq_len, embedding_dim]
        """
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class DiffusionGemmaMultimodalEmbedder(nn.Module):
    """
    Çok modlu (görüntü vb.) özellikleri dil modeli uzayına projekte eden katman.

    Vision tower çıktıları, önce RMSNorm ile normalize edilir, ardından Linear
    katmanı ile text hidden_size boyutuna projekte edilir. Böylece görüntü patch
    temsilleri, metin token gömmeleriyle aynı uzayda birleştirilebilir.
    """
    def __init__(
        self,
        multimodal_config: PreTrainedConfig,
        text_config: DiffusionGemmaTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = DiffusionGemmaRMSNorm(
            self.multimodal_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Çok modlu soft token gömmelerini dil modeli boyutuna projekte eder.

        Args:
            inputs_embeds: [batch, seq_len, multimodal_hidden_size] — vision tower çıktısı

        Returns:
            [batch, seq_len, text_hidden_size]
        """
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class DiffusionGemmaSelfConditioning(nn.Module):
    """
    Öz-koşullandırma (self-conditioning) modülü.

    Difüzyon decoder'ı, bir önceki gürültü giderme adımından gelen logits'leri
    soft embedding'e çevirip kendi girdisine ekler. Bu, modelin önceki tahmininden
    öğrenmesini sağlar (self-conditioning).

    Yapı: Gemma4 kapılı MLP (gate/up/down) + pre/post RMSNorm.
    Çıkış, decoder girdi gömmelerine eklenir.
    """
    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.pre_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, inputs_embeds, self_conditioning_signal: torch.Tensor) -> torch.Tensor:
        """
        Önceki adımın soft embedding'ini işleyip decoder girdisine ekler.

        Args:
            inputs_embeds: [batch, canvas_len, hidden_size] — tuval token gömmeleri
            self_conditioning_signal: [batch, canvas_len, hidden_size] — önceki adım logits'inden
                türetilmiş soft embedding (ilk adımda sıfır vektör)

        Returns:
            [batch, canvas_len, hidden_size] — öz-koşullandırılmış girdi gömmeleri
        """
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(self.act_fn(self.gate_proj(normed)) * self.up_proj(normed))
        combined = inputs_embeds + sc_signal
        return self.post_norm(combined)


class DiffusionGemmaEncoderTextModel(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma metin encoder gövdesi (saf dil modeli katmanları).

    Prompt/context tokenlarını işler, her katmanda KV önbelleğini günceller.
    Görüntü girdisi yoktur; çok modlu birleştirme DiffusionGemmaEncoderModel'de yapılır.

    Bileşenler: embed_tokens, N encoder katmanı, final RMSNorm, RoPE.
    """
    config: DiffusionGemmaTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # bfloat16 dönüşümü embed_scale'i hafif değiştirir; bkz. HF PR #29402
        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [DiffusionGemmaEncoderTextLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)

        # Ağırlıkları başlat ve son işleme uygula
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Metin encoder'ının tam ileri geçişi: gömme → katmanlar → norm → KV önbelleği.

        Args:
            input_ids: [batch, seq_len] — token kimlikleri (inputs_embeds ile birlikte verilemez)
            attention_mask: padding/causal mask veya önceden hazırlanmış dict
            position_ids: [batch, seq_len] — mutlak pozisyon kimlikleri
            past_key_values: mevcut KV önbelleği (yoksa DynamicCache oluşturulur)
            inputs_embeds: [batch, seq_len, hidden_size] — doğrudan gömme girdisi

        Returns:
            BaseModelOutputWithPast: last_hidden_state ve güncellenmiş past_key_values
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # generate() gibi çağrılarda mask zaten hazırlanmış olabilir
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def get_block_sequence_ids_for_mask(mm_token_type_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Çok modlu attention mask için görüntü blok kimliklerini hesaplar.

    Vision tokenları (tip 1 veya 2) ardışık gruplara ayrılır; her grup bir blok
    kimliği alır. Metin tokenları -1 ile işaretlenir (blok dışı).

    Args:
        mm_token_type_ids: [batch, seq_len] — 0=metin, 1/2=görüntü token tipi
        device: hedef cihaz

    Returns:
        [batch, seq_len] — blok kimlikleri (-1 metin, >=0 görüntü blokları)
    """
    mm_token_type_ids = mm_token_type_ids.to(device)

    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    block_sequence_ids = torch.where(is_vision, vision_group_ids, -1)
    return block_sequence_ids


class DiffusionGemmaEncoderModel(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma encoder modeli — metin + görüntü birleştirme ve KV önbelleği üretimi.

    Prompt tokenlarını (ve isteğe bağlı görüntü patch'lerini) işleyerek decoder'ın
    kullanacağı KV önbelleğini oluşturur. Vision tower çıktıları, placeholder
    tokenların yerine scatter edilir.

    Alt bileşenler:
      - language_model: saf metin encoder gövdesi
      - vision_tower: görüntü özellik çıkarıcı
      - embed_vision: görüntü özelliklerini dil uzayına projekte eder
    """
    # logits/labels filtrelenir; kayıp num_items_in_batch'e bölünmez
    accepts_loss_kwargs = False
    config: DiffusionGemmaConfig
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size

        self.language_model = DiffusionGemmaEncoderTextModel(config=config.text_config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.embed_vision = DiffusionGemmaMultimodalEmbedder(config.vision_config, config.text_config)

        # Ağırlıkları başlat ve son işleme uygula
        self.post_init()
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        """
        Görüntü piksel değerlerinden dil modeli uzayında özellik çıkarır.

        Args:
            pixel_values: [batch, channels, height, width] — ön işlenmiş görüntü
            image_position_ids: [batch, max_patches, 2] — patch (x,y) koordinatları;
                padding patch'ler (-1, -1) ile gösterilir

        Returns:
            Vision tower çıktısı; pooler_output alanına projekte edilmiş gömmeler yazılır
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        vision_outputs.pooler_output = self.embed_vision(inputs_embeds=last_hidden_state)
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Çok modlu placeholder (görüntü) tokenlarının maskesini üretir.

        input_ids veya inputs_embeds'ten hangisi verilmişse ondan türetilir;
        input_ids önceliklidir.

        Args:
            input_ids: [batch, seq_len] — tokenizer'dan gelen sert token ID'leri
            inputs_embeds: [batch, seq_len, hidden_size] — gömme temsilleri

        Returns:
            [batch, seq_len] — True olan pozisyonlar görüntü placeholder'ıdır
        """
        if input_ids is not None:
            special_image_mask = input_ids == self.config.image_token_id
        else:
            image_token_embeddings = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embeddings).all(-1)

        return special_image_mask
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Encoder'ın tam ileri geçişi: metin+görüntü birleştirme ve KV önbelleği üretimi.

        Args:
            input_ids: [batch, seq_len] — prompt token kimlikleri
            pixel_values: [batch, C, H, W] — görüntü piksel değerleri (opsiyonel)
            attention_mask: encoder attention mask'i
            position_ids: token pozisyon kimlikleri
            past_key_values: mevcut KV önbelleği
            mm_token_type_ids: çok modlu token tip kimlikleri (görüntü mask için)
            inputs_embeds: doğrudan gömme girdisi (input_ids ile birlikte verilemez)
            image_position_ids: [batch, max_patches, 2] — görüntü patch koordinatları

        Returns:
            BaseModelOutputWithPast: encoder çıktısı ve güncellenmiş KV önbelleği
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds)

        # OOV görüntü token'ını PAD ile değiştir (index hatası önleme)
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Metin ve görüntü gömmelerini birleştir
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Vision tower soft token sayısı ile placeholder slot sayısını doğrula
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        # generate() gibi çağrılarda position_ids zaten hazırlanmış olabilir
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            self.create_masks_for_generate(
                config=self.config.get_text_config(),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_per_layer_input_embeddings(self):
        return self.language_model.embed_tokens_per_layer

    def set_per_layer_input_embeddings(self, value):
        self.language_model.embed_tokens_per_layer = value

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> dict:
        """
        Üretim (generate) için encoder attention mask'lerini oluşturur.

        Görüntü girdisi varsa ve use_bidirectional_attention="vision" ise,
        görüntü blokları için çift yönlü mask kullanılır; aksi halde klasik
        causal mask uygulanır.

        Args:
            config: model konfigürasyonu
            inputs_embeds: [batch, seq_len, hidden_size]
            attention_mask: padding mask
            past_key_values: KV önbelleği
            position_ids: pozisyon kimlikleri
            mm_token_type_ids: çok modlu token tipleri

        Returns:
            full_attention ve sliding_attention mask dict'i
        """
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        # Büyük Gemma 4 modelleri görüntü girdileri için çift yönlü mask kullanır
        if getattr(config.get_text_config(), "use_bidirectional_attention", None) == "vision":
            block_sequence_ids = torch.full([*inputs_embeds.size()[:-1]], -1, device=inputs_embeds.device)
            if mm_token_type_ids is not None:
                block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)

            mask_kwargs["block_sequence_ids"] = block_sequence_ids

        return create_masks_for_generate(**mask_kwargs)


class DiffusionGemmaDecoderModel(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma decoder modeli — tuval gürültü giderme ve encoder çapraz dikkat.

    Tuval (canvas) tokenlarını işler: iki yönlü self-attention + encoder KV
    önbelleğine salt okunur çapraz dikkat. Decoder KV önbelleğini güncellemez.

    EncoderTextModel ile ortak ağırlıkları paylaşır; ek olarak self_conditioning
    modülü vardır. Çıktıda past_key_values döndürülmez.
    """
    config: DiffusionGemmaConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaDecoderTextLayer,
        "attentions": DiffusionGemmaDecoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.text_config = config.text_config
        self.padding_idx = config.text_config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.text_config.hidden_size,
            padding_idx=self.padding_idx,
            embed_scale=config.text_config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                DiffusionGemmaDecoderTextLayer(config.text_config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
        self.norm = DiffusionGemmaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config.text_config)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config.text_config)
        self.unique_layer_types = set(config.text_config.layer_types)

        # Ağırlıkları başlat ve son işleme uygula
        self.post_init()
    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        """
        Decoder'ın tam ileri geçişi: tuval gömme → öz-koşullandırma → katmanlar → norm.

        Args:
            decoder_input_ids: [batch, canvas_len] — tuval token kimlikleri
            past_key_values: encoder'ın salt okunur KV önbelleği (zorunlu)
            self_conditioning_logits: [batch, canvas_len, vocab_size] — önceki adım logits'leri
            self_conditioning_mask: [batch] — örnek bazında öz-koşullandırma açma/kapama
            decoder_attention_mask: encoder önbelleği + tuval için attention mask
            decoder_position_ids: [batch, canvas_len] — tuval pozisyon kimlikleri

        Returns:
            BaseModelOutput: tuval son katman gizli durumları (KV önbelleği döndürülmez)
        """
        if "use_cache" in kwargs:
            raise ValueError(
                "The decoder of DiffusionGemma always uses a cache, so it doesn't accept the `use_cache` argument"
            )

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # Öz-koşullandırma yoksa sıfır vektör (ilk gürültü giderme adımı)
        if self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(soft_embeddings.dtype)[:, None, None]
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        # Decoder pozisyonları encoder dizisinin devamından başlar
        if decoder_position_ids is None:
            canvas_length = inputs_embeds.shape[1]
            cache_seq_length = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            decoder_position_ids = decoder_position_ids.unsqueeze(0)

        if not isinstance(mask_mapping := decoder_attention_mask, dict):
            mask_mapping = self.create_diffusion_decoder_attention_mask(
                config=self.text_config,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, decoder_position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.text_config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.text_config.layer_types[i]],
                attention_mask=mask_mapping[self.text_config.layer_types[i]],
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # Decoder KV önbelleği üretmez
        return BaseModelOutput(last_hidden_state=hidden_states)

    @staticmethod
    def create_diffusion_decoder_attention_mask(
        config: DiffusionGemmaTextConfig,
        inputs_embeds: torch.Tensor,
        past_key_values: Cache,
        decoder_attention_mask: torch.Tensor | dict | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Decoder için iki yönlü attention mask'i oluşturur.

        Decoder mask'inin toplam uzunluğu = encoder KV önbellek uzunluğu + tuval uzunluğu.
        Mask iki bölümden oluşur:

        1. Encoder KV bölümü: AR modeldeki gibi çalışır; sol/sağ padding olabilir.
           Tuval tokenları yalnızca geçerli (1 olan) önbellek pozisyonlarına bakabilir.
        2. Tuval bölümü: her zaman 1 — tuval tokenları birbirlerine tam erişimlidir.

        Manuel decoder_attention_mask kullanımı için kurallar:
          - Şekil: (batch_size, önbellek_uzunluğu + canvas_uzunluğu)
          - Son canvas_uzunluğu pozisyon her zaman 1 olmalı

        Örnek (batch=2, önbellek=8, tuval=4):
        Dizi 0: 4 dolu token + 4 boş slot; dizi 1: 2 sol-padding + 2 dolu + 4 boş.
        Encoder KV mask (■=1, ⬚=0):

        [0, *] ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚
        [1, *] ⬚ ⬚ ■ ■ ⬚ ⬚ ⬚ ⬚

        Tuval bidirectional bölümü eklendikten sonra son 4 sütun hep ■ olur.

        Sliding attention katmanları için mask, dolu önbellek pozisyonlarına göre
        sağdan dilimlenir ve tuval mask'i pad ile eklenir.

        Not: create_bidirectional_mask gibi genel yardımcılar burada kullanılmaz;
        sliding window hesaplamasında AR varsayımları decoder senaryosuna uymaz.

        Args:
            config: metin model konfigürasyonu
            inputs_embeds: [batch, canvas_len, hidden_dim] — boyut bilgisi için
            past_key_values: encoder'ın ürettiği KV önbelleği
            decoder_attention_mask: [batch, önbellek+tuval] padding mask (opsiyonel)

        Returns:
            full_attention ve sliding_attention için 4D bool mask dict'i (veya None)
        """

        # create_bidirectional_mask AR varsayımları içerir; decoder için özel mantık gerekir
        batch_size, canvas_length, _ = inputs_embeds.shape

        if past_key_values is None:
            raise ValueError(
                "`past_key_values` must be a `Cache` instance in `create_diffusion_decoder_attention_mask`."
            )
        if past_key_values.is_compileable and decoder_attention_mask is None:
            raise ValueError(
                "When `past_key_values` is a compileable cache, i.e. a static-shaped cache, `decoder_attention_mask` "
                "must be set."
            )
        # Kısayol: derleme yok VE padding yok → iç fonksiyonlara None döndür
        if decoder_attention_mask is None or (not past_key_values.is_compileable and decoder_attention_mask.all()):
            return {"full_attention": None, "sliding_attention": None}

        # Padding ve/veya derleme varsa tam mask'i somutlaştır
        valid_cache_tokens = past_key_values.get_seq_length()
        if past_key_values.is_compileable:
            full_cache_kv_length = past_key_values.max_cache_len
        else:
            full_cache_kv_length = valid_cache_tokens
        full_kv_length = full_cache_kv_length + canvas_length
        if decoder_attention_mask.shape != (batch_size, full_kv_length):
            raise ValueError(
                "When set, `decoder_attention_mask` must have the length = cache length + canvas length."
                f" Got `decoder_attention_mask` with length {decoder_attention_mask.shape[1]} "
                f"(!= {full_cache_kv_length} + {canvas_length})"
            )
        if (decoder_attention_mask.sum(dim=-1) > valid_cache_tokens + canvas_length).any():
            raise ValueError(
                "Your `decoder_attention_mask` has more 1s than there are cached + canvas tokens. "
                "There is one or more rows in the `decoder_attention_mask` with "
                f"{decoder_attention_mask.sum(dim=-1).max()} 1s, while there are at most "
                f"{valid_cache_tokens + canvas_length} tokens to be processed in each "
                "row. If you're using a static cache, don't forget to set empty positions to 0."
            )

        # 2D [batch, full_kv] → 4D [batch, 1, canvas_len, full_kv]
        full_mask = decoder_attention_mask[:, None, None, :].bool()
        full_mask = full_mask.expand(batch_size, 1, canvas_length, full_kv_length)

        # Sliding window: tam mask'in sağ dilimini al
        sliding_cache_is_full = valid_cache_tokens >= config.sliding_window
        if sliding_cache_is_full:
            # Derlenebilir önbellekte sliding katman 1 eleman daha uzun olabilir
            if past_key_values.is_compileable:
                sliding_start_idx = valid_cache_tokens - config.sliding_window
            else:
                sliding_start_idx = valid_cache_tokens - config.sliding_window + 1
            sliding_end_idx = valid_cache_tokens
        else:
            sliding_start_idx = 0
            if past_key_values.is_compileable:
                sliding_end_idx = min(config.sliding_window, past_key_values.max_cache_len)
            else:
                sliding_end_idx = valid_cache_tokens
        sliding_mask = full_mask[..., sliding_start_idx:sliding_end_idx]
        # Tuval bidirectional mask'ini sağa ekle
        sliding_mask = torch.nn.functional.pad(sliding_mask, (0, canvas_length), value=True)

        return {"full_attention": full_mask, "sliding_attention": sliding_mask}


@dataclass
class DiffusionGemmaModelOutputWithPast(BaseModelOutputWithPast):
    """
    DiffusionGemmaModel çıktı konteyneri.

    Standart BaseModelOutputWithPast alanlarına ek olarak encoder son katman
    gizli durumunu taşır (eğitimde encoder kaybı hesaplamak için).

    Alanlar:
        last_hidden_state: decoder tuval çıktısı
        past_key_values: encoder KV önbelleği
        hidden_states: ara katman durumları (opsiyonel)
        attentions: attention ağırlıkları (opsiyonel)
        encoder_last_hidden_state: encoder son katman çıktısı (input_ids verilmişse)
    """

    encoder_last_hidden_state: torch.FloatTensor | None = None


@dataclass
class DiffusionGemmaBlockDiffusionOutputWithPast(CausalLMOutputWithPast):
    """
    Blok difüzyon dil modeli çıktı konteyneri.

    Tuval üzerindeki logits, gizli durumlar ve encoder çıktısını bir arada döner.

    Alanlar:
        loss: dil modelleme kaybı (opsiyonel)
        logits: [batch, canvas_len, vocab_size] — softcapping öncesi skorlar
        past_key_values: encoder KV önbelleği
        hidden_states: ara katman durumları (opsiyonel)
        attentions: attention ağırlıkları (opsiyonel)
        encoder_last_hidden_state: encoder son katman çıktısı (opsiyonel)
    """

    encoder_last_hidden_state: torch.FloatTensor | None = None


class DiffusionGemmaModel(DiffusionGemmaPreTrainedModel):
    """
    DiffusionGemma tam modeli — encoder + difüzyon decoder.

    İki aşamalı mimari:
      1. Encoder (DiffusionGemmaEncoderModel): prompt/context'i işler, KV önbelleği üretir.
         Gemma4Model'e çok benzer; görüntü desteği vardır.
      2. Decoder (DiffusionGemmaDecoderModel): tuval tokenlarını gürültü giderme ile işler.

    Klasik encoder-decoder'dan kritik fark:
      Encoder, decoder'a gizli durumları değil yalnızca KV önbelleğini verir.
      Decoder bu önbelleği salt okunur kullanır (çapraz dikkat).

    Ağırlık paylaşımı: encoder ve decoder metin katmanları ortak ağırlıkları paylaşır;
    self_conditioning yalnızca decoder'da bulunur.
    """
    # Encoder metin ağırlıkları decoder'da da var; self-conditioning yalnızca decoder'da
    _tied_weights_keys = {
        "encoder.language_model.norm.weight": "decoder.norm.weight",
        # Aşağıdaki satırlar katman ağırlıklarını bağlar; buffer'lar (layer_scalar vb.) bağlanmaz
        r"encoder.language_model.layers\.(?:[^.]+\.)*weight": r"decoder.layers\.(?:[^.]+\.)*weight",
        r"encoder.language_model.layers\.(?:[^.]+\.)*scale": r"decoder.layers\.(?:[^.]+\.)*scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*per_expert_scale": r"decoder.layers\.(?:[^.]+\.)*per_expert_scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*gate_up_proj": r"decoder.layers\.(?:[^.]+\.)*gate_up_proj",
        r"encoder.language_model.layers\.(?:[^.]+\.)*down_proj": r"decoder.layers\.(?:[^.]+\.)*down_proj",
        "encoder.language_model.embed_tokens.weight": "decoder.embed_tokens.weight",
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)

        self.encoder = DiffusionGemmaEncoderModel(config)
        self.decoder = DiffusionGemmaDecoderModel(config)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaModelOutputWithPast:
        """
        DiffusionGemma tam model ileri geçişi: encode → decode.

        İki aşama:
          1. Encoder: input_ids varsa prompt'u KV önbelleğine yazar.
             Yoksa mevcut past_key_values kullanılır (en az biri zorunlu).
          2. Decoder: tuval tokenlarını işler; encoder KV önbelleğine çapraz dikkat
             + tuval içi iki yönlü self-attention uygular.

        decoder_input_ids verilmezse tuval, vocab'tan uniform rastgele örneklenir
        (difüzyon başlangıç durumu).

        Args:
            input_ids: [batch, seq_len] — encode edilecek prompt tokenları
            attention_mask: encoder attention mask'i
            past_key_values: mevcut KV önbelleği
            position_ids: encoder pozisyon kimlikleri
            decoder_input_ids: [batch, canvas_len] — tuval tokenları
            self_conditioning_logits: önceki adım logits'leri
            self_conditioning_mask: örnek bazında öz-koşullandırma maskesi
            decoder_attention_mask: decoder attention mask'i
            decoder_position_ids: tuval pozisyon kimlikleri

        Returns:
            DiffusionGemmaModelOutputWithPast: decoder çıktısı + encoder KV önbelleği
        """

        # 1: Yeni prompt tokenlarını KV önbelleğine encode et
        encoder_last_hidden_state = None
        if input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            encoder_last_hidden_state = encoder_outputs.last_hidden_state
        elif past_key_values is None:
            raise ValueError("Either `input_ids` or `past_key_values` must be provided.")

        # 2: Decoder — tuval self-attention + encoder KV çapraz dikkat
        if decoder_input_ids is None:
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.text_config.vocab_size,
                size=(input_ids.shape[0], self.config.canvas_length),
                device=self.decoder.device,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        return DiffusionGemmaModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )


class DiffusionGemmaForBlockDiffusion(DiffusionGemmaPreTrainedModel, DiffusionGemmaGenerationMixin):
    """
    Blok difüzyon için DiffusionGemma dil modeli başlığı (LM Head).

    DiffusionGemmaModel'i çağırarak prompt KV önbelleği koşullu tuval gizli
    durumlarını üretir; lm_head ile vocab logits'lerine dönüştürür.

    Blok difüzyon döngüsü:
      1. Encoder prompt'u KV önbelleğine yazar
      2. Decoder gürültülü tuvali işler (öz-koşullandırma ile)
      3. lm_head logits üretir → softcapping uygulanır
      4. Logits bir sonraki gürültü giderme adımında self_conditioning_logits olarak
         decoder'a geri beslenir

    lm_head ağırlıkları decoder embed_tokens ile tied (paylaşımlı).
    """
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}
    generation_config_class = DiffusionGemmaGenerationConfig

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)

        self.model = DiffusionGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

        # Ağırlıkları başlat ve son işleme uygula
        self.post_init()

    def get_input_embeddings(self):
        return self.model.encoder.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.encoder.language_model.set_input_embeddings(value)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaBlockDiffusionOutputWithPast:
        """
        Blok difüzyon ileri geçişi: model → lm_head → softcapped logits.

        Args:
            input_ids: [batch, seq_len] — prompt tokenları
            attention_mask: encoder mask
            past_key_values: KV önbelleği
            position_ids: encoder pozisyon kimlikleri
            decoder_input_ids: [batch, canvas_len] — tuval tokenları
            self_conditioning_logits: önceki adım logits'leri
            self_conditioning_mask: öz-koşullandırma maskesi
            decoder_attention_mask: decoder mask
            decoder_position_ids: tuval pozisyon kimlikleri

        Returns:
            DiffusionGemmaBlockDiffusionOutputWithPast: softcapped logits ve model çıktıları
        """

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        # Logits hesapla ve Gemma softcapping uygula (aşırı büyük logitleri sınırla)
        logits = self.lm_head(model_outputs.last_hidden_state)
        logits = logits.to(torch.float32)
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping

        return DiffusionGemmaBlockDiffusionOutputWithPast(
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            past_key_values=model_outputs.past_key_values,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
        )