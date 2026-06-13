"""
DiffusionGemma model hiperparametreleri.

Bu modül, HuggingFace `configuration_diffusion_gemma.py` dosyasının
öğrenme amaçlı sadeleştirilmiş halidir. Sınıf değişkenleri doğrudan
varsayılan checkpoint değerlerini taşır; ayrı bir __init__ yoktur.

Öğrenciler için ipucu: Bu sayılar modelin bellek kullanımını, hızını ve
kapasitesini belirler. Örneğin hidden_size=2304, her token için 2304
boyutlu bir vektör anlamına gelir.
"""

from transformers.models.auto.configuration_auto import AutoConfig


class DiffusionGemmaTextConfig:
    """
    Metin gövdesi (encoder + decoder ortak text backbone) yapılandırması.

    DiffusionGemma'nın metin tarafı Gemma 4 mimarisine dayanır. Katmanların
    bir kısmı sliding window attention (yerel, 512 token pencere), son
    katman(lar) ise full/global attention (tüm diziye bakış, daha büyük head)
    kullanır.

    Önemli kavramlar
    ----------------
    GQA (Grouped Query Attention):
        num_attention_heads=8 sorgu head'i, num_key_value_heads=4 K/V head'i.
        Her K/V head'i 2 Q head'ine hizmet eder → bellek tasarrufu.

    Sliding vs Full attention:
        sliding: head_dim=256, pencere=512 — uzun dizilerde verimli
        full: global_head_dim=512 — geniş bağlam, daha fazla parametre

    use_bidirectional_attention:
        "all"  → encoder'da tüm tokenlar birbirini görür (difüzyon eğitimi)
        "vision" → görüntü tokenları çift yönlü, metin causal kalır
        None   → varsayılan davranış (genelde causal)

    MoE (Mixture of Experts):
        top_k_experts kadar expert her token için aktif edilir. Router skorları
        hangi expert'lerin çalışacağını belirler; kalan expert'ler atlanır.
    """

    # --- Kelime dağarcığı ve gömme ---
    vocab_size = 262_144
    """Tokenizer kelime dağarcığı boyutu. Her token 0..vocab_size-1 arası bir ID."""

    hidden_size = 2304
    """Ana gizli boyut d. Embedding, attention giriş/çıkışı ve FFN bu boyutta."""

    tie_word_embeddings = True
    """True ise giriş embedding matrisi ile lm_head ağırlıkları paylaşılır."""

    # --- Transformer gövdesi ---
    num_hidden_layers = 30
    """Toplam transformer katman sayısı (encoder ve decoder aynı ağırlıkları paylaşır)."""

    num_attention_heads = 8
    """Her katmandaki Query head sayısı."""

    num_key_value_heads = 4
    """Sliding katmanlardaki Key/Value head sayısı (GQA)."""

    head_dim = 256
    """Sliding attention katmanlarında head başına boyut."""

    global_head_dim = 512
    """Full/global attention katmanlarında head başına boyut (2× sliding)."""

    num_global_key_value_heads = None
    """Full katmanlarda K/V head sayısı. None → num_key_value_heads kullanılır."""

    intermediate_size = 9216
    """FFN ara katman genişliği. Gemma'da tipik olarak ~4 × hidden_size."""

    hidden_activation = "relu_pytorch_tanh"
    """FFN aktivasyonu: ReLU sonra tanh (Gemma 2+ standartı)."""

    attention_bias = False
    """Q/K/V/O lineer katmanlarında bias kullanılmaz."""

    attention_dropout = 0.0
    """Attention softmax sonrası dropout oranı."""

    # --- Pozisyon kodlaması ---
    max_position_embeddings = 131_072
    """Desteklenen maksimum dizi uzunluğu (128K token)."""

    rope_parameters = None
    """
    Katman tipine göre RoPE ayarları sözlüğü.
    Örnek anahtarlar: "full_attention", "sliding_attention"
    Her biri rope_theta, rope_type vb. içerir.
    """

    sliding_window = 512
    """Sliding attention penceresi: her token en fazla 512 komşuya bakar."""

    # --- Katman tipi dağılımı ---
    layer_types = None
    """
    Her katmanın attention tipi listesi, uzunluk = num_hidden_layers.
    Örnek: 29× "sliding_attention" + 1× "full_attention"
    """

    use_bidirectional_attention = None
    """Encoder attention yönü; bkz. sınıf docstring'i."""

    # --- MoE ---
    num_experts = None
    """Toplam expert sayısı (MoE katmanlarında)."""

    top_k_experts = None
    """Her token için router'ın seçtiği expert sayısı."""

    moe_intermediate_size = None
    """Her expert'in FFN gizli boyutu (dense FFN'den bağımsız olabilir)."""

    # --- Normalizasyon ve çıkış ---
    rms_norm_eps = 1e-6
    """RMSNorm sayısal kararlılık epsilon'u."""

    final_logt_softcapping = 30.0
    """
    Logits softcapping sınırı. logits = tanh(logits/30)*30 ile aşırı büyük
    değerler yumuşatılır; eğitim/çıkarım stabilitesi için.
    """

    initializer_range = 0.02
    """Ağırlık başlatmada Gaussian std sapması."""

    # --- Özel tokenlar ---
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2


class DiffusionGemmaConfig:
    """
    Tam DiffusionGemma yapılandırması: metin + görüntü + difüzyon tuvali.

    Mimari özeti
    ------------
    ┌─────────────────────────────────────────────────────────┐
    │  DiffusionGemmaConfig                                   │
    │  ├── text_config   → DiffusionGemmaTextConfig           │
    │  ├── vision_config → görüntü encoder (SigLIP vb.)       │
    │  └── canvas_length → difüzyon tuvali token sayısı       │
    └─────────────────────────────────────────────────────────┘

    Çok modlu akış:
        1. Görüntü → vision_tower → embed_vision → metin uzayına projeksiyon
        2. Metin + görüntü tokenları encoder'da işlenir → KV cache
        3. canvas_length boyutunda gürültülü tuval decoder'da refine edilir

    Özel görüntü tokenları:
        boi_token_id / eoi_token_id: görüntü bloğunun sınırları
        image_token_id: placeholder; gerçek görüntü embedding'i ile değiştirilir
    """

    sub_configs = {
        "text_config": DiffusionGemmaTextConfig,
        "vision_config": AutoConfig,
    }
    text_config = None
    vision_config = None

    boi_token_id = 255_999
    """Begin-of-image: görüntü prompt bloğunun başlangıç tokeni."""

    eoi_token_id = 258_882
    """End-of-image: görüntü prompt bloğunun bitiş tokeni."""

    image_token_id = 258_880
    """Görüntü yuvası placeholder tokeni; forward'da soft embedding ile doldurulur."""

    initializer_range = 0.02
    tie_word_embeddings = True

    canvas_length = 256
    """
    Difüzyon tuvali uzunluğu (token cinsinden).
    Blok difüzyonda bir seferde refine edilen alanın boyutudur.
    generate() döngüsü bu uzunlukta canvas_ids üzerinde çalışır.
    """
