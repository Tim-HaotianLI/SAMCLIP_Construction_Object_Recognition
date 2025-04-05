Transformer(
  (resblocks): Sequential(
    (0): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (1): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (2): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (3): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (4): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (5): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (6): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (7): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (8): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (9): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (10): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (11): ResidualAttentionBlock(
      (attn): MultiheadAttention(
        (out_proj): _LinearWithBias(in_features=768, out_features=768, bias=True)
      )
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (c_fc): Linear(in_features=768, out_features=3072, bias=True)
        (gelu): QuickGELU()
        (c_proj): Linear(in_features=3072, out_features=768, bias=True)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
  )
)