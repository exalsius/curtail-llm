from nanochat.gpt import GPT, GPTConfig


def find_num_heads(model_dim, target_head_dim):
    """Find num_heads that divides model_dim evenly, with head_dim closest to target."""
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1


def get_model(model_config: dict, max_seq_len: int):
    """Create a GPT model from a configuration dictionary."""
    n_layer = model_config["n_layer"]
    aspect_ratio = model_config["aspect_ratio"]
    head_dim = model_config["head_dim"]

    # Derive model parameters
    n_embd = n_layer * aspect_ratio
    n_head = find_num_heads(n_embd, head_dim)
    n_kv_head = n_head  # Default to GQA ratio of 1

    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=model_config["vocab_size"],
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
    )

    model = GPT(config)
    model.init_weights()
    return model
