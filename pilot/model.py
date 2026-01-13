from nanochat.gpt import GPT, GPTConfig


def get_model(model_type, max_seq_len):
    if model_type == "nanochat_d20":
        config = GPTConfig(
            sequence_len=max_seq_len,
            vocab_size=65536,
            n_layer=20,
            n_head=10,
            n_kv_head=10,
            n_embd=1280,
        )
    elif model_type == "nanochat_d32":
        config = GPTConfig(
            sequence_len=max_seq_len,
            vocab_size=65536,
            n_layer=32,
            n_head=16,
            n_kv_head=16,
            n_embd=2048,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = GPT(config)
    model.init_weights()
    return model
