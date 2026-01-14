from nanochat.gpt import GPT, GPTConfig


def get_model(model_config: dict, max_seq_len: int):
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=model_config["vocab_size"],
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_kv_head=model_config["n_kv_head"],
        n_embd=model_config["n_embd"],
    )
    
    model = GPT(config)
    model.init_weights()
    return model
