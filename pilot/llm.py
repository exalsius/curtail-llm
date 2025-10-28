"""LLM models, data loading, and training with LoRA fine-tuning."""

import itertools
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from pilot.data import ShardedDataset, DATASET_SIZES


# ============================================================================
# Models
# ============================================================================

def get_model(model_name, load_in_8bit=False, load_in_4bit=False, **kwargs):
    """Load HuggingFace LLM model."""
    # Quantization config
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        except ImportError:
            print("Warning: bitsandbytes not installed. Loading in full precision.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
        **kwargs
    )
    return model


def apply_lora(model, lora_config=None):
    """Apply LoRA adapters to model."""
    if lora_config is None:
        lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }
    peft_config = LoraConfig(**lora_config)
    return get_peft_model(model, peft_config)


def get_trainable_params(model):
    """Get trainable parameter statistics."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": 100 * trainable / total,
    }


# ============================================================================
# Data Loading & Tokenization
# ============================================================================

class TokenizedDataLoader:
    """Wraps raw dataloader with tokenization."""

    def __init__(self, raw_loader, tokenizer, max_length):
        self.raw_loader = raw_loader
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for batch in self.raw_loader:
            # Extract texts from batch
            texts = self._extract_texts(batch)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            yield tokenized

    def _extract_texts(self, batch):
        """Extract text from various dataset formats."""
        if "messages" in batch:
            # Chat format
            return [self._format_chat(msgs) for msgs in batch["messages"]]
        elif "instruction" in batch and "output" in batch:
            # Alpaca format
            return [f"Instruction: {inst}\n\nResponse: {out}"
                    for inst, out in zip(batch["instruction"], batch["output"])]
        elif "text" in batch:
            return batch["text"]
        else:
            raise ValueError(f"Unknown dataset format: {batch.keys()}")

    def _format_chat(self, messages):
        """Format chat messages into single string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted)


def get_train_loader(dataset_name, shard_id, num_shards, processed_batches,
                     batch_size, model_type, max_length=512):
    """Get training dataloader for LLM dataset with tokenization."""
    from datasets import load_dataset

    # LLM datasets always use streaming
    use_streaming = dataset_name in DATASET_SIZES

    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=use_streaming)

    # Create sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=dataset_name if use_streaming else None,
        streaming=use_streaming,
    )

    raw_loader = DataLoader(sharded_dataset, batch_size=batch_size)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap with tokenization
    return TokenizedDataLoader(raw_loader, tokenizer, max_length)


# ============================================================================
# Training
# ============================================================================

def train(model, trainloader, num_batches, lr, device, weight_decay=0.01,
          gradient_accumulation_steps=1):
    """Train LLM with LoRA."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    total_loss = 0.0
    batches_processed = 0

    pbar = tqdm(enumerate(trainloader), total=num_batches, desc="Training")

    for batch_idx, batch in pbar:
        if batches_processed >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        batches_processed += 1
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, batches_processed


# ============================================================================
# Federated Learning Client Handler
# ============================================================================

def train_client(msg, config, context):
    """Handle LLM training on federated client.

    Returns:
        tuple: (model_state_dict, train_loss, batches_processed)
    """
    model_type = context.run_config["model_type"]
    dataset_name = config["dataset_name"]
    shard_id = config["shard_id"]
    num_shards = config["num_shards"]
    processed_batches = config["processed_batches"]
    batch_size = context.run_config["batch_size"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.01)
    max_length = context.run_config.get("max_length", 512)
    gradient_accumulation_steps = context.run_config.get("gradient_accumulation_steps", 1)
    lora_config = context.run_config.get("lora_config", None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Determine number of batches
    import random
    num_batches = 80 + int(40 * random.random())

    # Load base model
    base_model = get_model(model_type)

    # Apply LoRA
    model = apply_lora(base_model, lora_config)

    # Load weights if provided
    if msg.content["arrays"] is not None:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)

    model.to(device)

    # Get data
    trainloader = get_train_loader(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        model_type=model_type,
        max_length=max_length,
    )

    # Train
    train_loss, batches_processed = train(
        model=model,
        trainloader=trainloader,
        num_batches=num_batches,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    return model.state_dict(), train_loss, batches_processed
