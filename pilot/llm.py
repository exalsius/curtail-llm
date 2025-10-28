"""LLM models, data loading, and training with LoRA fine-tuning."""

import itertools
import random

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from pilot.data import ShardedDataset


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
            # UltraChat format: always use chat template from the tokenizer
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer must support apply_chat_template for UltraChat datasets")
            msgs_list = batch["messages"]
            return [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for msgs in msgs_list
            ]
        raise ValueError("Unsupported dataset sample format for UltraChat — expected 'messages' key")

    def _format_chat(self, messages):
        """Format chat messages into single string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted)


def _ultrachat_splits(dataset_name: str) -> tuple[str, str]:
    """Return the exact splits for UltraChat SFT. Raise for other datasets."""
    if dataset_name != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("This codepath is specialized for HuggingFaceH4/ultrachat_200k only")
    return "train_sft", "test_sft"


def get_train_loader(dataset_name, shard_id, num_shards, processed_batches,
                     batch_size, model_type, max_length=512):
    """Get training dataloader for LLM dataset with tokenization."""
    from datasets import load_dataset

    # Use map-style datasets (Arrow, memory-mapped). No streaming.
    use_streaming = False

    # Load dataset (map-style) with UltraChat SFT split
    train_split, _ = _ultrachat_splits(dataset_name)
    dataset = load_dataset(dataset_name, split=train_split, streaming=use_streaming)

    # Create sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=None,
        streaming=use_streaming,
    )

    # Collate into dict of lists (avoids default tensor stacking on variable-length fields)
    def collate_to_lists(samples):
        if not samples:
            return {}
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}

    raw_loader = DataLoader(sharded_dataset, batch_size=batch_size, collate_fn=collate_to_lists)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap with tokenization
    return TokenizedDataLoader(raw_loader, tokenizer, max_length)


def get_eval_loader(dataset_name, batch_size, model_type, max_length=512):
    """Get evaluation dataloader (non-sharded) for LLM dataset with tokenization."""
    from datasets import load_dataset

    # Map-style datasets
    use_streaming = False

    _, eval_split = _ultrachat_splits(dataset_name)
    dataset = load_dataset(dataset_name, split=eval_split, streaming=use_streaming)

    # Wrap as a simple PyTorch DataLoader with dict-of-lists collation
    def collate_to_lists(samples):
        if not samples:
            return {}
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}
    raw_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_to_lists)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


@torch.no_grad()
def evaluate(model, evalloader, num_batches, device):
    """Evaluate model on a capped number of eval batches and return average loss."""
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch_idx, batch in enumerate(tqdm(evalloader, total=num_batches, desc="Evaluating")):
        if batches >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        batches += 1

    return (total_loss / batches) if batches > 0 else 0.0


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

    if dataset_name != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("LLM train_client is specialized for HuggingFaceH4/ultrachat_200k")

    num_batches = 1000 - 200 + int(400 * random.random())

    base_model = get_model(model_type)
    model = apply_lora(base_model, lora_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(device)

    trainloader = get_train_loader(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        model_type=model_type,
        max_length=max_length,
    )

    train_loss, batches_processed = train(
        model=model,
        trainloader=trainloader,
        num_batches=num_batches,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Compute perplexity from training loss
    train_ppl = float(torch.exp(torch.tensor(train_loss)).item()) if train_loss > 0 else float("inf")

    # Quick eval for progress tracking on UltraChat test_sft
    evalloader = get_eval_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        model_type=model_type,
        max_length=max_length,
    )
    # Cap eval to a moderate number of batches
    eval_batches = min(200, num_batches)
    val_loss = evaluate(model, evalloader, num_batches=eval_batches, device=device)
    val_ppl = float(torch.exp(torch.tensor(val_loss)).item()) if val_loss > 0 else float("inf")

    train_metrics = {"train_loss": train_loss, "train_ppl": train_ppl, "val_loss": val_loss, "val_ppl": val_ppl}

    return model.state_dict(), train_metrics, batches_processed
