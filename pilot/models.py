"""Model definitions for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class SimpleCNN(nn.Module):
    """Simple CNN with 2 conv layers and 3 fully connected layers."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the size after convolutions for CIFAR-10 (32x32 input)
        # After conv1 + pool: 16x16x32
        # After conv2 + pool: 8x8x64 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet18(nn.Module):
    """ResNet18 model for CIFAR-10 classification."""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.backbone = resnet18(weights=None)
        # Replace the classifier for CIFAR-10 (10 classes)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class LoRAMistral7B(nn.Module):
    """LoRA fine-tuned Mistral 7B model for text generation."""

    def __init__(
        self,
        model_name="mistralai/Mistral-7B-v0.1",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ):
        super(LoRAMistral7B, self).__init__()

        # Default target modules for Mistral (similar to Llama architecture)
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        # Configure LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.base_model, self.lora_config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the LoRA-adapted model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(self, input_ids, **kwargs):
        """Generate text using the LoRA-adapted model."""
        return self.model.generate(input_ids, **kwargs)

    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        return self.model.get_nb_trainable_parameters()

    def save_lora_weights(self, path):
        """Save only the LoRA adapter weights."""
        self.model.save_pretrained(path)

    def load_lora_weights(self, path):
        """Load LoRA adapter weights."""
        self.model.load_adapter(path)


def get_model(model_type="simple_cnn", **kwargs):
    """Factory function to create models based on type."""
    if model_type == "simple_cnn":
        return SimpleCNN(**kwargs)
    elif model_type == "resnet18":
        return ResNet18(**kwargs)
    elif model_type == "lora_mistral7b":
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and peft are required for LoRAMistral7B. "
                "Install with: pip install transformers peft"
            )
        return LoRAMistral7B(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


