"""
Multimodal trainer for EO satellite imagery analysis with GPT-OSS integration
Implements vision-language alignment and fine-tuning strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CLIPModel, CLIPProcessor, CLIPVisionModel,
    BlipProcessor, BlipForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from omegaconf import DictConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalConfig:
    """Configuration for multimodal training"""
    # Model configs
    vision_encoder: str = "openai/clip-vit-base-patch32"
    language_model: str = "gpt2-medium"  # GPT-OSS equivalent
    projection_dim: int = 768
    hidden_dim: int = 1024
    
    # Training configs
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # LoRA configs for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Data configs
    max_text_length: int = 512
    image_size: int = 224
    
    # Training strategy
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = True
    curriculum_learning: bool = True
    
    # Paths
    data_path: str = "./datasets"
    model_save_path: str = "./models"
    checkpoint_path: str = "./checkpoints"

class EOImageCaptionDataset(Dataset):
    """Dataset for EO image-caption pairs"""
    
    def __init__(
        self, 
        data_path: str, 
        processor_vision, 
        tokenizer, 
        max_length: int = 512,
        split: str = "train"
    ):
        self.data_path = Path(data_path)
        self.processor_vision = processor_vision
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset metadata
        with open(self.data_path / f"{split}_metadata.json") as f:
            self.metadata = json.load(f)
            
        self.samples = self.metadata['samples']
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_path / sample['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        image_inputs = self.processor_vision(
            images=image, 
            return_tensors="pt"
        )
        
        # Process text
        caption = sample['caption']
        text_inputs = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'image': image_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': text_inputs['input_ids'].squeeze(0),  # For language modeling
            'metadata': {
                'image_id': sample['image_id'],
                'coordinates': sample.get('coordinates', [0, 0]),
                'capture_date': sample.get('capture_date', ''),
                'satellite': sample.get('satellite', ''),
                'land_cover': sample.get('land_cover', [])
            }
        }

class VisionLanguageProjection(nn.Module):
    """Projection layer to align vision and language embeddings"""
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        # Multi-layer projection with residual connections
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, text_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Residual connection
        self.residual = nn.Linear(vision_dim, text_dim) if vision_dim != text_dim else nn.Identity()
        
    def forward(self, vision_features):
        projected = self.projection(vision_features)
        residual = self.residual(vision_features)
        return projected + residual * 0.1  # Weighted residual

class MultimodalEOModel(L.LightningModule):
    """Multimodal model for EO analysis with GPT-OSS integration"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Vision encoder (CLIP)
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_encoder)
        self.vision_processor = CLIPProcessor.from_pretrained(config.vision_encoder)
        
        # Language model (GPT-OSS equivalent)
        self.language_model = GPT2LMHeadModel.from_pretrained(config.language_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layer for alignment
        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.language_model.config.hidden_size
        
        self.projection = VisionLanguageProjection(
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=config.hidden_dim
        )
        
        # Freeze models if specified
        if config.freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
                
        if config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Apply LoRA for efficient fine-tuning
        if config.use_lora and not config.freeze_language_model:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["c_attn", "c_proj", "c_fc"]
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
        
        # Loss functions
        self.alignment_loss = nn.CosineEmbeddingLoss()
        self.language_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, batch):
        """Forward pass for multimodal input"""
        # Extract vision features
        vision_outputs = self.vision_model(pixel_values=batch['image'])
        vision_features = vision_outputs.pooler_output
        
        # Project vision features to text space
        aligned_features = self.projection(vision_features)
        
        # Generate text embeddings for alignment
        text_outputs = self.language_model.transformer(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Language modeling
        lm_outputs = self.language_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        return {
            'vision_features': vision_features,
            'aligned_features': aligned_features,
            'text_features': text_features,
            'lm_logits': lm_outputs.logits,
            'lm_loss': lm_outputs.loss
        }
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        
        # Alignment loss (cosine similarity)
        target = torch.ones(outputs['aligned_features'].size(0)).to(self.device)
        alignment_loss = self.alignment_loss(
            outputs['aligned_features'], 
            outputs['text_features'], 
            target
        )
        
        # Language modeling loss
        lm_loss = outputs['lm_loss']
        
        # Combined loss
        total_loss = alignment_loss + lm_loss
        
        # Logging
        self.log('train/alignment_loss', alignment_loss, prog_bar=True)
        self.log('train/lm_loss', lm_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        
        # Calculate losses
        target = torch.ones(outputs['aligned_features'].size(0)).to(self.device)
        alignment_loss = self.alignment_loss(
            outputs['aligned_features'], 
            outputs['text_features'], 
            target
        )
        lm_loss = outputs['lm_loss']
        total_loss = alignment_loss + lm_loss
        
        # Calculate similarity metrics
        similarities = F.cosine_similarity(
            outputs['aligned_features'], 
            outputs['text_features']
        )
        avg_similarity = similarities.mean()
        
        self.log('val/alignment_loss', alignment_loss, prog_bar=True)
        self.log('val/lm_loss', lm_loss, prog_bar=True)
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/avg_similarity', avg_similarity, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        # Only optimize unfrozen parameters
        params_to_optimize = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.config.warmup_steps,
            T_mult=2
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def generate_caption(self, image_tensor, max_length=50):
        """Generate caption for an image"""
        self.eval()
        with torch.no_grad():
            # Extract and align vision features
            vision_outputs = self.vision_model(pixel_values=image_tensor)
            vision_features = vision_outputs.pooler_output
            aligned_features = self.projection(vision_features)
            
            # Generate text using language model
            # This is a simplified version - in practice, you'd need more sophisticated generation
            prompt = "This satellite image shows"
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            generated = self.language_model.generate(
                prompt_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            caption = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return caption

class MultimodalTrainer:
    """Trainer class for multimodal EO analysis"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.model = MultimodalEOModel(config)
        
        # Initialize wandb for experiment tracking
        wandb.init(
            project="multimodal-eo-analysis",
            config=config.__dict__
        )
    
    def prepare_data(self):
        """Prepare training and validation datasets"""
        # Training dataset
        self.train_dataset = EOImageCaptionDataset(
            data_path=self.config.data_path,
            processor_vision=self.model.vision_processor,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_text_length,
            split="train"
        )
        
        # Validation dataset
        self.val_dataset = EOImageCaptionDataset(
            data_path=self.config.data_path,
            processor_vision=self.model.vision_processor,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_text_length,
            split="val"
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def train(self):
        """Train the multimodal model"""
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_path,
            filename="multimodal-eo-{epoch:02d}-{val/total_loss:.2f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3
        )
        
        early_stopping = EarlyStopping(
            monitor="val/total_loss",
            patience=5,
            mode="min"
        )
        
        # Trainer
        trainer = L.Trainer(
            max_epochs=self.config.num_epochs,
            callbacks=[checkpoint_callback, early_stopping],
            gradient_clip_val=self.config.gradient_clip,
            accelerator="auto",
            devices="auto",
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            precision="16-mixed",
            log_every_n_steps=50
        )
        
        # Train
        trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
        # Save final model
        self.model.save_pretrained(self.config.model_save_path)
        logger.info(f"Model saved to {self.config.model_save_path}")
    
    def evaluate(self, test_loader=None):
        """Evaluate the trained model"""
        if test_loader is None:
            test_loader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        similarities = []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(batch)
                
                # Calculate similarity
                sim = F.cosine_similarity(
                    outputs['aligned_features'],
                    outputs['text_features']
                ).cpu().numpy()
                similarities.extend(sim)
                
                total_loss += outputs['lm_loss'].item()
        
        avg_loss = total_loss / len(test_loader)
        avg_similarity = np.mean(similarities)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Similarity: {avg_similarity:.4f}")
        
        return {
            "avg_loss": avg_loss,
            "avg_similarity": avg_similarity,
            "similarities": similarities
        }

def main():
    """Main training function"""
    config = MultimodalConfig()
    trainer = MultimodalTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
