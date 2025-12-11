#!/usr/bin/env python3
"""
EOU Model Training Script
=========================

Train an End-of-Utterance detection model using AraBERT for Arabic text.

Usage:
    python train.py --config configs/training_config.yaml
    python train.py --dataset_name "your-dataset" --output_dir "./models/eou_model"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import TrainingArguments, Trainer
from arabert.preprocess import ArabertPreprocessor
from sklearn.metrics import classification_report, confusion_matrix
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EOUModelTrainer:
    """End-of-Utterance Model Trainer"""
    
    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        dataset_name: str = "arabic-eou-detection-10k",
        output_dir: str = "./models/eou_model",
        max_length: int = 512,
        num_labels: int = 2,
    ):
        """
        Initialize the EOU Model Trainer
        
        Args:
            model_name: HuggingFace model name or path
            dataset_name: Dataset name or path
            output_dir: Directory to save trained model
            max_length: Maximum sequence length
            num_labels: Number of classification labels (2 for binary)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.num_labels = num_labels
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.arabert_prep = None
        self.dataset = None
        self.tokenized_datasets = None
        
        logger.info(f"Initialized EOUModelTrainer with model: {model_name}")
    
    def setup(self):
        """Setup tokenizer, model, and preprocessor"""
        logger.info("Setting up tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"✓ Tokenizer loaded: {type(self.tokenizer).__name__}")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={0: "No EOU", 1: "EOU"},
            label2id={"No EOU": 0, "EOU": 1}
        )
        logger.info(f"✓ Model loaded: {type(self.model).__name__}")
        
        # Initialize AraBERT preprocessor
        self.arabert_prep = ArabertPreprocessor(
            model_name=self.model_name,
            keep_emojis=False
        )
        logger.info("✓ AraBERT preprocessor initialized")
    
    def load_data(self):
        """Load and prepare dataset"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            # Load dataset from HuggingFace Hub
            self.dataset = load_dataset(self.dataset_name)
            logger.info(f"✓ Dataset loaded from HuggingFace Hub")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Attempting to load from local path...")
            try:
                self.dataset = load_dataset(self.dataset_name)
                logger.info(f"✓ Dataset loaded from local path")
            except Exception as e2:
                logger.error(f"Failed to load dataset from local path: {e2}")
                raise
        
        # Print dataset info
        logger.info(f"Dataset columns: {self.dataset['train'].column_names}")
        logger.info(f"Train samples: {len(self.dataset['train'])}")
        if "validation" in self.dataset:
            logger.info(f"Validation samples: {len(self.dataset['validation'])}")
        if "test" in self.dataset:
            logger.info(f"Test samples: {len(self.dataset['test'])}")
    
    def preprocess_and_tokenize(self, examples):
        """
        Preprocess and tokenize examples
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Tokenized examples
        """
        # Determine text column name
        text_column = "utterance" if "utterance" in examples else "text"
        
        # Apply AraBERT preprocessing
        clean_texts = [
            self.arabert_prep.preprocess(t) for t in examples[text_column]
        ]
        
        # Tokenize
        return self.tokenizer(
            clean_texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
    
    def prepare_data(self):
        """Tokenize and prepare datasets"""
        logger.info("Tokenizing datasets...")
        
        self.tokenized_datasets = self.dataset.map(
            self.preprocess_and_tokenize,
            batched=True,
            desc="Tokenizing"
        )
        
        logger.info("✓ Tokenization complete")
        
        # Create test split if it doesn't exist
        if "test" not in self.tokenized_datasets:
            logger.info("Creating test split from train...")
            split = self.tokenized_datasets["train"].train_test_split(
                test_size=0.15,
                seed=42
            )
            self.tokenized_datasets["train"] = split["train"]
            self.tokenized_datasets["test"] = split["test"]
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Load accuracy and F1 metrics
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        
        # Compute metrics
        acc_score = accuracy.compute(predictions=predictions, references=labels)
        f1_score = f1.compute(
            predictions=predictions,
            references=labels,
            average="binary"
        )
        
        return {
            "accuracy": acc_score["accuracy"],
            "f1": f1_score["f1"]
        }
    
    def train(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
    ):
        """
        Train the model
        
        Args:
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
        """
        logger.info("Starting training...")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Weight decay: {weight_decay}")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets.get("validation") or self.tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        logger.info("Training started...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("✓ Training complete!")
        
        return trainer
    
    def evaluate(self, trainer: Trainer):
        """
        Evaluate the trained model
        
        Args:
            trainer: Trained Trainer object
        """
        logger.info("Evaluating model on test set...")
        
        # Get test dataset
        test_dataset = self.tokenized_datasets.get("test")
        if test_dataset is None:
            logger.warning("No test dataset found, skipping evaluation")
            return
        
        # Predict
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids
        
        # Classification report
        logger.info("\n" + "="*60)
        logger.info("Classification Report")
        logger.info("="*60)
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=["No EOU", "EOU"],
            digits=4
        )
        print(report)
        
        # Confusion matrix
        logger.info("\n" + "="*60)
        logger.info("Confusion Matrix")
        logger.info("="*60)
        cm = confusion_matrix(true_labels, pred_labels)
        print(cm)
        
        # Save evaluation results
        eval_results = {
            "classification_report": classification_report(
                true_labels,
                pred_labels,
                target_names=["No EOU", "EOU"],
                output_dict=True
            ),
            "confusion_matrix": cm.tolist(),
        }
        
        eval_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Evaluation results saved to {eval_path}")
    
    def run(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
    ):
        """
        Run the complete training pipeline
        
        Args:
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            num_epochs: Number of training epochs
            weight_decay: Weight decay for regularization
        """
        logger.info("="*60)
        logger.info("EOU Model Training Pipeline")
        logger.info("="*60)
        
        # Setup
        self.setup()
        
        # Load data
        self.load_data()
        
        # Prepare data
        self.prepare_data()
        
        # Train
        trainer = self.train(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
        )
        
        # Evaluate
        self.evaluate(trainer)
        
        logger.info("="*60)
        logger.info("✓ Pipeline Complete!")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train EOU detection model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="aubmindlab/bert-base-arabertv2",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="arabic-eou-detection-10k",
        help="Dataset name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/eou_model",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EOUModelTrainer(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )
    
    # Run training
    trainer.run(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
