"""
Main training script for EDiSS
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset.pdcare_dataset import PDCareDatasetCreator
from src.dataset.annotation import DialogueAnnotator, QualityChecker
from src.models.ediss_model import EDiSSModel, DSSModel, DialogueProcessor
from src.models.classifiers import (
    PGAClassifier,
    PolitenessStrategyClassifier,
    EmpathyStrategyClassifier,
    ClassifierTrainer
)
from src.training.rewards import RewardCalculator, AdaptiveRewardScheduler
from src.training.ppo_trainer import PPOTrainer


class PDCareDataset(Dataset):
    """PyTorch Dataset for PDCare"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = DialogueProcessor(tokenizer, max_length)
        
        # Process all dialogues into examples
        self.examples = []
        for dialogue in self.data:
            dialogue_examples = self.processor.process_dialogue(dialogue)
            self.examples.extend(dialogue_examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Custom collate function for batching"""
    
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Collect user profiles
    user_profile = {
        "persona": [item["user_profile"]["persona"] for item in batch],
        "gender": [item["user_profile"]["gender"] for item in batch],
        "age_group": [item["user_profile"]["age_group"] for item in batch],
        "disability_type": [item["user_profile"]["disability_type"] for item in batch]
    }
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "user_profile": user_profile
    }


def train_classifiers(args):
    """Train the strategy classifiers"""
    
    print("Training strategy classifiers...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    # Initialize classifiers
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Load annotated data
    print("Loading annotated training data...")
    with open(os.path.join(args.data_dir, "train_annotated.json"), 'r') as f:
        train_data = json.load(f)
    
    # Prepare training data for each classifier
    pga_data = []
    pol_data = []
    emp_data = []
    
    for dialogue in train_data:
        full_text = " ".join([u["text"] for u in dialogue["utterances"]])
        
        # PGA data
        if "pga_class" in dialogue:
            pga_idx = pga_clf.pga_to_idx.get(dialogue["pga_class"], 0)
            pga_data.append((full_text, pga_idx))
        
        # Politeness and empathy data
        for utt in dialogue["utterances"]:
            if utt["speaker"] == "doctor":
                text = utt["text"]
                
                if "politeness_strategy" in utt:
                    pol_idx = ["positive_politeness", "negative_politeness", "bald_on_record"].index(
                        utt["politeness_strategy"]
                    )
                    pol_data.append((text, pol_idx))
                
                if "empathy_strategy" in utt:
                    emp_labels = [
                        "genuine_engagement", "privacy_assurance", "forward_focus_encouragement",
                        "compassionate_validation", "practical_assistance", "continuous_support",
                        "strength_based_support", "no_strategy"
                    ]
                    if utt["empathy_strategy"] in emp_labels:
                        emp_idx = emp_labels.index(utt["empathy_strategy"])
                        emp_data.append((text, emp_idx))
    
    # Train classifiers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train PGA classifier
    print("\nTraining PGA classifier...")
    pga_trainer = ClassifierTrainer(pga_clf, tokenizer, device)
    
    for epoch in range(args.classifier_epochs):
        print(f"Epoch {epoch + 1}/{args.classifier_epochs}")
        
        # Shuffle data
        np.random.shuffle(pga_data)
        train_size = int(0.9 * len(pga_data))
        train_set = pga_data[:train_size]
        val_set = pga_data[train_size:]
        
        # Training
        total_loss = 0
        batch_size = 16
        optimizer = torch.optim.AdamW(pga_clf.parameters(), lr=2e-5)
        
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i+batch_size]
            batch_dict = pga_trainer.prepare_batch(
                [item[0] for item in batch],
                [item[1] for item in batch]
            )
            loss = pga_trainer.train_step(batch_dict, optimizer)
            total_loss += loss
        
        # Validation
        val_metrics = pga_trainer.evaluate(val_set)
        print(f"  PGA - Loss: {total_loss/(len(train_set)/batch_size):.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    pga_trainer.save_model(os.path.join(args.model_dir, "pga_classifier.pt"))
    
    # Train Politeness classifier
    print("\nTraining Politeness classifier...")
    pol_trainer = ClassifierTrainer(pol_clf, tokenizer, device)
    
    for epoch in range(args.classifier_epochs):
        print(f"Epoch {epoch + 1}/{args.classifier_epochs}")
        
        np.random.shuffle(pol_data)
        train_size = int(0.9 * len(pol_data))
        train_set = pol_data[:train_size]
        val_set = pol_data[train_size:]
        
        total_loss = 0
        optimizer = torch.optim.AdamW(pol_clf.parameters(), lr=2e-5)
        
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i+batch_size]
            batch_dict = pol_trainer.prepare_batch(
                [item[0] for item in batch],
                [item[1] for item in batch]
            )
            loss = pol_trainer.train_step(batch_dict, optimizer)
            total_loss += loss
        
        val_metrics = pol_trainer.evaluate(val_set)
        print(f"  Politeness - Loss: {total_loss/(len(train_set)/batch_size):.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    pol_trainer.save_model(os.path.join(args.model_dir, "politeness_classifier.pt"))
    
    # Train Empathy classifier
    print("\nTraining Empathy classifier...")
    emp_trainer = ClassifierTrainer(emp_clf, tokenizer, device)
    
    for epoch in range(args.classifier_epochs):
        print(f"Epoch {epoch + 1}/{args.classifier_epochs}")
        
        np.random.shuffle(emp_data)
        train_size = int(0.9 * len(emp_data))
        train_set = emp_data[:train_size]
        val_set = emp_data[train_size:]
        
        total_loss = 0
        optimizer = torch.optim.AdamW(emp_clf.parameters(), lr=2e-5)
        
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i+batch_size]
            batch_dict = emp_trainer.prepare_batch(
                [item[0] for item in batch],
                [item[1] for item in batch]
            )
            loss = emp_trainer.train_step(batch_dict, optimizer)
            total_loss += loss
        
        val_metrics = emp_trainer.evaluate(val_set)
        print(f"  Empathy - Loss: {total_loss/(len(train_set)/batch_size):.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    emp_trainer.save_model(os.path.join(args.model_dir, "empathy_classifier.pt"))
    
    print("\nClassifier training completed!")


def train_dss(args):
    """Train DSS model (warm-start)"""
    
    print("Training DSS model (warm-start)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and tokenizer
    model = DSSModel(device=device)
    tokenizer = model.tokenizer
    
    # Load dataset
    print("Loading training data...")
    train_dataset = PDCareDataset(
        os.path.join(args.data_dir, "train_annotated.json"),
        tokenizer
    )
    val_dataset = PDCareDataset(
        os.path.join(args.data_dir, "val_annotated.json"),
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Training
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.dss_epochs):
        print(f"\nEpoch {epoch + 1}/{args.dss_epochs}")
        
        # Training
        model.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            loss = model.train_on_batch(batch, optimizer)
            total_loss += loss
            
            progress_bar.set_postfix({"loss": loss})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = model.forward(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"]
                )
                val_loss += outputs["loss"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_model(os.path.join(args.model_dir, f"dss_epoch_{epoch+1}"))
    
    # Save final model
    model.save_model(os.path.join(args.model_dir, "dss_final"))
    print("\nDSS training completed!")


def train_ediss_ppo(args):
    """Train EDiSS with PPO"""
    
    print("Training EDiSS with PPO...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    print("Loading models...")
    policy_model = EDiSSModel(device=device)
    
    # Load DSS checkpoint as reference model
    ref_model = DSSModel(device=device)
    if os.path.exists(os.path.join(args.model_dir, "dss_final")):
        ref_model.load_model(os.path.join(args.model_dir, "dss_final"))
    
    # Load classifiers
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Load classifier checkpoints if available
    if os.path.exists(os.path.join(args.model_dir, "pga_classifier.pt")):
        pga_checkpoint = torch.load(os.path.join(args.model_dir, "pga_classifier.pt"))
        pga_clf.load_state_dict(pga_checkpoint["model_state_dict"])
    
    if os.path.exists(os.path.join(args.model_dir, "politeness_classifier.pt")):
        pol_checkpoint = torch.load(os.path.join(args.model_dir, "politeness_classifier.pt"))
        pol_clf.load_state_dict(pol_checkpoint["model_state_dict"])
    
    if os.path.exists(os.path.join(args.model_dir, "empathy_classifier.pt")):
        emp_checkpoint = torch.load(os.path.join(args.model_dir, "empathy_classifier.pt"))
        emp_clf.load_state_dict(emp_checkpoint["model_state_dict"])
    
    # Initialize tokenizer
    tokenizer = policy_model.tokenizer
    
    # Initialize reward calculator
    reward_calc = RewardCalculator(
        pga_clf,
        pol_clf,
        emp_clf,
        tokenizer,
        device=device
    )
    
    # Load dataset
    print("Loading training data...")
    train_dataset = PDCareDataset(
        os.path.join(args.data_dir, "train_annotated.json"),
        tokenizer
    )
    val_dataset = PDCareDataset(
        os.path.join(args.data_dir, "val_annotated.json"),
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.ppo_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.ppo_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # PPO config
    ppo_config = {
        "learning_rate": args.ppo_learning_rate,
        "batch_size": args.ppo_batch_size,
        "mini_batch_size": args.ppo_mini_batch_size,
        "ppo_epochs": args.ppo_epochs,
        "clip_epsilon": 0.2,
        "device": device
    }
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        policy_model,
        ref_model,
        reward_calc,
        tokenizer,
        config=ppo_config
    )
    
    # Train with PPO
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.ppo_train_epochs,
        use_wandb=args.use_wandb
    )
    
    # Save final model
    policy_model.save_model(os.path.join(args.model_dir, "ediss_final"))
    print("\nEDiSS PPO training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train EDiSS model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/pdcare",
                       help="Directory containing PDCare dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                       help="Directory to save models")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="all",
                       choices=["create_dataset", "annotate", "classifiers", "dss", "ppo", "all"],
                       help="Training mode")
    
    # Dataset creation arguments
    parser.add_argument("--num_dialogues", type=int, default=6796,
                       help="Number of dialogues to create")
    
    # Classifier training arguments
    parser.add_argument("--classifier_epochs", type=int, default=10,
                       help="Number of epochs for classifier training")
    
    # DSS training arguments
    parser.add_argument("--dss_epochs", type=int, default=5,
                       help="Number of epochs for DSS training")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for DSS training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for DSS training")
    
    # PPO training arguments
    parser.add_argument("--ppo_train_epochs", type=int, default=3,
                       help="Number of epochs for PPO training")
    parser.add_argument("--ppo_batch_size", type=int, default=2,
                       help="Batch size for PPO training")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=1,
                       help="Mini-batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="Number of PPO epochs per update")
    parser.add_argument("--ppo_learning_rate", type=float, default=1e-5,
                       help="Learning rate for PPO")
    
    # Other arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Execute based on mode
    if args.mode == "create_dataset" or args.mode == "all":
        print("Creating PDCare dataset...")
        creator = PDCareDatasetCreator(seed=args.seed)
        dataset = creator.create_dataset(num_dialogues=args.num_dialogues)
        creator.save_dataset(dataset, args.data_dir)
    
    if args.mode == "annotate" or args.mode == "all":
        print("Annotating dataset...")
        annotator = DialogueAnnotator()
        
        for split in ["train", "validation", "test"]:
            input_file = os.path.join(args.data_dir, f"{split}.json")
            output_file = os.path.join(args.data_dir, f"{split}_annotated.json")
            
            if os.path.exists(input_file):
                print(f"Annotating {split} split...")
                annotator.annotate_dataset(input_file, output_file)
        
        # Rename validation to val for consistency
        if os.path.exists(os.path.join(args.data_dir, "validation_annotated.json")):
            os.rename(
                os.path.join(args.data_dir, "validation_annotated.json"),
                os.path.join(args.data_dir, "val_annotated.json")
            )
    
    if args.mode == "classifiers" or args.mode == "all":
        train_classifiers(args)
    
    if args.mode == "dss" or args.mode == "all":
        train_dss(args)
    
    if args.mode == "ppo" or args.mode == "all":
        train_ediss_ppo(args)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()