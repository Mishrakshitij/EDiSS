"""
PPO Training for EDiSS
Proximal Policy Optimization implementation for dialogue model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb
from collections import deque


class PPOTrainer:
    """PPO trainer for EDiSS model"""
    
    def __init__(
        self,
        policy_model,  # EDiSS model
        ref_model,     # Reference model (frozen DSS)
        reward_calculator,
        tokenizer,
        config: Optional[Dict] = None
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_calculator = reward_calculator
        self.tokenizer = tokenizer
        
        # Default PPO config
        default_config = {
            "learning_rate": 1e-5,
            "batch_size": 4,
            "mini_batch_size": 1,
            "ppo_epochs": 4,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 1.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "kl_penalty": 0.01,
            "target_kl": 0.05,
            "max_steps": 100000,
            "warmup_steps": 500,
            "log_interval": 10,
            "save_interval": 1000,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.device = torch.device(self.config["device"])
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.policy_model.model.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # Initialize value network
        self.value_net = ValueNetwork(
            self.policy_model.config.hidden_size
        ).to(self.device)
        
        self.value_optimizer = AdamW(
            self.value_net.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # Training statistics
        self.global_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        use_wandb: bool = False
    ):
        """Main training loop"""
        
        if use_wandb:
            wandb.init(project="EDiSS-PPO", config=self.config)
        
        self.policy_model.model.train()
        self.ref_model.model.eval()
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            epoch_stats = self._train_epoch(train_dataloader)
            
            # Validation
            if val_dataloader:
                val_stats = self._validate(val_dataloader)
                epoch_stats.update(val_stats)
            
            # Log statistics
            self._log_stats(epoch_stats, use_wandb)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)
        
        if use_wandb:
            wandb.finish()
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        epoch_stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "mean_reward": 0,
            "mean_kl": 0,
            "num_updates": 0
        }
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Collect trajectories
            trajectories = self._collect_trajectories(batch)
            
            if not trajectories:
                continue
            
            # Compute advantages
            advantages, returns = self._compute_advantages(trajectories)
            
            # PPO update
            update_stats = self._ppo_update(
                trajectories,
                advantages,
                returns
            )
            
            # Update statistics
            for key in update_stats:
                if key in epoch_stats:
                    epoch_stats[key] += update_stats[key]
            epoch_stats["num_updates"] += 1
            
            # Update progress bar
            if epoch_stats["num_updates"] > 0:
                progress_bar.set_postfix({
                    "reward": epoch_stats["mean_reward"] / epoch_stats["num_updates"],
                    "kl": epoch_stats["mean_kl"] / epoch_stats["num_updates"]
                })
            
            self.global_step += 1
            
            # Early stopping if KL divergence too high
            if epoch_stats["num_updates"] > 0:
                mean_kl = epoch_stats["mean_kl"] / epoch_stats["num_updates"]
                if mean_kl > self.config["target_kl"] * 1.5:
                    print(f"Early stopping: KL divergence {mean_kl:.4f} > target")
                    break
        
        # Average statistics
        if epoch_stats["num_updates"] > 0:
            for key in epoch_stats:
                if key != "num_updates":
                    epoch_stats[key] /= epoch_stats["num_updates"]
        
        return epoch_stats
    
    def _collect_trajectories(self, batch: Dict) -> List[Dict]:
        """Collect trajectories by generating responses"""
        
        trajectories = []
        
        for i in range(len(batch["input_ids"])):
            # Extract single example
            input_ids = batch["input_ids"][i].unsqueeze(0).to(self.device)
            attention_mask = batch["attention_mask"][i].unsqueeze(0).to(self.device)
            user_profile = {
                k: batch["user_profile"][k][i] 
                for k in batch["user_profile"]
            }
            
            # Generate response
            with torch.no_grad():
                # Policy model generation
                policy_outputs = self.policy_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                generated_ids = policy_outputs.sequences[0]
                generated_text = self.tokenizer.decode(
                    generated_ids[input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                # Reference model log probs
                ref_outputs = self.ref_model.forward(
                    input_ids=generated_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(generated_ids).unsqueeze(0)
                )
                ref_logits = ref_outputs["logits"]
                
                # Policy model log probs
                policy_outputs_forward = self.policy_model.forward(
                    input_ids=generated_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(generated_ids).unsqueeze(0)
                )
                policy_logits = policy_outputs_forward["logits"]
            
            # Calculate reward
            target_text = self.tokenizer.decode(
                batch["labels"][i][batch["labels"][i] != -100],
                skip_special_tokens=True
            )
            
            context = self.tokenizer.decode(
                input_ids[0][:100],  # First 100 tokens as context
                skip_special_tokens=True
            )
            
            previous_utterance = ""  # Extract from context if available
            
            reward, components = self.reward_calculator.calculate_reward(
                generated_text,
                target_text,
                user_profile,
                context,
                previous_utterance
            )
            
            # Create trajectory
            trajectory = {
                "input_ids": input_ids,
                "generated_ids": generated_ids,
                "attention_mask": attention_mask,
                "policy_logits": policy_logits,
                "ref_logits": ref_logits,
                "reward": reward,
                "reward_components": components,
                "value": None  # Will be filled by value network
            }
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _compute_advantages(
        self,
        trajectories: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        
        rewards = torch.tensor(
            [t["reward"] for t in trajectories],
            dtype=torch.float32
        ).to(self.device)
        
        # Get value estimates
        values = []
        for trajectory in trajectories:
            with torch.no_grad():
                value = self.value_net(
                    trajectory["policy_logits"].mean(dim=-1)
                )
            values.append(value.item())
            trajectory["value"] = value
        
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config["gamma"] * next_value - values[t]
            gae = delta + self.config["gamma"] * self.config["gae_lambda"] * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _ppo_update(
        self,
        trajectories: List[Dict],
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Perform PPO update"""
        
        update_stats = {
            "total_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "mean_reward": np.mean([t["reward"] for t in trajectories]),
            "mean_kl": 0
        }
        
        # Mini-batch updates
        for _ in range(self.config["ppo_epochs"]):
            for i in range(0, len(trajectories), self.config["mini_batch_size"]):
                mini_batch = trajectories[i:i + self.config["mini_batch_size"]]
                mini_advantages = advantages[i:i + self.config["mini_batch_size"]]
                mini_returns = returns[i:i + self.config["mini_batch_size"]]
                
                # Compute current policy log probs
                policy_loss = 0
                entropy_loss = 0
                kl_div = 0
                
                for j, trajectory in enumerate(mini_batch):
                    # Get log probabilities
                    policy_logits = trajectory["policy_logits"]
                    ref_logits = trajectory["ref_logits"]
                    
                    # Compute log probs for taken actions
                    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    
                    # Get action log probs (simplified - using mean)
                    action_log_probs = policy_log_probs.mean()
                    old_action_log_probs = ref_log_probs.mean()
                    
                    # Compute ratio
                    ratio = torch.exp(action_log_probs - old_action_log_probs)
                    
                    # Clipped surrogate loss
                    surr1 = ratio * mini_advantages[j]
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config["clip_epsilon"],
                        1 + self.config["clip_epsilon"]
                    ) * mini_advantages[j]
                    
                    policy_loss -= torch.min(surr1, surr2)
                    
                    # Entropy bonus
                    entropy = -(policy_log_probs * torch.exp(policy_log_probs)).mean()
                    entropy_loss -= self.config["entropy_coef"] * entropy
                    
                    # KL penalty
                    kl = (torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs)).mean()
                    kl_div += kl
                
                # Value loss
                value_loss = 0
                for j, trajectory in enumerate(mini_batch):
                    value_pred = trajectory["value"]
                    value_loss += F.mse_loss(value_pred, mini_returns[j])
                
                value_loss *= self.config["value_loss_coef"]
                
                # Total loss
                total_loss = policy_loss + value_loss + entropy_loss
                total_loss += self.config["kl_penalty"] * kl_div
                
                # Backprop
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy_model.model.parameters(),
                    self.config["max_grad_norm"]
                )
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.config["max_grad_norm"]
                )
                
                self.optimizer.step()
                self.value_optimizer.step()
                
                # Update statistics
                update_stats["total_loss"] += total_loss.item()
                update_stats["policy_loss"] += policy_loss.item()
                update_stats["value_loss"] += value_loss.item()
                update_stats["entropy"] += entropy_loss.item()
                update_stats["mean_kl"] += kl_div.item()
        
        # Average over mini-batches and epochs
        num_updates = self.config["ppo_epochs"] * \
                     (len(trajectories) // self.config["mini_batch_size"])
        
        if num_updates > 0:
            for key in ["total_loss", "policy_loss", "value_loss", "entropy", "mean_kl"]:
                update_stats[key] /= num_updates
        
        return update_stats
    
    def _validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        
        self.policy_model.model.eval()
        val_stats = {
            "val_reward": 0,
            "val_perplexity": 0,
            "num_samples": 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                trajectories = self._collect_trajectories(batch)
                
                for trajectory in trajectories:
                    val_stats["val_reward"] += trajectory["reward"]
                    val_stats["num_samples"] += 1
        
        if val_stats["num_samples"] > 0:
            val_stats["val_reward"] /= val_stats["num_samples"]
        
        self.policy_model.model.train()
        
        return val_stats
    
    def _log_stats(self, stats: Dict[str, float], use_wandb: bool):
        """Log training statistics"""
        
        print("\nTraining Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        if use_wandb:
            wandb.log(stats, step=self.global_step)
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "policy_model_state": self.policy_model.model.state_dict(),
            "value_net_state": self.value_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "value_optimizer_state": self.value_optimizer.state_dict(),
            "config": self.config
        }
        
        path = f"checkpoints/ediss_epoch_{epoch}_step_{self.global_step}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


class ValueNetwork(nn.Module):
    """Value network for advantage estimation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # Ensure correct dimensions
        if x.shape[-1] != self.net[0].in_features:
            # Use adaptive pooling to match dimensions
            x = F.adaptive_avg_pool1d(
                x.unsqueeze(1),
                self.net[0].in_features
            ).squeeze(1)
        
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    # Example usage
    from src.models.ediss_model import EDiSSModel, DSSModel
    from src.models.classifiers import (
        PGAClassifier,
        PolitenessStrategyClassifier,
        EmpathyStrategyClassifier
    )
    from src.training.rewards import RewardCalculator
    from transformers import AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    print("Initializing models...")
    policy_model = EDiSSModel(device=device)
    ref_model = DSSModel(device=device)
    
    # Initialize classifiers
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-8k-instruct")
    
    # Initialize reward calculator
    reward_calc = RewardCalculator(
        pga_clf,
        pol_clf,
        emp_clf,
        tokenizer,
        device=device
    )
    
    # Initialize PPO trainer
    ppo_config = {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "device": device
    }
    
    trainer = PPOTrainer(
        policy_model,
        ref_model,
        reward_calc,
        tokenizer,
        config=ppo_config
    )
    
    print("PPO Trainer initialized successfully!")