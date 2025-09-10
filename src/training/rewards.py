"""
Reward functions for EDiSS reinforcement learning training
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


class RewardCalculator:
    """Calculate rewards for EDiSS training"""
    
    def __init__(
        self,
        pga_classifier,
        politeness_classifier,
        empathy_classifier,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights: Optional[Dict[str, float]] = None
    ):
        self.pga_classifier = pga_classifier.to(device)
        self.politeness_classifier = politeness_classifier.to(device)
        self.empathy_classifier = empathy_classifier.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Default weights
        if weights is None:
            weights = {
                "w1": 0.33,  # User-profile alignment
                "w2": 0.33,  # Politeness adherence
                "w3": 0.34,  # Empathy consistency
                "w4": 0.5,   # Syntactic smoothness
                "w5": 0.5,   # Semantic smoothness
                "gamma": 0.7,  # Task-relevance vs smoothness
                "lambda": 2.0,  # Scaling factor
                "alpha": 1.5   # Penalization factor
            }
        self.weights = weights
    
    def calculate_reward(
        self,
        generated_response: str,
        target_response: str,
        user_profile: Dict[str, str],
        context: str,
        previous_utterance: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate total reward and component scores"""
        
        # Calculate task-relevance rewards
        delta_1 = self._calculate_user_profile_alignment(
            generated_response, target_response, user_profile, context
        )
        delta_2 = self._calculate_politeness_adherence(
            generated_response, target_response
        )
        delta_3 = self._calculate_empathy_consistency(
            generated_response, target_response
        )
        
        # Calculate smoothness rewards
        delta_syn = self._calculate_syntactic_smoothness(generated_response)
        delta_sem = self._calculate_semantic_smoothness(
            generated_response, previous_utterance
        )
        
        # Combine task-relevance rewards
        task_relevance_score = (
            self.weights["w1"] * delta_1 +
            self.weights["w2"] * delta_2 +
            self.weights["w3"] * delta_3
        )
        
        # Apply sigmoid to task-relevance
        task_relevance_reward = 1 / (1 + np.exp(
            -self.weights["lambda"] * task_relevance_score
        ))
        
        # Combine smoothness rewards
        smoothness_score = (
            self.weights["w4"] * delta_syn +
            self.weights["w5"] * delta_sem
        )
        
        # Apply sigmoid to smoothness
        smoothness_reward = 1 / (1 + np.exp(
            -self.weights["lambda"] * smoothness_score
        ))
        
        # Calculate total reward
        total_reward = (
            self.weights["gamma"] * task_relevance_reward +
            (1 - self.weights["gamma"]) * smoothness_reward
        )
        
        # Return detailed scores for analysis
        component_scores = {
            "user_profile_alignment": delta_1,
            "politeness_adherence": delta_2,
            "empathy_consistency": delta_3,
            "syntactic_smoothness": delta_syn,
            "semantic_smoothness": delta_sem,
            "task_relevance_reward": task_relevance_reward,
            "smoothness_reward": smoothness_reward,
            "total_reward": total_reward
        }
        
        return total_reward, component_scores
    
    def _calculate_user_profile_alignment(
        self,
        generated: str,
        target: str,
        user_profile: Dict[str, str],
        context: str
    ) -> float:
        """Calculate user-profile alignment reward (Delta_1)"""
        
        # Prepare input for PGA classifier
        gen_input = self._prepare_text_for_classifier(generated, context)
        tgt_input = self._prepare_text_for_classifier(target, context)
        
        # Get PGA predictions
        gen_pga, gen_probs = self._get_pga_prediction(gen_input)
        tgt_pga, tgt_probs = self._get_pga_prediction(tgt_input)
        
        # Extract target PGA class from user profile
        target_pga = self._profile_to_pga_class(user_profile)
        
        # Calculate probabilities for target class
        gen_prob = gen_probs.get(target_pga, 0.0)
        tgt_prob = tgt_probs.get(target_pga, 1.0)
        
        # Calculate reward with penalization
        delta_1 = tgt_prob - self.weights["alpha"] * (1 - gen_prob)
        
        return delta_1
    
    def _calculate_politeness_adherence(
        self,
        generated: str,
        target: str
    ) -> float:
        """Calculate politeness adherence reward (Delta_2)"""
        
        # Get politeness predictions
        gen_strategy, gen_prob = self._get_politeness_prediction(generated)
        tgt_strategy, tgt_prob = self._get_politeness_prediction(target)
        
        # Calculate reward
        if gen_strategy == tgt_strategy:
            delta_2 = tgt_prob - self.weights["alpha"] * (1 - gen_prob)
        else:
            delta_2 = -self.weights["alpha"] * 0.5  # Penalty for wrong strategy
        
        return delta_2
    
    def _calculate_empathy_consistency(
        self,
        generated: str,
        target: str
    ) -> float:
        """Calculate empathy consistency reward (Delta_3)"""
        
        # Get empathy predictions
        gen_strategy, gen_prob = self._get_empathy_prediction(generated)
        tgt_strategy, tgt_prob = self._get_empathy_prediction(target)
        
        # Calculate reward
        if gen_strategy == tgt_strategy:
            delta_3 = tgt_prob - self.weights["alpha"] * (1 - gen_prob)
        else:
            delta_3 = -self.weights["alpha"] * 0.5  # Penalty for wrong strategy
        
        return delta_3
    
    def _calculate_syntactic_smoothness(self, text: str) -> float:
        """Calculate syntactic smoothness (Delta_syn)"""
        
        # Calculate perplexity (simplified version)
        # In practice, use a language model for proper perplexity calculation
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Simple heuristic: shorter, well-formed sentences have lower perplexity
        num_tokens = tokens.shape[1]
        
        # Check for basic syntax issues
        syntax_score = 1.0
        
        # Penalize very short or very long responses
        if num_tokens < 5:
            syntax_score *= 0.5
        elif num_tokens > 100:
            syntax_score *= 0.7
        
        # Check for proper punctuation
        if text and text[-1] not in ".!?":
            syntax_score *= 0.9
        
        # Check for capitalization
        if text and not text[0].isupper():
            syntax_score *= 0.95
        
        # Reciprocal of pseudo-perplexity
        delta_syn = syntax_score / (1 + np.log(max(1, num_tokens)))
        
        return delta_syn
    
    def _calculate_semantic_smoothness(
        self,
        generated: str,
        previous: str
    ) -> float:
        """Calculate semantic smoothness (Delta_sem)"""
        
        if not previous or not generated:
            return 0.5
        
        # Get embeddings using tokenizer (simplified)
        # In practice, use sentence embeddings
        gen_tokens = self.tokenizer.encode(generated, max_length=512, truncation=True)
        prev_tokens = self.tokenizer.encode(previous, max_length=512, truncation=True)
        
        # Create simple bag-of-words vectors
        vocab_size = max(max(gen_tokens + prev_tokens) + 1, 1000)
        gen_vec = np.zeros(vocab_size)
        prev_vec = np.zeros(vocab_size)
        
        for token in gen_tokens:
            if token < vocab_size:
                gen_vec[token] += 1
        
        for token in prev_tokens:
            if token < vocab_size:
                prev_vec[token] += 1
        
        # Normalize
        gen_vec = gen_vec / (np.linalg.norm(gen_vec) + 1e-8)
        prev_vec = prev_vec / (np.linalg.norm(prev_vec) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(gen_vec, prev_vec)
        
        # Ensure reasonable semantic connection (not too similar, not too different)
        if similarity > 0.9:  # Too similar, might be repetitive
            delta_sem = 0.7
        elif similarity < 0.1:  # Too different, might be off-topic
            delta_sem = 0.3
        else:
            delta_sem = 0.5 + similarity * 0.5
        
        return delta_sem
    
    def _prepare_text_for_classifier(self, text: str, context: str) -> str:
        """Prepare text input for classifiers"""
        if context:
            return f"{context} {text}"
        return text
    
    def _get_pga_prediction(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Get PGA prediction from classifier"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            pga_label, pga_info = self.pga_classifier.predict_pga(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        return pga_label, {
            pga_label: pga_info["pga_confidence"]
        }
    
    def _get_politeness_prediction(self, text: str) -> Tuple[str, float]:
        """Get politeness prediction from classifier"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            pred, probs = self.politeness_classifier.predict(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        strategy = self.politeness_classifier.get_strategy_name(pred.item())
        confidence = probs[0, pred].item()
        
        return strategy, confidence
    
    def _get_empathy_prediction(self, text: str) -> Tuple[str, float]:
        """Get empathy prediction from classifier"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            pred, probs = self.empathy_classifier.predict(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        strategy = self.empathy_classifier.get_strategy_name(pred.item())
        confidence = probs[0, pred].item()
        
        return strategy, confidence
    
    def _profile_to_pga_class(self, user_profile: Dict[str, str]) -> str:
        """Convert user profile to PGA class string"""
        
        persona_map = {
            "openness": "O",
            "conscientiousness": "C",
            "extraversion": "E",
            "agreeableness": "A",
            "neuroticism": "N"
        }
        
        gender_map = {
            "male": "M",
            "female": "F"
        }
        
        age_map = {
            "younger": "Y",
            "middle_aged": "M",
            "older": "O"
        }
        
        persona = persona_map.get(user_profile.get("persona", ""), "O")
        gender = gender_map.get(user_profile.get("gender", ""), "M")
        age = age_map.get(user_profile.get("age_group", ""), "M")
        
        return f"{persona}_{gender}_{age}"


class RewardAggregator:
    """Aggregate rewards across batches for PPO training"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.reset()
    
    def reset(self):
        """Reset aggregated rewards"""
        self.rewards = []
        self.component_scores = []
    
    def add_reward(
        self,
        reward: float,
        components: Dict[str, float]
    ):
        """Add a single reward"""
        self.rewards.append(reward)
        self.component_scores.append(components)
    
    def get_batch_rewards(self) -> torch.Tensor:
        """Get rewards as tensor for current batch"""
        if not self.rewards:
            return torch.zeros(1)
        
        return torch.tensor(self.rewards, dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics of rewards"""
        if not self.rewards:
            return {}
        
        stats = {
            "mean_reward": np.mean(self.rewards),
            "std_reward": np.std(self.rewards),
            "min_reward": np.min(self.rewards),
            "max_reward": np.max(self.rewards)
        }
        
        # Add component statistics
        if self.component_scores:
            component_keys = self.component_scores[0].keys()
            for key in component_keys:
                values = [cs[key] for cs in self.component_scores]
                stats[f"mean_{key}"] = np.mean(values)
        
        return stats


class AdaptiveRewardScheduler:
    """Adaptive scheduling for reward weights"""
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.01
    ):
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.history = []
    
    def update_weights(
        self,
        performance_metrics: Dict[str, float]
    ):
        """Update weights based on performance"""
        
        # Store history
        self.history.append({
            "metrics": performance_metrics,
            "weights": self.weights.copy()
        })
        
        # Adaptive weight adjustment based on performance
        # If task-relevance metrics are low, increase their weights
        if "mean_user_profile_alignment" in performance_metrics:
            if performance_metrics["mean_user_profile_alignment"] < 0.5:
                self.weights["w1"] = min(0.5, self.weights["w1"] + self.adaptation_rate)
        
        if "mean_politeness_adherence" in performance_metrics:
            if performance_metrics["mean_politeness_adherence"] < 0.5:
                self.weights["w2"] = min(0.5, self.weights["w2"] + self.adaptation_rate)
        
        if "mean_empathy_consistency" in performance_metrics:
            if performance_metrics["mean_empathy_consistency"] < 0.5:
                self.weights["w3"] = min(0.5, self.weights["w3"] + self.adaptation_rate)
        
        # Normalize weights
        total = self.weights["w1"] + self.weights["w2"] + self.weights["w3"]
        self.weights["w1"] /= total
        self.weights["w2"] /= total
        self.weights["w3"] /= total
        
        # Adjust gamma based on overall performance
        if "mean_reward" in performance_metrics:
            if performance_metrics["mean_reward"] < 0.4:
                # Increase focus on task-relevance
                self.weights["gamma"] = min(0.9, self.weights["gamma"] + self.adaptation_rate)
            elif performance_metrics["mean_reward"] > 0.7:
                # Balance between task-relevance and smoothness
                self.weights["gamma"] = max(0.5, self.weights["gamma"] - self.adaptation_rate)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weight configuration"""
        return self.weights.copy()


if __name__ == "__main__":
    # Example usage
    from transformers import RobertaTokenizer
    from src.models.classifiers import (
        PGAClassifier,
        PolitenessStrategyClassifier,
        EmpathyStrategyClassifier
    )
    
    # Initialize components
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Create reward calculator
    reward_calc = RewardCalculator(
        pga_clf,
        pol_clf,
        emp_clf,
        tokenizer
    )
    
    # Example inputs
    user_profile = {
        "persona": "extraversion",
        "gender": "female",
        "age_group": "middle_aged",
        "disability_type": "spinal_cord_injury"
    }
    
    context = "Discussion about physical therapy exercises"
    previous_utterance = "I need help with my daily exercises."
    generated_response = "I understand you need assistance with exercises. Let me help you with a personalized routine."
    target_response = "I completely understand your needs. Let's work together on creating an exercise plan that suits you."
    
    # Calculate reward
    reward, components = reward_calc.calculate_reward(
        generated_response,
        target_response,
        user_profile,
        context,
        previous_utterance
    )
    
    print(f"Total Reward: {reward:.4f}")
    print("\nComponent Scores:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")