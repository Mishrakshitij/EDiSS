"""
Evaluation metrics for EDiSS
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import re
from collections import Counter


class EDiSSEvaluator:
    """Comprehensive evaluator for EDiSS model"""
    
    def __init__(
        self,
        model,
        tokenizer,
        pga_classifier=None,
        politeness_classifier=None,
        empathy_classifier=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pga_classifier = pga_classifier
        self.politeness_classifier = politeness_classifier
        self.empathy_classifier = empathy_classifier
        self.device = device
        self.rouge = Rouge()
        
    def evaluate_task_relevance(
        self,
        test_dialogues: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate task-relevance metrics (UPC, PSA, ESA)"""
        
        results = {
            "user_profile_consistency": [],
            "politeness_strategy_accuracy": [],
            "empathy_strategy_accuracy": []
        }
        
        for dialogue in test_dialogues:
            context = ""
            user_profile = dialogue["user_profile"]
            
            for i in range(0, len(dialogue["utterances"]) - 1, 2):
                if i + 1 >= len(dialogue["utterances"]):
                    break
                
                user_utt = dialogue["utterances"][i]
                doctor_utt = dialogue["utterances"][i + 1]
                
                # Generate response
                generated = self.model.generate_response(
                    context,
                    user_profile,
                    user_utt["text"]
                )
                
                # Evaluate UPC
                if self.pga_classifier:
                    upc_score = self._evaluate_upc(
                        generated,
                        doctor_utt["text"],
                        user_profile
                    )
                    results["user_profile_consistency"].append(upc_score)
                
                # Evaluate PSA
                if self.politeness_classifier and "politeness_strategy" in doctor_utt:
                    psa_score = self._evaluate_psa(
                        generated,
                        doctor_utt["politeness_strategy"]
                    )
                    results["politeness_strategy_accuracy"].append(psa_score)
                
                # Evaluate ESA
                if self.empathy_classifier and "empathy_strategy" in doctor_utt:
                    esa_score = self._evaluate_esa(
                        generated,
                        doctor_utt["empathy_strategy"]
                    )
                    results["empathy_strategy_accuracy"].append(esa_score)
                
                # Update context
                context += f"\nUser: {user_utt['text']}\nDoctor: {generated}"
                if len(context) > 1000:
                    context = context[-1000:]
        
        # Calculate averages
        metrics = {}
        for key, values in results.items():
            if values:
                metrics[key] = np.mean(values)
            else:
                metrics[key] = 0.0
        
        return metrics
    
    def _evaluate_upc(
        self,
        generated: str,
        target: str,
        user_profile: Dict[str, str]
    ) -> float:
        """Evaluate user profile consistency"""
        
        # Get PGA prediction for generated text
        inputs = self.tokenizer(
            generated,
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
        
        # Check if predicted PGA matches user profile
        target_pga = self._profile_to_pga(user_profile)
        
        return 1.0 if pga_label == target_pga else 0.0
    
    def _evaluate_psa(
        self,
        generated: str,
        target_strategy: str
    ) -> float:
        """Evaluate politeness strategy accuracy"""
        
        inputs = self.tokenizer(
            generated,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            pred, _ = self.politeness_classifier.predict(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        pred_strategy = self.politeness_classifier.get_strategy_name(pred.item())
        
        return 1.0 if pred_strategy == target_strategy else 0.0
    
    def _evaluate_esa(
        self,
        generated: str,
        target_strategy: str
    ) -> float:
        """Evaluate empathy strategy accuracy"""
        
        inputs = self.tokenizer(
            generated,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            pred, _ = self.empathy_classifier.predict(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        pred_strategy = self.empathy_classifier.get_strategy_name(pred.item())
        
        return 1.0 if pred_strategy == target_strategy else 0.0
    
    def _profile_to_pga(self, user_profile: Dict[str, str]) -> str:
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
    
    def evaluate_language_quality(
        self,
        test_dialogues: List[Dict],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Evaluate language quality metrics"""
        
        import random
        sampled = random.sample(test_dialogues, min(num_samples, len(test_dialogues)))
        
        metrics = {
            "perplexity": [],
            "response_length": [],
            "non_repetitiveness": [],
            "bleu": [],
            "rouge_l": []
        }
        
        for dialogue in sampled:
            context = ""
            user_profile = dialogue["user_profile"]
            
            for i in range(0, len(dialogue["utterances"]) - 1, 2):
                if i + 1 >= len(dialogue["utterances"]):
                    break
                
                user_utt = dialogue["utterances"][i]
                doctor_utt = dialogue["utterances"][i + 1]
                
                # Generate response
                generated = self.model.generate_response(
                    context,
                    user_profile,
                    user_utt["text"]
                )
                
                # Calculate perplexity (simplified)
                ppl = self._calculate_perplexity(generated)
                metrics["perplexity"].append(ppl)
                
                # Response length
                length = len(generated.split())
                metrics["response_length"].append(length)
                
                # Non-repetitiveness
                if i > 0:
                    prev_response = dialogue["utterances"][i-1]["text"] if i > 0 else ""
                    rep_score = self._calculate_repetitiveness(generated, prev_response)
                    metrics["non_repetitiveness"].append(rep_score)
                
                # BLEU score
                reference = doctor_utt["text"].split()
                hypothesis = generated.split()
                bleu = sentence_bleu(
                    [reference],
                    hypothesis,
                    smoothing_function=SmoothingFunction().method1
                )
                metrics["bleu"].append(bleu)
                
                # ROUGE-L score
                try:
                    scores = self.rouge.get_scores(generated, doctor_utt["text"])[0]
                    metrics["rouge_l"].append(scores["rouge-l"]["f"])
                except:
                    metrics["rouge_l"].append(0.0)
                
                # Update context
                context += f"\nUser: {user_utt['text']}\nDoctor: {generated}"
                if len(context) > 1000:
                    context = context[-1000:]
        
        # Calculate averages
        result = {}
        for key, values in metrics.items():
            if values:
                if key == "perplexity":
                    result[key] = np.exp(np.mean(np.log(values)))  # Geometric mean
                else:
                    result[key] = np.mean(values)
            else:
                result[key] = 0.0
        
        return result
    
    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for generated text"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                labels=inputs["input_ids"]
            )
            
            loss = outputs["loss"]
            perplexity = torch.exp(loss).item()
        
        return min(perplexity, 1000)  # Cap at 1000 to avoid overflow
    
    def _calculate_repetitiveness(
        self,
        current: str,
        previous: str
    ) -> float:
        """Calculate non-repetitiveness score"""
        
        # Tokenize
        current_tokens = set(current.lower().split())
        previous_tokens = set(previous.lower().split())
        
        # Calculate Jaccard similarity
        intersection = current_tokens.intersection(previous_tokens)
        union = current_tokens.union(previous_tokens)
        
        if not union:
            return 1.0
        
        similarity = len(intersection) / len(union)
        
        # Non-repetitiveness is inverse of similarity
        return 1.0 - similarity
    
    def evaluate_dialogue_coherence(
        self,
        test_dialogues: List[Dict],
        num_samples: int = 50
    ) -> Dict[str, float]:
        """Evaluate dialogue coherence and flow"""
        
        import random
        sampled = random.sample(test_dialogues, min(num_samples, len(test_dialogues)))
        
        coherence_scores = []
        topic_consistency_scores = []
        response_relevance_scores = []
        
        for dialogue in sampled:
            context = ""
            user_profile = dialogue["user_profile"]
            support_area = dialogue.get("support_area", "")
            
            dialogue_coherence = []
            
            for i in range(0, len(dialogue["utterances"]) - 1, 2):
                if i + 1 >= len(dialogue["utterances"]):
                    break
                
                user_utt = dialogue["utterances"][i]
                doctor_utt = dialogue["utterances"][i + 1]
                
                # Generate response
                generated = self.model.generate_response(
                    context,
                    user_profile,
                    user_utt["text"]
                )
                
                # Evaluate coherence with previous context
                if context:
                    coherence = self._evaluate_coherence(generated, context)
                    dialogue_coherence.append(coherence)
                
                # Evaluate topic consistency
                if support_area:
                    topic_score = self._evaluate_topic_consistency(
                        generated,
                        support_area,
                        user_profile.get("disability_type", "")
                    )
                    topic_consistency_scores.append(topic_score)
                
                # Evaluate response relevance to user query
                relevance = self._evaluate_response_relevance(
                    generated,
                    user_utt["text"]
                )
                response_relevance_scores.append(relevance)
                
                # Update context
                context += f"\nUser: {user_utt['text']}\nDoctor: {generated}"
                if len(context) > 1000:
                    context = context[-1000:]
            
            if dialogue_coherence:
                coherence_scores.append(np.mean(dialogue_coherence))
        
        return {
            "dialogue_coherence": np.mean(coherence_scores) if coherence_scores else 0.0,
            "topic_consistency": np.mean(topic_consistency_scores) if topic_consistency_scores else 0.0,
            "response_relevance": np.mean(response_relevance_scores) if response_relevance_scores else 0.0
        }
    
    def _evaluate_coherence(self, response: str, context: str) -> float:
        """Evaluate coherence between response and context"""
        
        # Simple coherence based on shared vocabulary
        response_tokens = set(response.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove stop words (simplified)
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "as", "are", "was", "were", "i", "you", "he", "she", "it", "we", "they"}
        response_tokens -= stop_words
        context_tokens -= stop_words
        
        if not context_tokens:
            return 0.5
        
        overlap = len(response_tokens.intersection(context_tokens))
        coherence = overlap / len(context_tokens)
        
        return min(coherence * 2, 1.0)  # Scale and cap at 1.0
    
    def _evaluate_topic_consistency(
        self,
        response: str,
        support_area: str,
        disability_type: str
    ) -> float:
        """Evaluate if response stays on topic"""
        
        response_lower = response.lower()
        
        # Check for topic keywords
        topic_mentioned = support_area.lower() in response_lower or \
                         disability_type.lower() in response_lower
        
        # Check for related keywords
        related_keywords = {
            "physical_therapy": ["exercise", "therapy", "movement", "stretch"],
            "emotional_support": ["feeling", "emotion", "support", "cope"],
            "pain_management": ["pain", "discomfort", "relief", "manage"],
            "mobility_aids": ["wheelchair", "cane", "walker", "assistive"],
            "daily_living": ["daily", "routine", "activity", "task"]
        }
        
        area_keywords = related_keywords.get(support_area, [])
        keyword_found = any(keyword in response_lower for keyword in area_keywords)
        
        if topic_mentioned:
            return 1.0
        elif keyword_found:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_response_relevance(
        self,
        response: str,
        user_query: str
    ) -> float:
        """Evaluate if response is relevant to user query"""
        
        # Check for question answering
        if "?" in user_query:
            # Response should contain answer indicators
            answer_indicators = ["yes", "no", "can", "will", "should", "would", "help", "suggest", "recommend"]
            has_answer = any(indicator in response.lower() for indicator in answer_indicators)
            
            if has_answer:
                return 1.0
            else:
                return 0.5
        
        # Check for keyword overlap
        query_tokens = set(user_query.lower().split())
        response_tokens = set(response.lower().split())
        
        overlap = len(query_tokens.intersection(response_tokens))
        
        if len(query_tokens) > 0:
            relevance = overlap / len(query_tokens)
            return min(relevance * 1.5, 1.0)
        
        return 0.5
    
    def comprehensive_evaluation(
        self,
        test_dialogues: List[Dict],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Perform comprehensive evaluation"""
        
        print("Evaluating task relevance...")
        task_metrics = self.evaluate_task_relevance(test_dialogues[:num_samples])
        
        print("Evaluating language quality...")
        language_metrics = self.evaluate_language_quality(test_dialogues, num_samples)
        
        print("Evaluating dialogue coherence...")
        coherence_metrics = self.evaluate_dialogue_coherence(test_dialogues, num_samples)
        
        # Combine all metrics
        all_metrics = {
            **task_metrics,
            **language_metrics,
            **coherence_metrics
        }
        
        return all_metrics


class HumanEvaluationSimulator:
    """Simulate human evaluation scores for testing"""
    
    def __init__(self):
        self.criteria = [
            "persona_accuracy",
            "gender_age_accuracy",
            "politeness_accuracy",
            "empathy_accuracy",
            "fluency",
            "consistency",
            "non_repetitiveness"
        ]
    
    def simulate_evaluation(
        self,
        generated_response: str,
        target_response: str,
        user_profile: Dict[str, str]
    ) -> Dict[str, float]:
        """Simulate human evaluation scores"""
        
        scores = {}
        
        # Simulate scores based on response similarity and quality
        base_score = self._calculate_base_score(generated_response, target_response)
        
        # Add noise to simulate human variation
        for criterion in self.criteria:
            score = base_score + np.random.normal(0, 0.5)
            score = max(1, min(5, score))  # Clamp to 1-5 scale
            scores[criterion] = score
        
        return scores
    
    def _calculate_base_score(
        self,
        generated: str,
        target: str
    ) -> float:
        """Calculate base score based on similarity"""
        
        # Simple word overlap similarity
        gen_words = set(generated.lower().split())
        tgt_words = set(target.lower().split())
        
        if not tgt_words:
            return 3.0
        
        overlap = len(gen_words.intersection(tgt_words))
        similarity = overlap / len(tgt_words)
        
        # Map similarity to 1-5 scale
        base_score = 1 + similarity * 4
        
        # Adjust for length
        len_ratio = len(generated.split()) / max(len(target.split()), 1)
        if 0.5 < len_ratio < 2.0:
            base_score += 0.5
        
        return min(base_score, 5.0)


if __name__ == "__main__":
    # Example usage
    from src.models.ediss_model import EDiSSModel
    from src.models.classifiers import (
        PGAClassifier,
        PolitenessStrategyClassifier,
        EmpathyStrategyClassifier
    )
    from transformers import AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model and classifiers
    model = EDiSSModel(device=device)
    tokenizer = model.tokenizer
    
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Initialize evaluator
    evaluator = EDiSSEvaluator(
        model,
        tokenizer,
        pga_clf,
        pol_clf,
        emp_clf,
        device=device
    )
    
    # Example dialogue for testing
    test_dialogue = {
        "user_profile": {
            "persona": "extraversion",
            "gender": "female",
            "age_group": "middle_aged",
            "disability_type": "spinal_cord_injury"
        },
        "support_area": "physical_therapy",
        "utterances": [
            {"speaker": "user", "text": "I need help with exercises.", "turn": 0},
            {"speaker": "doctor", "text": "I can help you with exercises suitable for your condition.", "turn": 1,
             "politeness_strategy": "positive_politeness",
             "empathy_strategy": "practical_assistance"}
        ]
    }
    
    # Evaluate
    print("Running evaluation...")
    metrics = evaluator.comprehensive_evaluation([test_dialogue], num_samples=1)
    
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")