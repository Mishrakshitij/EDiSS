"""
Annotation module for PDCare dataset
Handles politeness and empathy strategy annotations
"""

import json
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.nn import functional as F


class PolitenessStrategy(Enum):
    """Politeness strategies based on Brown & Levinson (1987)"""
    POSITIVE = "positive_politeness"
    NEGATIVE = "negative_politeness"
    BALD_ON_RECORD = "bald_on_record"


class EmpathyStrategy(Enum):
    """Eight empathy strategies for disability support"""
    GENUINE_ENGAGEMENT = "genuine_engagement"
    PRIVACY_ASSURANCE = "privacy_assurance"
    FORWARD_FOCUS = "forward_focus_encouragement"
    COMPASSIONATE_VALIDATION = "compassionate_validation"
    PRACTICAL_ASSISTANCE = "practical_assistance"
    CONTINUOUS_SUPPORT = "continuous_support"
    STRENGTH_BASED = "strength_based_support"
    NO_STRATEGY = "no_strategy"


@dataclass
class AnnotatedUtterance:
    """Annotated utterance with strategies"""
    text: str
    speaker: str
    turn: int
    politeness_strategy: Optional[PolitenessStrategy] = None
    empathy_strategy: Optional[EmpathyStrategy] = None
    confidence_scores: Optional[Dict[str, float]] = None


class StrategyClassifier:
    """Base classifier for strategies"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None
    
    def load_model(self, model_path: str):
        """Load pre-trained classifier model"""
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, text: str) -> Tuple[int, Dict[str, float]]:
        """Predict strategy for given text"""
        if not self.model:
            # Return random prediction if model not loaded
            return 0, {"confidence": 0.5}
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence_scores = {
            f"class_{i}": prob.item() 
            for i, prob in enumerate(probs[0])
        }
        
        return pred_class, confidence_scores


class PolitenessClassifier(StrategyClassifier):
    """Classifier for politeness strategies"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self.strategies = list(PolitenessStrategy)
    
    def classify(self, text: str) -> Tuple[PolitenessStrategy, float]:
        """Classify politeness strategy"""
        # Rule-based classification as fallback
        if not self.model:
            return self._rule_based_classification(text)
        
        pred_class, scores = self.predict(text)
        strategy = self.strategies[pred_class]
        confidence = scores[f"class_{pred_class}"]
        
        return strategy, confidence
    
    def _rule_based_classification(self, text: str) -> Tuple[PolitenessStrategy, float]:
        """Simple rule-based classification"""
        text_lower = text.lower()
        
        # Positive politeness indicators
        positive_markers = [
            "appreciate", "thank", "glad", "happy", "wonderful",
            "great", "excellent", "together", "we", "our"
        ]
        
        # Negative politeness indicators
        negative_markers = [
            "would you", "could you", "if possible", "perhaps",
            "might", "sorry", "excuse", "please", "kindly"
        ]
        
        positive_count = sum(1 for marker in positive_markers if marker in text_lower)
        negative_count = sum(1 for marker in negative_markers if marker in text_lower)
        
        if positive_count > negative_count:
            return PolitenessStrategy.POSITIVE, 0.7
        elif negative_count > positive_count:
            return PolitenessStrategy.NEGATIVE, 0.7
        else:
            return PolitenessStrategy.BALD_ON_RECORD, 0.6


class EmpathyClassifier(StrategyClassifier):
    """Classifier for empathy strategies"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self.strategies = list(EmpathyStrategy)
    
    def classify(self, text: str) -> Tuple[EmpathyStrategy, float]:
        """Classify empathy strategy"""
        # Rule-based classification as fallback
        if not self.model:
            return self._rule_based_classification(text)
        
        pred_class, scores = self.predict(text)
        strategy = self.strategies[pred_class]
        confidence = scores[f"class_{pred_class}"]
        
        return strategy, confidence
    
    def _rule_based_classification(self, text: str) -> Tuple[EmpathyStrategy, float]:
        """Rule-based empathy classification"""
        text_lower = text.lower()
        
        # Strategy markers
        strategy_markers = {
            EmpathyStrategy.GENUINE_ENGAGEMENT: [
                "understand", "tell me more", "listening", "hear",
                "appreciate you sharing", "interested"
            ],
            EmpathyStrategy.PRIVACY_ASSURANCE: [
                "confidential", "private", "secure", "safe space",
                "between us", "trust"
            ],
            EmpathyStrategy.FORWARD_FOCUS: [
                "future", "goals", "progress", "positive", "forward",
                "opportunity", "growth", "achieve"
            ],
            EmpathyStrategy.COMPASSIONATE_VALIDATION: [
                "valid", "understand how you feel", "difficult",
                "challenging", "it's okay", "normal to feel"
            ],
            EmpathyStrategy.PRACTICAL_ASSISTANCE: [
                "help", "assist", "support", "resources", "guidance",
                "advice", "solution", "strategy"
            ],
            EmpathyStrategy.CONTINUOUS_SUPPORT: [
                "always here", "ongoing", "continue", "whenever",
                "available", "not alone"
            ],
            EmpathyStrategy.STRENGTH_BASED: [
                "strength", "capable", "ability", "empower",
                "resilient", "confident", "independence"
            ]
        }
        
        scores = {}
        for strategy, markers in strategy_markers.items():
            score = sum(1 for marker in markers if marker in text_lower)
            scores[strategy] = score
        
        if max(scores.values()) == 0:
            return EmpathyStrategy.NO_STRATEGY, 0.5
        
        best_strategy = max(scores, key=scores.get)
        confidence = min(0.9, 0.5 + (scores[best_strategy] * 0.1))
        
        return best_strategy, confidence


class PGAClassifier:
    """Classifier for Persona-Gender-Age combinations"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        
        # PGA combinations (5 personas × 2 genders × 3 age groups = 30 classes)
        self.pga_classes = self._create_pga_classes()
        
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None
    
    def _create_pga_classes(self) -> List[str]:
        """Create all PGA class combinations"""
        personas = ["O", "C", "E", "A", "N"]
        genders = ["M", "F"]
        ages = ["Y", "M", "O"]  # Younger, Middle-aged, Older
        
        classes = []
        for p in personas:
            for g in genders:
                for a in ages:
                    classes.append(f"{p}_{g}_{a}")
        
        return classes
    
    def load_model(self, model_path: str):
        """Load pre-trained PGA classifier"""
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=30
        )
        model.to(self.device)
        model.eval()
        return model
    
    def classify(self, text: str, context: Optional[str] = None) -> Tuple[str, float]:
        """Classify PGA for given text"""
        if not self.model:
            # Return random PGA class if model not loaded
            import random
            pga_class = random.choice(self.pga_classes)
            return pga_class, 0.5
        
        # Combine text with context if provided
        input_text = f"{context} {text}" if context else text
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()
        
        return self.pga_classes[pred_idx], confidence


class DialogueAnnotator:
    """Annotates dialogues with strategies"""
    
    def __init__(
        self,
        politeness_model_path: Optional[str] = None,
        empathy_model_path: Optional[str] = None,
        pga_model_path: Optional[str] = None
    ):
        self.politeness_classifier = PolitenessClassifier(politeness_model_path)
        self.empathy_classifier = EmpathyClassifier(empathy_model_path)
        self.pga_classifier = PGAClassifier(pga_model_path)
    
    def annotate_utterance(
        self,
        utterance: Dict[str, str],
        context: Optional[str] = None
    ) -> AnnotatedUtterance:
        """Annotate a single utterance"""
        
        text = utterance["text"]
        speaker = utterance["speaker"]
        turn = utterance.get("turn", 0)
        
        annotated = AnnotatedUtterance(
            text=text,
            speaker=speaker,
            turn=turn
        )
        
        # Only annotate doctor utterances with strategies
        if speaker == "doctor":
            pol_strategy, pol_conf = self.politeness_classifier.classify(text)
            emp_strategy, emp_conf = self.empathy_classifier.classify(text)
            
            annotated.politeness_strategy = pol_strategy
            annotated.empathy_strategy = emp_strategy
            annotated.confidence_scores = {
                "politeness_confidence": pol_conf,
                "empathy_confidence": emp_conf
            }
        
        return annotated
    
    def annotate_dialogue(self, dialogue: Dict) -> Dict:
        """Annotate an entire dialogue"""
        
        annotated_dialogue = dialogue.copy()
        annotated_utterances = []
        
        context = ""
        for utterance in dialogue["utterances"]:
            # Annotate utterance
            annotated_utt = self.annotate_utterance(utterance, context)
            
            # Convert to dict for storage
            utt_dict = {
                "text": annotated_utt.text,
                "speaker": annotated_utt.speaker,
                "turn": annotated_utt.turn
            }
            
            if annotated_utt.politeness_strategy:
                utt_dict["politeness_strategy"] = annotated_utt.politeness_strategy.value
            if annotated_utt.empathy_strategy:
                utt_dict["empathy_strategy"] = annotated_utt.empathy_strategy.value
            if annotated_utt.confidence_scores:
                utt_dict["confidence_scores"] = annotated_utt.confidence_scores
            
            annotated_utterances.append(utt_dict)
            
            # Update context
            context += f" {utterance['text']}"
            if len(context) > 1000:  # Limit context length
                context = context[-1000:]
        
        # Classify PGA for the dialogue
        full_text = " ".join([u["text"] for u in dialogue["utterances"]])
        pga_class, pga_conf = self.pga_classifier.classify(full_text)
        
        annotated_dialogue["utterances"] = annotated_utterances
        annotated_dialogue["pga_class"] = pga_class
        annotated_dialogue["pga_confidence"] = pga_conf
        
        return annotated_dialogue
    
    def annotate_dataset(
        self,
        dataset_path: str,
        output_path: str,
        batch_size: int = 32
    ):
        """Annotate entire dataset"""
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        annotated_data = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            for dialogue in batch:
                annotated = self.annotate_dialogue(dialogue)
                annotated_data.append(annotated)
            
            print(f"Annotated {min(i+batch_size, len(data))}/{len(data)} dialogues")
        
        with open(output_path, 'w') as f:
            json.dump(annotated_data, f, indent=2)
        
        print(f"Saved annotated dataset to {output_path}")


class QualityChecker:
    """Quality checking for annotations"""
    
    def __init__(self):
        self.metrics = {
            "topic_consistency": [],
            "context_adequacy": [],
            "fluency": []
        }
    
    def check_topic_consistency(self, dialogue: Dict) -> float:
        """Check if dialogue stays on topic"""
        support_area = dialogue.get("support_area", "")
        disability_type = dialogue["user_profile"].get("disability_type", "")
        
        relevance_count = 0
        total_utterances = len(dialogue["utterances"])
        
        for utt in dialogue["utterances"]:
            text_lower = utt["text"].lower()
            if support_area.lower() in text_lower or disability_type.lower() in text_lower:
                relevance_count += 1
        
        return relevance_count / total_utterances if total_utterances > 0 else 0
    
    def check_context_adequacy(self, dialogue: Dict) -> float:
        """Check if responses are contextually appropriate"""
        score = 1.0
        
        for i in range(1, len(dialogue["utterances"])):
            prev_utt = dialogue["utterances"][i-1]["text"]
            curr_utt = dialogue["utterances"][i]["text"]
            
            # Simple check: responses shouldn't be identical
            if prev_utt == curr_utt:
                score -= 0.1
            
            # Check for question-answer patterns
            if "?" in prev_utt and dialogue["utterances"][i]["speaker"] == "doctor":
                if not any(word in curr_utt.lower() for word in 
                          ["yes", "no", "sure", "understand", "help", "can", "will"]):
                    score -= 0.05
        
        return max(0, score)
    
    def check_fluency(self, dialogue: Dict) -> float:
        """Check linguistic fluency"""
        import re
        
        total_score = 0
        total_utterances = len(dialogue["utterances"])
        
        for utt in dialogue["utterances"]:
            text = utt["text"]
            utt_score = 1.0
            
            # Check for proper capitalization
            if text and not text[0].isupper():
                utt_score -= 0.1
            
            # Check for proper punctuation
            if text and text[-1] not in ".!?":
                utt_score -= 0.1
            
            # Check for repeated words
            words = text.split()
            for i in range(len(words)-1):
                if words[i] == words[i+1] and words[i] not in ["the", "a", "an"]:
                    utt_score -= 0.05
            
            total_score += utt_score
        
        return total_score / total_utterances if total_utterances > 0 else 0
    
    def evaluate_dialogue(self, dialogue: Dict) -> Dict[str, float]:
        """Evaluate dialogue quality"""
        return {
            "topic_consistency": self.check_topic_consistency(dialogue),
            "context_adequacy": self.check_context_adequacy(dialogue),
            "fluency": self.check_fluency(dialogue)
        }


if __name__ == "__main__":
    # Example usage
    annotator = DialogueAnnotator()
    
    # Example dialogue
    example_dialogue = {
        "dialogue_id": "test_001",
        "user_profile": {
            "persona": "extraversion",
            "gender": "female",
            "age_group": "middle_aged",
            "disability_type": "spinal_cord_injury"
        },
        "support_area": "physical_therapy",
        "utterances": [
            {"speaker": "user", "text": "I need help with exercises for my condition.", "turn": 0},
            {"speaker": "doctor", "text": "I understand you're dealing with a spinal cord injury. I'm here to help you with appropriate exercises.", "turn": 1}
        ]
    }
    
    annotated = annotator.annotate_dialogue(example_dialogue)
    print(json.dumps(annotated, indent=2))