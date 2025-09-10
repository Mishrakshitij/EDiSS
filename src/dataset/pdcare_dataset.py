"""
PDCare Dataset Creation Module
This module handles the creation of the PDCare dataset for disability support dialogues.
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from tqdm import tqdm


class Persona(Enum):
    """OCEAN personality traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class Gender(Enum):
    """Gender categories"""
    MALE = "male"
    FEMALE = "female"


class AgeGroup(Enum):
    """Age group categories"""
    YOUNGER = "younger"
    MIDDLE_AGED = "middle_aged"
    OLDER = "older"


class DisabilityType(Enum):
    """Physical disability types"""
    MOBILITY_IMPAIRMENTS = "mobility_impairments"
    VISUAL_IMPAIRMENTS = "visual_impairments"
    HEARING_IMPAIRMENTS = "hearing_impairments"
    SPEECH_IMPAIRMENTS = "speech_impairments"
    NEUROLOGICAL_DISORDERS = "neurological_disorders"
    SPINAL_CORD_INJURIES = "spinal_cord_injuries"
    AMPUTATIONS = "amputations"
    ORTHOPEDIC_DISABILITIES = "orthopedic_disabilities"
    CEREBRAL_PALSY = "cerebral_palsy"
    MUSCULAR_DYSTROPHY = "muscular_dystrophy"
    BALANCE_GAIT_DISORDERS = "balance_gait_disorders"
    CHRONIC_PAIN = "chronic_pain"
    AGING_RELATED_DISABILITIES = "aging_related_disabilities"


class SupportArea(Enum):
    """Support areas for disabilities"""
    ACCESSIBILITY_INFORMATION = "accessibility_information"
    TRAVEL_TIPS = "travel_tips"
    ADVOCACY_RIGHTS = "advocacy_rights"
    FINANCIAL_INSURANCE = "financial_insurance"
    MOBILITY_AIDS = "mobility_aids"
    HOME_MODIFICATIONS = "home_modifications"
    PHYSICAL_THERAPY = "physical_therapy"
    ASSISTIVE_TECHNOLOGY = "assistive_technology"
    PAIN_MANAGEMENT = "pain_management"
    DAILY_LIVING = "daily_living"
    EMOTIONAL_SUPPORT = "emotional_support"
    EMPLOYMENT_EDUCATION = "employment_education"
    SOCIAL_INTERACTION = "social_interaction"
    FITNESS_RECREATION = "fitness_recreation"
    PEER_SUPPORT = "peer_support"
    PARENTING = "parenting"
    LIFE_TRANSITIONS = "life_transitions"


@dataclass
class UserProfile:
    """User profile information"""
    persona: Persona
    gender: Gender
    age_group: AgeGroup
    disability_type: DisabilityType
    
    def to_dict(self):
        return {
            "persona": self.persona.value,
            "gender": self.gender.value,
            "age_group": self.age_group.value,
            "disability_type": self.disability_type.value
        }


@dataclass
class Dialogue:
    """Dialogue data structure"""
    dialogue_id: str
    user_profile: UserProfile
    support_area: SupportArea
    utterances: List[Dict[str, str]]
    metadata: Optional[Dict] = None
    
    def to_dict(self):
        return {
            "dialogue_id": self.dialogue_id,
            "user_profile": self.user_profile.to_dict(),
            "support_area": self.support_area.value,
            "utterances": self.utterances,
            "metadata": self.metadata or {}
        }


class PDCareDatasetCreator:
    """Creates the PDCare dataset"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.dialogues = []
        
    def create_prompt(
        self,
        user_profile: UserProfile,
        support_area: SupportArea,
        seed_utterances: Optional[List[Dict]] = None
    ) -> str:
        """Create a prompt for dialogue generation"""
        
        prompt = f"""
        Generate a supportive dialogue between a healthcare professional and a user with the following profile:
        
        User Profile:
        - Personality: {user_profile.persona.value}
        - Gender: {user_profile.gender.value}
        - Age Group: {user_profile.age_group.value}
        - Disability Type: {user_profile.disability_type.value}
        - Support Area: {support_area.value}
        
        Instructions:
        1. The dialogue should be empathetic and supportive
        2. Use appropriate politeness strategies
        3. Address the specific needs related to the disability
        4. Maintain consistency with the user's personality traits
        5. Provide practical assistance and emotional support
        """
        
        if seed_utterances:
            prompt += "\n\nSeed utterances:\n"
            for utterance in seed_utterances:
                prompt += f"{utterance['speaker']}: {utterance['text']}\n"
            prompt += "\nContinue this dialogue naturally..."
        
        return prompt
    
    def generate_seed_utterances(
        self,
        user_profile: UserProfile,
        support_area: SupportArea,
        num_turns: int = 4
    ) -> List[Dict[str, str]]:
        """Generate seed utterances for dialogue initialization"""
        
        seed_utterances = []
        
        # Opening from user based on personality
        if user_profile.persona == Persona.EXTRAVERSION:
            opening = f"Hi! I'm dealing with {user_profile.disability_type.value} and I really need help with {support_area.value}. Can you assist me?"
        elif user_profile.persona == Persona.NEUROTICISM:
            opening = f"I'm quite worried about my {user_profile.disability_type.value}. I don't know how to handle {support_area.value}."
        else:
            opening = f"Hello, I need assistance with {support_area.value} related to my {user_profile.disability_type.value}."
        
        seed_utterances.append({
            "speaker": "user",
            "text": opening,
            "turn": 0
        })
        
        # Doctor's response
        seed_utterances.append({
            "speaker": "doctor",
            "text": f"I understand you're dealing with {user_profile.disability_type.value}. I'm here to help you with {support_area.value}. Could you tell me more about your specific challenges?",
            "turn": 1
        })
        
        # Continue based on support area
        if support_area == SupportArea.PHYSICAL_THERAPY:
            seed_utterances.append({
                "speaker": "user",
                "text": "I've been experiencing difficulty with daily movements and would like to know about exercises that could help.",
                "turn": 2
            })
            seed_utterances.append({
                "speaker": "doctor",
                "text": "I can definitely help you with appropriate exercises. Let's start by understanding your current mobility level and any pain you might be experiencing.",
                "turn": 3
            })
        elif support_area == SupportArea.EMOTIONAL_SUPPORT:
            seed_utterances.append({
                "speaker": "user",
                "text": "It's been emotionally challenging to adapt to my condition. I feel isolated sometimes.",
                "turn": 2
            })
            seed_utterances.append({
                "speaker": "doctor",
                "text": "Your feelings are completely valid. Many people with similar conditions experience these emotions. Let's explore some coping strategies together.",
                "turn": 3
            })
        else:
            seed_utterances.append({
                "speaker": "user",
                "text": f"I need specific guidance about {support_area.value} that suits my condition.",
                "turn": 2
            })
            seed_utterances.append({
                "speaker": "doctor",
                "text": f"I'll provide you with tailored advice for {support_area.value}. Let me understand your specific needs better.",
                "turn": 3
            })
        
        return seed_utterances
    
    def create_dialogue(
        self,
        dialogue_id: str,
        user_profile: UserProfile,
        support_area: SupportArea,
        min_turns: int = 12,
        max_turns: int = 30
    ) -> Dialogue:
        """Create a complete dialogue"""
        
        # Generate seed utterances
        seed_utterances = self.generate_seed_utterances(user_profile, support_area)
        
        # Create prompt for dialogue generation
        prompt = self.create_prompt(user_profile, support_area, seed_utterances)
        
        # In practice, this would call Llama3-70B or another LLM
        # For now, we'll create a structured dialogue template
        utterances = seed_utterances.copy()
        
        # Generate additional turns (simplified version)
        num_additional_turns = random.randint(min_turns - 4, max_turns - 4)
        
        for i in range(num_additional_turns):
            turn_num = len(utterances)
            
            if turn_num % 2 == 0:  # User turn
                utterances.append({
                    "speaker": "user",
                    "text": self._generate_user_utterance(user_profile, support_area, turn_num),
                    "turn": turn_num
                })
            else:  # Doctor turn
                utterances.append({
                    "speaker": "doctor",
                    "text": self._generate_doctor_utterance(user_profile, support_area, turn_num),
                    "turn": turn_num
                })
        
        # Create dialogue object
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            user_profile=user_profile,
            support_area=support_area,
            utterances=utterances,
            metadata={
                "num_turns": len(utterances),
                "created_at": "2024-01-01"  # Placeholder
            }
        )
        
        return dialogue
    
    def _generate_user_utterance(
        self,
        user_profile: UserProfile,
        support_area: SupportArea,
        turn_num: int
    ) -> str:
        """Generate user utterance based on profile and context"""
        
        # Simplified utterance generation based on personality
        templates = {
            Persona.OPENNESS: [
                "I'm open to trying new approaches for managing my condition.",
                "Could you explain more about innovative solutions?",
                "I'd like to explore different options available."
            ],
            Persona.CONSCIENTIOUSNESS: [
                "I want to make sure I follow the correct procedures.",
                "Could you provide detailed steps I should follow?",
                "I prefer to have a structured plan."
            ],
            Persona.EXTRAVERSION: [
                "I enjoy social activities. How can I stay active despite my condition?",
                "Are there group programs I could join?",
                "I'd love to connect with others in similar situations."
            ],
            Persona.AGREEABLENESS: [
                "Thank you for your help. I appreciate your guidance.",
                "That sounds reasonable. I'm willing to try.",
                "Your suggestions are very helpful."
            ],
            Persona.NEUROTICISM: [
                "I'm worried about potential complications.",
                "What if the treatment doesn't work?",
                "I'm anxious about trying new things."
            ]
        }
        
        utterances = templates.get(user_profile.persona, ["Could you help me with this?"])
        return random.choice(utterances)
    
    def _generate_doctor_utterance(
        self,
        user_profile: UserProfile,
        support_area: SupportArea,
        turn_num: int
    ) -> str:
        """Generate doctor utterance with appropriate strategies"""
        
        # Simplified doctor response generation
        base_responses = [
            f"I understand your concerns about {user_profile.disability_type.value}.",
            f"Let me provide you with specific guidance for {support_area.value}.",
            "Your feelings are completely valid, and I'm here to support you.",
            "Here are some practical strategies that might help:",
            "Many individuals with similar conditions have found success with these approaches.",
            "I want to ensure you feel comfortable with the plan we develop together."
        ]
        
        return random.choice(base_responses)
    
    def create_dataset(
        self,
        num_dialogues: int = 6796,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, List[Dialogue]]:
        """Create the complete PDCare dataset"""
        
        dialogues = []
        
        # Generate dialogues with diverse profiles
        for i in tqdm(range(num_dialogues), desc="Creating dialogues"):
            # Random user profile
            user_profile = UserProfile(
                persona=random.choice(list(Persona)),
                gender=random.choice(list(Gender)),
                age_group=random.choice(list(AgeGroup)),
                disability_type=random.choice(list(DisabilityType))
            )
            
            # Random support area
            support_area = random.choice(list(SupportArea))
            
            # Create dialogue
            dialogue = self.create_dialogue(
                dialogue_id=f"dialogue_{i:05d}",
                user_profile=user_profile,
                support_area=support_area
            )
            
            dialogues.append(dialogue)
        
        # Split into train/val/test
        num_train = int(num_dialogues * train_ratio)
        num_val = int(num_dialogues * val_ratio)
        
        random.shuffle(dialogues)
        
        dataset = {
            "train": dialogues[:num_train],
            "validation": dialogues[num_train:num_train + num_val],
            "test": dialogues[num_train + num_val:]
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[Dialogue]], output_dir: str):
        """Save dataset to JSON files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, dialogues in dataset.items():
            output_file = os.path.join(output_dir, f"{split_name}.json")
            
            data = [d.to_dict() for d in dialogues]
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(dialogues)} dialogues to {output_file}")


if __name__ == "__main__":
    # Create dataset
    creator = PDCareDatasetCreator()
    dataset = creator.create_dataset(num_dialogues=100)  # Small sample for testing
    creator.save_dataset(dataset, "../../data/pdcare")