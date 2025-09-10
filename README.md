# EDiSS: Empathetic Disability Support System

An AI-powered conversational system designed to provide personalized, polite, and empathetic support to individuals with physical disabilities. EDiSS leverages reinforcement learning with custom reward functions to generate contextually appropriate responses tailored to users' personality traits, gender, and age.

## Dataset Access

**The PDCare dataset is available for research purposes.** To request access:
1. Visit the `Dataset/` folder
2. Fill out the access request form
3. Your request will be reviewed within 2-3 business days
4. Upon approval, the dataset will be sent to your email

## Overview

EDiSS (Empathetic Disability Support System) is based on the paper "Breaking Barriers: A Paradigm Shift in Technology Accessibility for Individuals with Physical Disabilities". The system:

- Provides personalized support across 13 types of physical disabilities
- Covers 17 different support areas from physical therapy to emotional support
- Adapts responses based on user's OCEAN personality traits, gender, and age
- Employs 3 politeness strategies and 8 empathy strategies
- Uses Phi-3-small model with LoRA fine-tuning and PPO reinforcement learning

## Key Features

- **Personalized Responses**: Tailors support based on user profiles (personality, gender, age)
- **Multi-Strategy Approach**: Implements politeness and empathy strategies for better user experience
- **Comprehensive Coverage**: Addresses various disabilities and support areas
- **Advanced Training**: Uses PPO with custom reward functions for optimal response generation
- **Robust Evaluation**: Includes automatic and human evaluation metrics

## Project Structure

```
EDiSS/
├── src/
│   ├── dataset/
│   │   ├── pdcare_dataset.py      # Dataset creation
│   │   └── annotation.py          # Strategy annotation
│   ├── models/
│   │   ├── ediss_model.py        # Main EDiSS model
│   │   └── classifiers.py        # PGA, politeness, empathy classifiers
│   ├── training/
│   │   ├── rewards.py            # Reward functions
│   │   └── ppo_trainer.py        # PPO training implementation
│   └── evaluation/
│       └── metrics.py             # Evaluation metrics
├── configs/
│   └── training_config.yaml      # Configuration file
├── train.py                       # Main training script
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended
- 40GB+ free disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EDiSS.git
cd EDiSS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Quick Start

### 1. Create the PDCare Dataset

```bash
python train.py --mode create_dataset --num_dialogues 100
```

For the full dataset (6,796 dialogues):
```bash
python train.py --mode create_dataset --num_dialogues 6796
```

### 2. Annotate the Dataset

```bash
python train.py --mode annotate
```

### 3. Train Classifiers

```bash
python train.py --mode classifiers --classifier_epochs 10
```

### 4. Train DSS (Warm-start)

```bash
python train.py --mode dss --dss_epochs 5 --batch_size 4
```

### 5. Train EDiSS with PPO

```bash
python train.py --mode ppo --ppo_train_epochs 3 --ppo_batch_size 2
```

### 6. Complete Training Pipeline

To run the entire training pipeline:
```bash
python train.py --mode all
```

## Evaluation

### Evaluate a Single Model

```bash
python evaluate.py --mode evaluate --model_path models/ediss_final --num_samples 100
```

### Compare Multiple Models

```bash
python evaluate.py --mode compare --compare_models dss ediss
```

### Simulate Human Evaluation

```bash
python evaluate.py --mode human --model_path models/ediss_final --num_samples 50
```

## Usage Example

```python
from src.models.ediss_model import EDiSSModel

# Initialize model
model = EDiSSModel(device="cuda")
model.load_model("models/ediss_final")

# Define user profile
user_profile = {
    "persona": "extraversion",
    "gender": "female", 
    "age_group": "middle_aged",
    "disability_type": "spinal_cord_injury"
}

# Generate response
context = "User has been discussing physical therapy needs."
user_utterance = "I'm finding it difficult to do the exercises on my own."

response = model.generate_response(
    context,
    user_profile,
    user_utterance
)

print(f"EDiSS: {response}")
```

## Training Configuration

Edit `configs/training_config.yaml` to customize:

- Dataset parameters
- Model hyperparameters
- Training settings
- Reward weights
- Hardware configuration

## Model Architecture

### Base Model
- **Phi-3-small-8k-instruct**: Lightweight but powerful language model
- **LoRA Fine-tuning**: Parameter-efficient adaptation

### Classifiers
- **PGA Classifier**: 30-class classifier for persona-gender-age combinations
- **Politeness Classifier**: 3-class classifier for politeness strategies
- **Empathy Classifier**: 8-class classifier for empathy strategies

### Training Process
1. **DSS Training**: Warm-start with supervised fine-tuning
2. **PPO Training**: Reinforcement learning with custom rewards
   - Task-relevance rewards (user profile alignment, strategy accuracy)
   - Smoothness rewards (syntactic and semantic coherence)

## Evaluation Metrics

### Automatic Metrics
- **UPC**: User Profile Consistency
- **PSA**: Politeness Strategy Accuracy
- **ESA**: Empathy Strategy Accuracy
- **PPL**: Perplexity
- **Response Length Ratio**
- **Non-repetitiveness**
- **BLEU & ROUGE scores**

### Human Evaluation Metrics
- Persona accuracy
- Gender-age accuracy
- Politeness accuracy
- Empathy accuracy
- Fluency
- Consistency
- Non-repetitiveness

## Performance

Based on the paper's results:

| Metric | EDiSS | DSS | Improvement |
|--------|-------|-----|-------------|
| UPC | 60.7% | 57.9% | +2.8% |
| PSA | 81.5% | 79.1% | +2.4% |
| ESA | 74.9% | 71.5% | +3.4% |
| Perplexity | 3.60 | 4.85 | -25.8% |

## Supported Disabilities

- Mobility Impairments
- Visual Impairments
- Hearing Impairments
- Speech Impairments
- Neurological Disorders
- Spinal Cord Injuries
- Amputations
- Orthopedic Disabilities
- Cerebral Palsy
- Muscular Dystrophy
- Balance and Gait Disorders
- Chronic Pain
- Aging-Related Disabilities

## Support Areas

- Accessibility Information
- Travel Tips
- Advocacy and Rights
- Financial and Insurance Guidance
- Mobility Aids
- Home Modifications
- Physical Therapy Exercises
- Assistive Technology
- Pain Management
- Activities of Daily Living
- Emotional Support
- Employment and Education
- Social Interaction
- Fitness and Recreation
- Peer Support Groups
- Parenting with Disabilities
- Transitions and Life Changes

## Troubleshooting

### Out of Memory Error
- Reduce batch size in training
- Use gradient accumulation
- Enable fp16 training
- Use CPU if GPU memory insufficient

### Slow Training
- Ensure CUDA is properly installed
- Check GPU utilization
- Reduce model max_length
- Use smaller LoRA rank

### Poor Performance
- Increase training epochs
- Adjust reward weights
- Ensure quality of annotated data
- Fine-tune hyperparameters

## Citation

If you use EDiSS in your research, please cite:

```bibtex
@inproceedings{mishra2025breaking,
  title={Breaking Barriers: A Paradigm Shift in Technology Accessibility for Individuals with Physical Disabilities},
  author={Mishra, Kshitij and Burja, Manisha and Ekbal, Asif},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
```


## Acknowledgments
- The disability support community for inspiration
- Open-source contributors

## Contact

For questions or issues:
- Email: mishra.kshitij07@gmail.com
- GitHub Issues: [Create an issue](https://github.com/Mishrakshitij/EDiSS/issues)

