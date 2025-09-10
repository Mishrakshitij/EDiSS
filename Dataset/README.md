# PDCare Dataset Access

The PDCare dataset is a comprehensive collection of 6,796 empathetic support dialogues designed for individuals with physical disabilities. This dataset is a key component of the EDiSS (Empathetic Disability Support System) project.

## Dataset Overview

- **Size**: 6,796 dialogues
- **Utterances**: 156,094 total utterances
- **Average Length**: ~23 utterances per dialogue
- **Languages**: English

### Coverage

**Physical Disabilities (13 types):**
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

**Support Areas (17 categories):**
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

### Annotations

Each dialogue includes:
- **User Profile**: OCEAN personality traits, gender, age group
- **Politeness Strategies**: Positive, Negative, Bald-on-record
- **Empathy Strategies**: 8 different empathetic response strategies
- **Quality Metrics**: Topic consistency, context adequacy, fluency scores

## Requesting Access

To request access to the PDCare dataset:

1. **Visit the Request Form**: Open `index.html` in your browser or host it on your website
2. **Complete All Required Fields**: Provide your research details and institutional affiliation
3. **Upload Verification**: Submit your institutional ID or verification document
4. **Agree to Terms**: Accept the usage terms and citation requirements
5. **Submit Request**: Your request will be reviewed within 2-3 business days

### Access Requirements

- **Academic/Research Purpose**: The dataset is primarily for research use
- **Institutional Affiliation**: Must be affiliated with a recognized institution
- **Citation Agreement**: Must cite the original paper in any publications
- **Ethical Use**: Commitment to responsible and ethical use of the data

## Data Format

The dataset is provided in JSON format with the following structure:

```json
{
  "dialogue_id": "dialogue_00001",
  "user_profile": {
    "persona": "extraversion",
    "gender": "female",
    "age_group": "middle_aged",
    "disability_type": "spinal_cord_injury"
  },
  "support_area": "physical_therapy",
  "utterances": [
    {
      "speaker": "user",
      "text": "I need help with exercises...",
      "turn": 0
    },
    {
      "speaker": "doctor",
      "text": "I understand your needs...",
      "turn": 1,
      "politeness_strategy": "positive_politeness",
      "empathy_strategy": "practical_assistance"
    }
  ]
}
```

## Dataset Splits

- **Training Set**: 5,436 dialogues (80%)
- **Validation Set**: 681 dialogues (10%)
- **Test Set**: 679 dialogues (10%)

## Usage Guidelines

### Permitted Uses
- Academic research
- Training dialogue systems
- Studying empathetic communication
- Accessibility technology development
- Healthcare communication research

### Restrictions
- No commercial use without permission
- No redistribution without authorization
- Must maintain data privacy and confidentiality
- Cannot use for harmful or discriminatory purposes

## Citation

If you use the PDCare dataset in your research, please cite:

```bibtex
@inproceedings{mishra2025breaking,
  title={Breaking Barriers: A Paradigm Shift in Technology Accessibility for Individuals with Physical Disabilities},
  author={Mishra, Kshitij and Burja, Manisha and Ekbal, Asif},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Privacy and Ethics

- All dialogues are synthetically generated to protect privacy
- No real patient data or personally identifiable information
- Created with ethical considerations for disability representation
- Reviewed by domain experts for appropriateness

## Support

For questions about the dataset or access issues:
- **Email**: mishra.kshitij07@gmail.com
- **GitHub Issues**: https://github.com/Mishrakshitij/EDiSS/issues

## Acknowledgments

We thank the disability support community for their invaluable insights and the reviewers who helped ensure the quality and appropriateness of the dataset content.

---

**Note**: The dataset access form requires JavaScript to function properly. Please ensure JavaScript is enabled in your browser when submitting the request.