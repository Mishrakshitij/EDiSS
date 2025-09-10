"""
Evaluation script for EDiSS
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.ediss_model import EDiSSModel, DSSModel
from src.models.classifiers import (
    PGAClassifier,
    PolitenessStrategyClassifier,
    EmpathyStrategyClassifier
)
from src.evaluation.metrics import EDiSSEvaluator, HumanEvaluationSimulator


def load_test_data(data_path: str) -> List[Dict]:
    """Load test dataset"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_path: str,
    model_type: str = "ediss",
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Evaluate a model on test data"""
    
    print(f"Evaluating {model_type} model...")
    
    # Load model
    if model_type == "ediss":
        model = EDiSSModel(device=device)
    else:
        model = DSSModel(device=device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_model(model_path)
    
    # Load classifiers
    pga_clf = PGAClassifier()
    pol_clf = PolitenessStrategyClassifier()
    emp_clf = EmpathyStrategyClassifier()
    
    # Try to load classifier checkpoints
    model_dir = os.path.dirname(model_path)
    
    if os.path.exists(os.path.join(model_dir, "pga_classifier.pt")):
        pga_checkpoint = torch.load(os.path.join(model_dir, "pga_classifier.pt"))
        pga_clf.load_state_dict(pga_checkpoint["model_state_dict"])
        print("Loaded PGA classifier")
    
    if os.path.exists(os.path.join(model_dir, "politeness_classifier.pt")):
        pol_checkpoint = torch.load(os.path.join(model_dir, "politeness_classifier.pt"))
        pol_clf.load_state_dict(pol_checkpoint["model_state_dict"])
        print("Loaded Politeness classifier")
    
    if os.path.exists(os.path.join(model_dir, "empathy_classifier.pt")):
        emp_checkpoint = torch.load(os.path.join(model_dir, "empathy_classifier.pt"))
        emp_clf.load_state_dict(emp_checkpoint["model_state_dict"])
        print("Loaded Empathy classifier")
    
    # Load test data
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} test dialogues")
    
    # Initialize evaluator
    evaluator = EDiSSEvaluator(
        model,
        model.tokenizer,
        pga_clf,
        pol_clf,
        emp_clf,
        device=device
    )
    
    # Perform evaluation
    print(f"\nEvaluating on {min(num_samples, len(test_data))} samples...")
    metrics = evaluator.comprehensive_evaluation(test_data, num_samples)
    
    # Save results
    results = {
        "model_type": model_type,
        "model_path": model_path,
        "num_samples": min(num_samples, len(test_data)),
        "metrics": metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results for {model_type.upper()}")
    print("="*50)
    
    print("\nTask Relevance Metrics:")
    print(f"  UPC (User Profile Consistency): {metrics.get('user_profile_consistency', 0):.4f}")
    print(f"  PSA (Politeness Strategy Acc.): {metrics.get('politeness_strategy_accuracy', 0):.4f}")
    print(f"  ESA (Empathy Strategy Acc.):    {metrics.get('empathy_strategy_accuracy', 0):.4f}")
    
    print("\nLanguage Quality Metrics:")
    print(f"  Perplexity:         {metrics.get('perplexity', 0):.2f}")
    print(f"  Response Length:    {metrics.get('response_length', 0):.2f}")
    print(f"  Non-repetitiveness: {metrics.get('non_repetitiveness', 0):.4f}")
    print(f"  BLEU Score:         {metrics.get('bleu', 0):.4f}")
    print(f"  ROUGE-L Score:      {metrics.get('rouge_l', 0):.4f}")
    
    print("\nDialogue Coherence Metrics:")
    print(f"  Dialogue Coherence:  {metrics.get('dialogue_coherence', 0):.4f}")
    print(f"  Topic Consistency:   {metrics.get('topic_consistency', 0):.4f}")
    print(f"  Response Relevance:  {metrics.get('response_relevance', 0):.4f}")
    
    return metrics


def compare_models(
    results_dir: str,
    model_names: List[str]
):
    """Compare results from multiple models"""
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load all results
    all_results = {}
    for model_name in model_names:
        result_file = os.path.join(results_dir, f"{model_name}_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results[model_name] = json.load(f)
    
    if not all_results:
        print("No results found to compare")
        return
    
    # Create comparison table
    metrics_to_compare = [
        "user_profile_consistency",
        "politeness_strategy_accuracy",
        "empathy_strategy_accuracy",
        "perplexity",
        "response_length",
        "non_repetitiveness",
        "bleu",
        "rouge_l",
        "dialogue_coherence",
        "topic_consistency",
        "response_relevance"
    ]
    
    comparison_data = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for metric in metrics_to_compare:
            row[metric] = results["metrics"].get(metric, 0)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index("Model")
    
    # Print comparison table
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    print(df.to_string())
    
    # Save comparison table
    df.to_csv(os.path.join(results_dir, "model_comparison.csv"))
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_compare):
        if idx < len(axes):
            ax = axes[idx]
            values = df[metric].values
            models = df.index.tolist()
            
            bars = ax.bar(range(len(models)), values)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            
            # Color best performer
            best_idx = np.argmax(values) if metric != "perplexity" else np.argmin(values)
            bars[best_idx].set_color('green')
    
    # Remove unused subplots
    for idx in range(len(metrics_to_compare), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=150)
    print(f"\nVisualization saved to {os.path.join(results_dir, 'model_comparison.png')}")


def simulate_human_evaluation(
    model_path: str,
    test_data_path: str,
    output_path: str,
    num_interactions: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Simulate human evaluation"""
    
    print("Simulating human evaluation...")
    
    # Load model
    model = EDiSSModel(device=device)
    if os.path.exists(model_path):
        model.load_model(model_path)
    
    # Load test data
    test_data = load_test_data(test_data_path)
    
    # Initialize human evaluation simulator
    evaluator = HumanEvaluationSimulator()
    
    # Collect scores
    all_scores = []
    
    for i in tqdm(range(min(num_interactions, len(test_data))), desc="Simulating evaluations"):
        dialogue = test_data[i]
        context = ""
        user_profile = dialogue["user_profile"]
        
        # Get a sample interaction
        if len(dialogue["utterances"]) >= 2:
            user_utt = dialogue["utterances"][0]
            doctor_utt = dialogue["utterances"][1]
            
            # Generate response
            generated = model.generate_response(
                context,
                user_profile,
                user_utt["text"]
            )
            
            # Simulate evaluation
            scores = evaluator.simulate_evaluation(
                generated,
                doctor_utt["text"],
                user_profile
            )
            
            all_scores.append(scores)
    
    # Calculate average scores
    avg_scores = {}
    for criterion in evaluator.criteria:
        values = [s[criterion] for s in all_scores]
        avg_scores[criterion] = np.mean(values)
        avg_scores[f"{criterion}_std"] = np.std(values)
    
    # Save results
    results = {
        "num_interactions": len(all_scores),
        "average_scores": avg_scores,
        "all_scores": all_scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("Simulated Human Evaluation Results")
    print("="*50)
    
    for criterion in evaluator.criteria:
        score = avg_scores[criterion]
        std = avg_scores[f"{criterion}_std"]
        print(f"  {criterion:20s}: {score:.2f} ± {std:.2f}")
    
    return avg_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate EDiSS models")
    
    parser.add_argument("--mode", type=str, default="evaluate",
                       choices=["evaluate", "compare", "human"],
                       help="Evaluation mode")
    
    parser.add_argument("--model_path", type=str, default="models/ediss_final",
                       help="Path to model checkpoint")
    
    parser.add_argument("--model_type", type=str, default="ediss",
                       choices=["ediss", "dss"],
                       help="Type of model to evaluate")
    
    parser.add_argument("--test_data", type=str, default="data/pdcare/test_annotated.json",
                       help="Path to test data")
    
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    
    parser.add_argument("--compare_models", nargs="+",
                       default=["gpt2", "ardm", "llama2", "mistral", "dss", "ediss"],
                       help="Models to compare")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "evaluate":
        # Evaluate single model
        output_path = os.path.join(args.output_dir, f"{args.model_type}_results.json")
        evaluate_model(
            args.model_path,
            args.test_data,
            output_path,
            args.model_type,
            args.num_samples,
            args.device
        )
    
    elif args.mode == "compare":
        # Compare multiple models
        compare_models(args.output_dir, args.compare_models)
    
    elif args.mode == "human":
        # Simulate human evaluation
        output_path = os.path.join(args.output_dir, "human_eval_results.json")
        simulate_human_evaluation(
            args.model_path,
            args.test_data,
            output_path,
            args.num_samples,
            args.device
        )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()