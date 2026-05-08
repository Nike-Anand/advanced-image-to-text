"""
Evaluation Script for Tamil Handwritten Character Recognition
Evaluates trained model on test set and generates metrics
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tamil_char_dataset import TamilCharDataset, get_default_transforms
from char_mapping import get_charset_string, get_char_mapping, char_to_class_id
import config_handwritten as config
from ocr_tamil.strhub.models.parseq.system import PARSeq


def load_model(checkpoint_path, charset):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    model = PARSeq.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set.
    
    Returns:
        predictions: List of predicted characters
        ground_truths: List of ground truth characters
        confidences: List of prediction confidences
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    ground_truths = []
    confidences = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            # Forward pass
            logits = model(images)
            probs = logits.softmax(-1)
            
            # Decode predictions
            preds, probs_decoded = model.tokenizer.decode(probs)
            
            # Store results
            for pred, prob, gt in zip(preds, probs_decoded, labels):
                pred_char = model.charset_adapter(pred)
                predictions.append(pred_char)
                ground_truths.append(gt)
                confidences.append(prob.prod().item())
    
    return predictions, ground_truths, confidences


def calculate_metrics(predictions, ground_truths):
    """Calculate accuracy metrics."""
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy


def plot_confusion_matrix(predictions, ground_truths, save_path=None):
    """Generate and plot confusion matrix."""
    char_mapping = get_char_mapping()
    unique_chars = sorted(set(ground_truths + predictions))
    
    # Limit to top N most common characters for readability
    MAX_CHARS = 50
    if len(unique_chars) > MAX_CHARS:
        from collections import Counter
        char_counts = Counter(ground_truths)
        unique_chars = [char for char, _ in char_counts.most_common(MAX_CHARS)]
    
    # Create confusion matrix
    cm = confusion_matrix(ground_truths, predictions, labels=unique_chars)
    
    # Plot
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_chars, yticklabels=unique_chars,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Tamil Character Recognition', fontsize=16, pad=20)
    plt.xlabel('Predicted Character', fontsize=12)
    plt.ylabel('True Character', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_errors(predictions, ground_truths, confidences, top_n=20):
    """Analyze common error patterns."""
    errors = defaultdict(int)
    low_confidence_errors = []
    
    for pred, gt, conf in zip(predictions, ground_truths, confidences):
        if pred != gt:
            errors[(gt, pred)] += 1
            low_confidence_errors.append((gt, pred, conf))
    
    # Sort by frequency
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"TOP {top_n} MOST COMMON ERRORS")
    print("=" * 80)
    print(f"{'True':<10} {'Predicted':<10} {'Count':<10}")
    print("-" * 40)
    
    for (true_char, pred_char), count in sorted_errors[:top_n]:
        print(f"{true_char:<10} {pred_char:<10} {count:<10}")
    
    # Low confidence errors
    low_confidence_errors.sort(key=lambda x: x[2])
    
    print("\n" + "=" * 80)
    print(f"LOWEST CONFIDENCE ERRORS (Top {min(10, len(low_confidence_errors))})")
    print("=" * 80)
    print(f"{'True':<10} {'Predicted':<10} {'Confidence':<15}")
    print("-" * 40)
    
    for true_char, pred_char, conf in low_confidence_errors[:10]:
        print(f"{true_char:<10} {pred_char:<10} {conf:<15.4f}")


def per_class_accuracy(predictions, ground_truths):
    """Calculate per-class accuracy."""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, gt in zip(predictions, ground_truths):
        class_total[gt] += 1
        if pred == gt:
            class_correct[gt] += 1
    
    class_accuracies = {
        char: (class_correct[char] / class_total[char] * 100) 
        for char in class_total
    }
    
    # Sort by accuracy
    sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])
    
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY")
    print("=" * 80)
    
    print("\nWorst performing classes (Bottom 10):")
    print(f"{'Character':<15} {'Accuracy':<15} {'Total Samples':<15}")
    print("-" * 50)
    for char, acc in sorted_accuracies[:10]:
        print(f"{char:<15} {acc:<15.2f}% {class_total[char]:<15}")
    
    print("\nBest performing classes (Top 10):")
    print(f"{'Character':<15} {'Accuracy':<15} {'Total Samples':<15}")
    print("-" * 50)
    for char, acc in sorted_accuracies[-10:]:
        print(f"{char:<15} {acc:<15.2f}% {class_total[char]:<15}")
    
    return class_accuracies


def save_predictions(predictions, ground_truths, confidences, save_path):
    """Save predictions to file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Ground Truth,Prediction,Confidence,Correct\n")
        for gt, pred, conf in zip(ground_truths, predictions, confidences):
            correct = "✓" if gt == pred else "✗"
            f.write(f"{gt},{pred},{conf:.4f},{correct}\n")
    
    print(f"\nPredictions saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tamil Handwritten Character Recognition')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=config.TEST_DIR, help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--subset_size', type=int, default=None, help='Use subset of test data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get charset
    charset = get_charset_string()
    
    # Load model
    model = load_model(args.checkpoint, charset)
    
    # Create test dataset
    print(f"\nLoading test data from: {args.test_dir}")
    test_dataset = TamilCharDataset(
        args.test_dir,
        transform=get_default_transforms(config.IMG_SIZE, augment=False),
        subset_size=args.subset_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    predictions, ground_truths, confidences = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = calculate_metrics(predictions, ground_truths)
    
    # Per-class accuracy
    class_accuracies = per_class_accuracy(predictions, ground_truths)
    
    # Analyze errors
    analyze_errors(predictions, ground_truths, confidences)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(predictions, ground_truths, save_path=cm_path)
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, 'predictions.csv')
    save_predictions(predictions, ground_truths, confidences, pred_path)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Test data: {args.test_dir}\n")
        f.write(f"Total samples: {len(predictions)}\n")
        f.write(f"Overall accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"\nAverage confidence: {np.mean(confidences):.4f}\n")
        f.write(f"Median confidence: {np.median(confidences):.4f}\n")
    
    print(f"\nEvaluation summary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
