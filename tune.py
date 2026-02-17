import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

def specificity(gts, preds):
    cm = confusion_matrix(gts, preds)
    return cm[0, 0]/cm[0].sum()

def find_balanced_threshold(y_true, y_prob, spe_weight=1.0, sen_weight=1.0):
    """
    Find threshold that gives weighted balanced sensitivity and specificity
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        spe_weight: Weight for specificity (default: 1.0)
        sen_weight: Weight for sensitivity (default: 1.0)
        
    Returns:
        best_threshold: Threshold that minimizes weighted difference between sensitivity and specificity
    """
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0.5
    min_diff = float('inf')
    best_metrics = None
    
    # Normalize weights to sum to 1
    total_weight = spe_weight + sen_weight
    spe_weight = spe_weight / total_weight
    sen_weight = sen_weight / total_weight
    
    print(f"\nFinding balanced threshold with weights - Specificity: {spe_weight:.2f}, Sensitivity: {sen_weight:.2f}")
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate weighted difference
        weighted_sensitivity = sensitivity * sen_weight
        weighted_specificity = specificity * spe_weight
        diff = abs(weighted_specificity - weighted_sensitivity)
        
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'weighted_sensitivity': weighted_sensitivity,
                'weighted_specificity': weighted_specificity
            }
    
    # Print detailed information about the selected threshold
    print(f"Selected threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Raw Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Raw Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"  Weighted Specificity: {best_metrics['weighted_specificity']:.4f}")
    print(f"  Weighted Sensitivity: {best_metrics['weighted_sensitivity']:.4f}")
            
    return best_threshold

def find_best_threshold_pr_curve(probs, truth):
    """Find best threshold using PR curve"""
    precision, recall, thresholds = precision_recall_curve(truth, probs)
    epsilon = 1e-7
    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
    best_index = np.nanargmax(f1_scores)
    if best_index >= len(thresholds):
        best_index = len(thresholds) - 1
    best_threshold = thresholds[best_index]
    return best_threshold

def calculate_metrics(pred_probs, true_labels, threshold):
    """Calculate all metrics using a given threshold"""
    predictions = (pred_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'auc': roc_auc_score(true_labels, pred_probs),
        'acc': accuracy_score(true_labels, predictions),
        'spe': spec,
        'sen': sensitivity,
        'f1': f1_score(true_labels, predictions),
        'ppv': ppv,
        'npv': npv,
        'threshold': threshold
    }

def process_fold(val_file, test_file, fold, spe_weight=2.0, sen_weight=1.0):
    """Process a single fold using both threshold methods with weighted balance"""
    print(f"\nProcessing fold {fold}")
    
    try:
        # Read validation and test data
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        print(f"Validation set size: {len(val_df)}, Test set size: {len(test_df)}")
        
        # Method 1: Weighted balanced threshold
        balanced_threshold = find_balanced_threshold(
            y_true=val_df['true_label'],
            y_prob=val_df['prob_class_1'],
            spe_weight=spe_weight,
            sen_weight=sen_weight
        )
        print(f"Weighted balanced threshold for fold {fold}: {balanced_threshold:.4f}")
        
        balanced_metrics = calculate_metrics(
            pred_probs=test_df['prob_class_1'],
            true_labels=test_df['true_label'],
            threshold=balanced_threshold
        )
        balanced_metrics['fold'] = fold
        
        # Method 2: PR curve threshold
        pr_threshold = find_best_threshold_pr_curve(
            probs=val_df['prob_class_1'],
            truth=val_df['true_label']
        )
        print(f"PR curve threshold for fold {fold}: {pr_threshold:.4f}")
        
        pr_metrics = calculate_metrics(
            pred_probs=test_df['prob_class_1'],
            true_labels=test_df['true_label'],
            threshold=pr_threshold
        )
        pr_metrics['fold'] = fold
        
        return balanced_metrics, pr_metrics
        
    except Exception as e:
        print(f"Error processing fold {fold}: {str(e)}")
        return None, None

def process_results(results, method_name, args):
    """Process results for one method and save/print them"""
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean and std for numeric columns
    numeric_cols = ['fold', 'threshold', 'auc', 'acc', 'spe', 'sen', 'f1', 'ppv', 'npv']
    mean_results = results_df[numeric_cols].mean().to_frame().T
    std_results = results_df[numeric_cols].std().to_frame().T
    
    mean_results.index = ['mean']
    std_results.index = ['std']
    
    # Combine results
    final_df = pd.concat([results_df, mean_results, std_results])
    
    # Save results
    output_file = os.path.join(args.input_dir, f'tune_results_{method_name}.csv')
    final_df.to_csv(output_file, index=True)
    print(f"\nResults for {method_name} method saved to: {output_file}")
    
    # Print results
    print_results(final_df, method_name)
    
    return final_df


def print_results(results_df, method_name):
    print(f"\nDetailed Results ({method_name}):")
    headers = ['fold', 'threshold', 'auc', 'acc', 'spe', 'sen', 'f1', 'ppv', 'npv']
    format_str = "{:4}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}  {:8.4f}"
    
    print("fold  threshold    auc       acc       spe       sen        f1        ppv       npv   ")
    print("-" * 95)
    
    for idx, row in results_df.iterrows():
        if isinstance(idx, (int, np.integer)):
            values = [idx] + [row[col] for col in headers[1:]]
            print(format_str.format(*values))
    
    print("-" * 95)
    mean_row = results_df.loc['mean']
    std_row = results_df.loc['std']
    
    values_mean = ['mean'] + [mean_row[col] for col in headers[1:]]
    values_std = ['std'] + [std_row[col] for col in headers[1:]]
    
    print(format_str.format(*values_mean))
    print(format_str.format(*values_std))
def main():
    parser = argparse.ArgumentParser(description='Tune model using two threshold methods')
    parser.add_argument('--input_dir', type=str, required=True, 
                      help='Directory containing the fold_*.csv files')
    parser.add_argument('--num_folds', type=int, default=3,
                      help='Number of folds to process (default: 3)')
    parser.add_argument('--spe_weight', type=float, default=1,
                      help='Weight for specificity in balanced threshold (default: 1.0)')
    parser.add_argument('--sen_weight', type=float, default=1.2,
                      help='Weight for sensitivity in balanced threshold (default: 1.0)')
    
    args = parser.parse_args()
    os.makedirs(args.input_dir, exist_ok=True)

    print(f"\nUsing weights - Specificity: {args.spe_weight}, Sensitivity: {args.sen_weight}")
    balanced_results = []
    pr_curve_results = []

    for i in range(args.num_folds):
        val_file = os.path.join(args.input_dir, f'fold_{i}_val.csv')
        test_file = os.path.join(args.input_dir, f'fold_{i}_test.csv')
        
        if os.path.exists(val_file) and os.path.exists(test_file):
            balanced_result, pr_result = process_fold(
                val_file, test_file, i,
                spe_weight=args.spe_weight,
                sen_weight=args.sen_weight
            )
            if balanced_result is not None and pr_result is not None:
                balanced_results.append(balanced_result)
                pr_curve_results.append(pr_result)
        else:
            print(f"Missing files for fold {i}")

    if not balanced_results or not pr_curve_results:
        raise ValueError("No results were calculated successfully")

    # Process results for both methods
    suffix = f"spe{args.spe_weight}_sen{args.sen_weight}"
    print(f"\nProcessing results for weighted balanced threshold method ({suffix}):")
    balanced_df = process_results(balanced_results, f'balanced_{suffix}', args)
    
    print("\nProcessing results for PR curve threshold method:")
    pr_curve_df = process_results(pr_curve_results, 'pr_curve', args)

if __name__ == '__main__':
    main()