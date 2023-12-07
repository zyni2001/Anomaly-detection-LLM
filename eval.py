import os
import re
import json
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
import argparse


def get_confidence(input_str, review_id):
    # Only the part of the string after "confidence for outlier"
    start_index = input_str.find("confidence for outlier:")
    relevant_part = input_str[start_index:]

    pattern = f"review{review_id}"
    regex = r"\{" + pattern + r"\}\((.*?)\)"

    # "Confidence for outlier part", note index is 0 not 1.
    line = relevant_part.split("\n")[0]

    match = re.search(regex, line)
    if match:
        confidence = match.group(1)
        return confidence


def count_data(file_path, medium_type):
    """_summary_

    Args:
        file_path (str): json file path
        model_name (str): gpt-3.5-turbo or llamas
        medium_type (int): True if MEDIUM as an outlier.

    Returns:
        _type_: _description_
    """
    detected_outliers_count = 0
    total_reviews_count = 0
    total_outliers_count = 0
    correct_outlier_predictions = 0 # True Positives
    ground_truths = [] # 1 if outlier, 0 if not outlier
    ground_truth_probs = []
    
    # Outlier Types
    outlier_types = ["HIGH"] + ["MEDIUM"] if medium_type else ["HIGH"]
    outlier_confidence_scores_type = [0.9, 0.6, 0.3]

    with open(file_path, "r") as file:
        data = json.load(file)

        for entry in data:
            # print(entry["ID"])
            # Check if the entry is a valid JSON object with the required fields
            if all(key in entry for key in ["output", "ground_truth"]) and isinstance(
                entry["output"], str
            ):
                output = entry["output"]
                ground_truth = entry["ground_truth"]
                total_reviews_count += len(ground_truth)
                total_outliers_count += sum(1 for gt in ground_truth if -1 in gt.values())

                # 1 if outlier, 0 if not outlier (outliers as positives)
                ground_truths += [0 if list(gt.values())[0] == 1 else 1 for gt in ground_truth]

                for i in range(1, len(ground_truth)+1):
                    # confidence = get_confidence(output, i)
                    # if confidence == "HIGH":
                    #     outlier_confidence_score = outlier_confidence_scores_type[0]
                    # elif confidence == "MEDIUM":
                    #     outlier_confidence_score = outlier_confidence_scores_type[1]
                    # else:
                    #     outlier_confidence_score = outlier_confidence_scores_type[2]
                    # ground_truth_probs.append(outlier_confidence_score)

                    # # Detected as outlier
                    # if f"review{i}" in output.split("\n")[0] and confidence in outlier_types:
                    #     detected_outliers_count += 1

                    #     # If the prediction is correct, add ONE to the correct_outlier_predictions counter
                    #     if list(ground_truth[i - 1].values())[0] == -1:
                    #         correct_outlier_predictions += 1

                    # Test: All detected as ground truth
                    if list(ground_truth[i - 1].values())[0] == -1:
                        detected_outliers_count += 1
                        correct_outlier_predictions += 1
                        ground_truth_probs.append(1)
                    else:
                        ground_truth_probs.append(0)


    return (
        total_reviews_count,
        total_outliers_count,
        detected_outliers_count,
        correct_outlier_predictions,
        ground_truths,
        ground_truth_probs
    )


def main(args):
    """Do evaluation on gpt-3.5-turbo and llama 70B. And draw the P-R Curve, ROC Curve and histogram of F1, AP, and AUC scores.
    """

    result_text = ""

    model_name = args.model

    file_paths = [f'output_{model_name}.json', f'output_{model_name}2.json', f'output_{model_name}3.json']
    llm_output_path = './llm_output'
    file_paths = [os.path.join(llm_output_path, file_path) for file_path in file_paths]

    total_reviews_count = 0
    total_outliers_count = 0
    detected_outliers_count = 0
    correct_outlier_predictions = 0
    ground_truths = []
    ground_truth_probs = []

    for file_path in file_paths:
        total_reviews, total_outliers, detected_outliers, correct_predictions, truths, probs = count_data(file_path, args.medium_type)
        total_reviews_count += total_reviews
        total_outliers_count += total_outliers
        detected_outliers_count += detected_outliers
        correct_outlier_predictions += correct_predictions
        ground_truths += truths
        ground_truth_probs += probs

    true_positives = correct_outlier_predictions
    false_positives = detected_outliers_count - correct_outlier_predictions
    false_negatives = total_outliers_count - correct_outlier_predictions
    true_negatives = total_reviews_count - total_outliers_count - false_positives

    result_text += f'TP: {true_positives}\n'
    result_text += f'FP: {false_positives}\n'
    result_text += f'TN: {true_negatives}\n'
    result_text += f'FN: {false_negatives}\n\n'

    # Precision-Recall curve & AP score
    precisions, recalls, pr_thresholds = precision_recall_curve(ground_truths, ground_truth_probs)
    for i, value in enumerate(pr_thresholds):
        result_text += f"pr_thresholds: {value:<10.4f}  precisions: {precisions[i]:<10.4f}  recalls: {recalls[i]:<10.4f}\n"
    ap_score = average_precision_score(ground_truths, ground_truth_probs)
    result_text += f"average_precision_score: {ap_score:.4f}\n\n"

    # ROC & AUC
    fpr, tpr, roc_thresholds = roc_curve(ground_truths, ground_truth_probs)
    for i, value in enumerate(roc_thresholds):
        result_text += f"roc_thresholds: {value:<10.4f}  fpr: {fpr[i]:<10.4f}  tpr: {tpr[i]:<10.4f}\n"
    # print(f"auc: {auc(fpr, tpr)}") # same number as roc_auc_score
    auc_score = roc_auc_score(ground_truths, ground_truth_probs)
    result_text += f"roc_auc_score: {auc_score:.4f}\n\n"

    # Output results
    output_path = './output'
    with open(os.path.join(output_path, f'{model_name}_medium_{int(args.medium_type)}.txt'), 'w') as f:
        f.write(result_text)
    print(result_text)

    # plotting
    plt.figure(figsize=(12, 5))

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, color='blue', label=f'P-R Curve (AP = {ap_score:.4f})')
    plt.scatter(recalls, precisions, c=pr_thresholds)
    # plt.colorbar(label='P-R Threshold')
    # annotate thresholds text above the points
    for i, value in enumerate(pr_thresholds):
        if value <= 1:
            plt.annotate(str(value), (recalls[i], precisions[i]), textcoords="offset points", xytext=(0,3), ha='center')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")

    # ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='red', label=f'ROC (AUC = {auc_score:.4f})')
    plt.scatter(fpr, tpr, c=roc_thresholds)
    # plt.colorbar(label='Threshold')

    # annotate thresholds text above the points
    for i, value in enumerate(roc_thresholds):
        if value <= 1:
            plt.annotate(str(value), (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,3), ha='center')

    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(True)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")

    # plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_path, f'{model_name}_medium_{int(args.medium_type)}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMs for Anomaly Detection Evaluation')
    parser.add_argument('--model', default='gpt', type=str, help='Model name') # 'gpt', 'llama'
    parser.add_argument('--medium_type', default=False, type=bool, help='Evaluation metric') # True if MEDIUM as an outlier

    args = parser.parse_args()

    main(args)