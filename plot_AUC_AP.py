# %% [markdown]
# # ChatGpt

# %%
# GPT-3.5-Turbo
import json
import re


def get_confidence(input_str, review_id):
    # Only the part of the string after "confidence for outlier"
    start_index = input_str.find("confidence for outlier:")
    relevant_part = input_str[start_index:]

    pattern = f"review{review_id}"
    regex = r"\{" + pattern + r"\}\((.*?)\)"

    # "Confidence for outlier part"
    line = relevant_part.split("\n")[0]

    match = re.search(regex, line)
    if match:
        confidence = match.group(1)
        return confidence



def count_data(file_path):
    detected_outlier_review_count = 0
    all_review_count = 0
    all_outlier_count = 0
    correct_outlier_predictions = 0
    ground_truths = []
    ground_truth_probs = []

    with open(file_path, "r") as file:
        data = json.load(file)

        for entry in data:
            print(entry["ID"])
            # Check if the entry is a valid JSON object with the required fields
            if all(key in entry for key in ["output", "ground_truth"]) and isinstance(
                entry["output"], str
            ):
                output = entry["output"]
                ground_truth = entry["ground_truth"]
                all_review_count += len(ground_truth)
                all_outlier_count += sum(1 for gt in ground_truth if -1 in gt.values())
                
                ground_truths += [1 if list(gt.values())[0] == 1 else 0 for gt in ground_truth]

                for i in range(1, len(ground_truth)+1):
                    # Detected Outlier
                    # Different from llama & gpt
                    confidence = get_confidence(output, i)
                    if f"review{i}" in output.split("\n")[0] and confidence in ["HIGH", "MEDIUM"]:
                        detected_outlier_review_count += 1

                        if confidence == "HIGH":
                            # confidence_score = 1
                            confidence_score = 0.9
                        elif confidence == "MEDIUM":
                            # confidence_score = 0.7
                            confidence_score = 0.6
                        
                        # If the prediction is correct, add ONE to the correct_outlier_predictions counter
                        if list(ground_truth[i - 1].values())[0] == -1:
                            correct_outlier_predictions += 1
                        
                        ground_truth_probs.append(confidence_score)
                        # else:
                            # incorrect_predictions += 1
                            # ground_truth_probs.append(1 - confidence_score)
                    # Detected Real
                    else:
                        # If the prediction is non-outlier, add the confidence score 0.3 to the ground_truth_probs list
                        ground_truth_probs.append(0.3)

                        # if list(ground_truth[i - 1].values())[0] == 1:
                        #     # ground_truth_probs.append(0.7)
                        #     ground_truth_probs.append(0.6)
                        # else:
                        #     ground_truth_probs.append(0.3)


    return (
        detected_outlier_review_count,
        all_outlier_count,
        all_review_count,
        correct_outlier_predictions,
        ground_truths,
        ground_truth_probs
        # incorrect_predictions,
    )

detected_outlier_review_count1, all_outlier_count1, all_review_count1, correct_outlier_predictions1, ground_truths1, ground_truth_probs1 = count_data('./output_gpt.json')
detected_outlier_review_count2, all_outlier_count2, all_review_count2, correct_outlier_predictions2, ground_truths2, ground_truth_probs2 = count_data('./output_gpt2.json')
detected_outlier_review_count3, all_outlier_count3, all_review_count3, correct_outlier_predictions3, ground_truths3, ground_truth_probs3 = count_data('./output_gpt3.json')


print()
print()
print()
print("----------------")
print(detected_outlier_review_count1, all_outlier_count1, all_review_count1, correct_outlier_predictions1)
print(detected_outlier_review_count2, all_outlier_count2, all_review_count2, correct_outlier_predictions2)
print(detected_outlier_review_count3, all_outlier_count3, all_review_count3, correct_outlier_predictions3)


# %%
ground_truths = ground_truths1 + ground_truths2 + ground_truths3
ground_truth_probs = ground_truth_probs1 + ground_truth_probs2 + ground_truth_probs3

from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(ground_truths, ground_truth_probs)
ap = average_precision_score(ground_truths, ground_truth_probs)
auc, ap


# %%
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Assuming ground_truths and ground_truth_probs are already defined
# If not, they need to be provided for the calculations

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(ground_truths, ground_truth_probs)
ap = average_precision_score(ground_truths, ground_truth_probs)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(ground_truths, ground_truth_probs)
roc_auc = auc(fpr, tpr)

# Plotting both curves
plt.figure(figsize=(12, 5))

# Precision-Recall curve
plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='blue')
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
# Adjust the position of the text
# Check the axes limits and set the text position accordingly
# For example, if recall ranges from 0 to 1, and precision ranges from 0.7 to 1,
# you might want to place the text at (0.1, 0.72)
plt.text(0.6, 0.85, f'AP = {ap:.2f}', fontsize=12)

# ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='red')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.text(0.6, 0.2, f'AUC = {roc_auc:.2f}', fontsize=12)

plt.tight_layout()
# plt.show()
plt.savefig('gpt.png')


# %%
# ground_truth_probs

# %%
        # detected_outlier_review_count,
        # all_outlier_count,
        # all_review_count,
        # correct_outlier_predictions,


# 3118 + 1791 + 247, 1473 + 1040 + 158, 10665 + 6760 + 957, 418 + 288 + 58


# # %% [markdown]
# # GPT:
# # - F1: 0.1950
# # - AP: 0.9250
# # - AUC: 0.6296
# # 
# # llama:
# # - F1: 0.1113
# # - AP: 0.1131
# # - AUC: 0.4897

# # %%
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# new_data = {
#     "Model": ["CARE-GNN", "CARE-GNN", "CARE-GNN", 
#               "PC-GNN", "PC-GNN", "PC-GNN",
#               "GTAN", "GTAN", "GTAN", 
#               "Llama 70B", "Llama 70B", "Llama 70B",
#               "GPT-3.5-Turbo", "GPT-3.5-Turbo", "GPT-3.5-Turbo"],
#     "Metric": ["F1", "AP", "AUC", "F1", "AP", "AUC", "F1", "AP", "AUC", "F1", "AP", "AUC", "F1", "AP", "AUC"],
#     "Value": [0.5971, 0.3285, 0.7420, 0.4895, 0.3238, 0.8101, 0.8280, 0.7930, 0.9348, 0.1113, 0.1131, 0.4897, 0.1950, 0.9250, 0.6296]
# }

# new_df = pd.DataFrame(new_data)

# # 移除N/A的数据
# new_df = new_df.dropna()

# # 暖色调
# warm_palette = sns.color_palette("YlOrRd", n_colors=3)

# # 设置图表样式
# sns.set_context("poster")  # 更大的字体尺寸
# sns.set(style="darkgrid", rc={"axes.facecolor": "0.9"})

# # 重新创建柱状图
# plt.figure(figsize=(16, 10))
# barplot = sns.barplot(x="Model", y="Value", hue="Metric", data=new_df, palette=warm_palette)

# # 在每个柱子上添加数值标签
# for p in barplot.patches:
#     barplot.annotate(format(p.get_height(), '.2f'), 
#                      (p.get_x() + p.get_width() / 2., p.get_height()), 
#                      ha = 'center', va = 'center', 
#                      xytext = (0, 10), 
#                      textcoords = 'offset points', 
#                      fontsize=20)

# for label in barplot.get_xticklabels():
#     label.set_fontsize(25)
#     if label.get_text() in ['Llama 70B', 'GPT-3.5-Turbo']:
#         label.set_weight('bold')

# plt.legend(title="Metric", title_fontsize='20', fontsize='18')

# for label in barplot.get_yticklabels():
#     label.set_fontsize(20)

# plt.title("")
# plt.xlabel("")
# plt.ylabel("")

# plt.show()


# # %%



