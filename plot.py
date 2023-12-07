import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

new_data = {
    "Model": [
        "CARE-GNN",
        "CARE-GNN",
        "CARE-GNN",
        "PC-GNN",
        "PC-GNN",
        "PC-GNN",
        "GTAN",
        "GTAN",
        "GTAN",
        "Llama 70B",
        "Llama 70B",
        "Llama 70B",
        "GPT-3.5-Turbo",
        "GPT-3.5-Turbo",
        "GPT-3.5-Turbo",
    ],
    "Metric": [
        "F1",
        "AP",
        "AUC",
        "F1",
        "AP",
        "AUC",
        "F1",
        "AP",
        "AUC",
        "F1",
        "AP",
        "AUC",
        "F1",
        "AP",
        "AUC",
    ],
    "Value": [
        0.5971,
        0.3285,
        0.7420,  # CARE-GNN
        0.4895,
        0.3238,
        0.8101,  # PC-GNN
        0.8280,
        0.7930,
        0.9348,  # GTAN
        0.2497,
        0.1502,
        0.5064,  # Llama 70B
        0.1980,
        0.1543,
        0.5104,  # GPT-3.5-Turbo
    ],
}

new_df = pd.DataFrame(new_data)
new_df = new_df.dropna()

# Warm color palette
warm_palette = sns.color_palette("YlOrRd", n_colors=3)

sns.set_context("poster")  # Bigger font size
sns.set(style="darkgrid", rc={"axes.facecolor": "0.9"})


plt.figure(figsize=(16, 10))
barplot = sns.barplot(
    x="Model", y="Value", hue="Metric", data=new_df, palette=warm_palette
)

# add text annotation corresponding to the values
for p in barplot.patches:
    barplot.annotate(
        format(p.get_height(), ".4f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=20,
    )

for label in barplot.get_xticklabels():
    label.set_fontsize(25)
    if label.get_text() in ["Llama 70B", "GPT-3.5-Turbo"]:
        label.set_weight("bold")

plt.legend(title="Metric", title_fontsize="20", fontsize="18")

for label in barplot.get_yticklabels():
    label.set_fontsize(20)

plt.title("")
plt.xlabel("")
plt.ylabel("")

output_path = "./output"
plt.savefig(os.path.join(output_path, 'results.png'))
print("Done!")
