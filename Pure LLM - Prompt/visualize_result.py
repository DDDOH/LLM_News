import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os


result_dir = 'Pure LLM - Prompt/prompt-results temperature zero'

result_gpt = np.load(os.path.join(result_dir, 'result_gpt.npy'), allow_pickle=True)
significant_mask = np.load(os.path.join(result_dir, 'significant_mask.npy'), allow_pickle=True)
CI_FOR_SIGNIFICANT = False # if True, calculate CI for significant news only, otherwise calculate CI for all news

def bootstrap_ci(data, n_bootstrap=1000, ci=95, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_bound = np.percentile(bootstrap_means, (100-ci)/2)
    upper_bound = np.percentile(bootstrap_means, 100 - (100-ci)/2)
    return lower_bound, upper_bound

# Apply bootstrap on acc_vec for each row and update ci_lower and ci_upper
for experiment in result_gpt:
    correct = [1 if x == y else 0 for x, y in zip(experiment['true_label'], experiment['pred_label'])]
    if CI_FOR_SIGNIFICANT:
        correct = np.array(correct)[significant_mask]
    lower, upper = bootstrap_ci(correct, random_state=42)
    # result_gpt_df.at[i, 'ci_lower_bs'] = lower
    # result_gpt_df.at[i, 'ci_upper_bs'] = upper
    experiment['ci_lower_bs'] = lower
    experiment['ci_upper_bs'] = upper

# Show CI for each model in a more readable format
print("Model Name".ljust(20), "CI Lower Bound".ljust(15), "CI Upper Bound".ljust(15))
print("-" * 50)
for experiment in result_gpt:
    model_name = experiment['model_name']
    ci_lower = experiment['ci_lower_bs']
    ci_upper = experiment['ci_upper_bs']
    print(f"{model_name.ljust(20)} {str(ci_lower).ljust(20)} {str(ci_upper).ljust(20)}")

  
model_list = [
    "GPT-3.5-0-0",
    "GPT-3.5-2-0",
    "GPT-3.5-2-1",
    "GPT-3.5-5-0",
    "GPT-3.5-5-1",
    "GPT-4-0-0",
    "GPT-4-2-0",
    "GPT-4-2-1",
    "GPT-4-5-0",
    "GPT-4-5-1"
]

# mean_list = np.array(result_gpt_df['acc'])
mean_list = [experiment['acc'] for experiment in result_gpt]
mean_diff_matrix = np.zeros((len(model_list) + 1, len(model_list) + 1))
pval_matrix = np.zeros((len(model_list) + 1, len(model_list) + 1))

# Append "random guess" model to the list
model_list.append("Random Guess")


for i in range(len(model_list) - 1):  # Exclude "random guess" in this loop
    for j in range(len(model_list) - 1):  # Exclude "random guess" in this loop
        rvs1 = [1 if x == y else 0 for x, y in zip(result_gpt[i]['true_label'], result_gpt[i]['pred_label'])]
        rvs2 = [1 if x == y else 0 for x, y in zip(result_gpt[j]['true_label'], result_gpt[j]['pred_label'])]
        if CI_FOR_SIGNIFICANT:
            rvs1 = np.array(rvs1)[significant_mask]
            rvs2 = np.array(rvs2)[significant_mask]
        # rvs1 = (np.array(result_gpt_df['pred_label'])[i] == true_label_sig).astype(int)
        # rvs2 = (np.array(result_gpt_df['pred_label'])[j] == true_label_sig).astype(int)
        stat, pval = stats.ttest_rel(rvs1, rvs2)

        # Store mean difference and p-value
        mean_diff_matrix[i, j] = (np.mean(rvs1) - np.mean(rvs2)) * 100
        pval_matrix[i, j] = pval

# Compare each model with "random guess"
for i in range(len(model_list) - 1):
    # rvs = (np.array(result_gpt_df['pred_label'])[i] == true_label_sig).astype(int)
    rvs = [1 if x == y else 0 for x, y in zip(result_gpt[i]['true_label'], result_gpt[i]['pred_label'])]
    if CI_FOR_SIGNIFICANT:
        rvs = np.array(rvs)[significant_mask]
    mean_diff = (np.mean(rvs) - 0.3302) * 100
    stat, pval = stats.ttest_1samp(rvs, 0.5)

    # Store mean difference and p-value
    mean_diff_matrix[i, len(model_list) - 1] = mean_diff
    mean_diff_matrix[len(model_list) - 1, i] = -mean_diff
    pval_matrix[i, len(model_list) - 1] = pval
    pval_matrix[len(model_list) - 1, i] = pval

# Store the "random guess" model's self-comparison (diagonal)
mean_diff_matrix[len(model_list) - 1, len(model_list) - 1] = 0
pval_matrix[len(model_list) - 1, len(model_list) - 1] = 1

# Move "random guess" to the first place
model_list = ["Random Guess"] + model_list[:-1]
mean_diff_matrix = np.vstack([mean_diff_matrix[-1], mean_diff_matrix[:-1]])
mean_diff_matrix = np.hstack([mean_diff_matrix[:, -1].reshape(-1, 1), mean_diff_matrix[:, :-1]])
pval_matrix = np.vstack([pval_matrix[-1], pval_matrix[:-1]])
pval_matrix = np.hstack([pval_matrix[:, -1].reshape(-1, 1), pval_matrix[:, :-1]])

mean_diff_df = pd.DataFrame(mean_diff_matrix, index=model_list, columns=model_list)
pval_df = pd.DataFrame(pval_matrix, index=model_list, columns=model_list)

mask_pval = pval_df >= 0.05

# Create a mask to hide the upper triangle and the diagonal
upper_triangle_mask = np.triu(np.ones_like(mean_diff_matrix, dtype=bool))
np.fill_diagonal(upper_triangle_mask, True)

# Combined mask to hide upper triangle and non-significant p-values
combined_mask = mask_pval | upper_triangle_mask

# Create a cross-style mask for p-values greater than or equal to 0.05
cross_mask = np.zeros_like(mean_diff_matrix, dtype=bool)
cross_mask[mask_pval & ~upper_triangle_mask] = True

# Plot the heatmap of mean differences with the combined mask applied
plt.figure(figsize=(12, 10))
ax = sns.heatmap(mean_diff_df, mask=upper_triangle_mask, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Difference (%)'},
                  linewidths=.5, linecolor='white', vmin=-15, vmax=15, center=0)

# Apply cross-style mask
for i in range(len(model_list)):
    for j in range(len(model_list)):
        if cross_mask[i, j]:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='white', lw=1, hatch='\\'))

plt.title('Mean Differences in Accuracy (%)')
plt.xlabel('Model')
plt.ylabel('Model')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig(os.path.join(result_dir, 'mean_differences_heatmap_multiple.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# Create a mask to hide the upper triangle for the p-value heatmap
lower_triangle_mask = np.tril(np.ones_like(pval_df, dtype=bool), -1)

# Plot the heatmap of p-values with the lower triangle mask applied
plt.figure(figsize=(12, 10))
sns.heatmap(pval_df, mask=~lower_triangle_mask, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'P-value'}, linewidths=.5, linecolor='white')
plt.title('P-values')
plt.xlabel('Model')
plt.ylabel('Model')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig(os.path.join(result_dir, 'p_values_heatmap_multiple.pdf'), format='pdf', bbox_inches='tight')
plt.close()
