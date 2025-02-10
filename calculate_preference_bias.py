import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


input_folder = Path("./Preference Data")
sigfigs = 4
dct = {}

judge_variances = {'chatgpt-4o-latest': [1315.1596096064077, 0], 'claude-3-haiku-20240307': [1178.3537536536885, 0], 'claude-3-opus-20240229': [1247.6312370612695, 0], 'claude-3.5-sonnet-20240620': [1270.975339008378, 0], 'command-r': [1148.546452040903, 0], 'command-r-plus': [1189.8069202864776, 0], 'gemini-1.5-flash-002': [1272, 0], 'gemini-1.5-pro-002': [1301, 0], 'gemma-7b-it': [1037.3011946180168, 0], 'gpt-4o-mini-2024-07-18': [1274.8487742018067, 0], 'llama-3-8b-instruct': [1151.8633874882728, 0], 'llama-3-70b-instruct': [1206.116131996019, 0], 'llama-3.1-8b-instruct': [1167.6097444004965, 0], 'llama-3.1-70b-instruct': [1245.82198737406, 0], 'llama-3.1-405b-instruct': [1263.494122234061, 0], 'mistral-7b-instruct': [1008.1704862874974, 0], 'mistral-large-2407': [1250.6308684722885, 0], 'mixtral-8x7b-instruct-v0.1': [1114.0, 0], 'openchat-3.5-0106': [1091.4297939395608, 0], 'phi-3-medium-4k-instruct': [1122.7668502642346, 0], 'qwen1.5-14b-chat': [1108.5413057460873, 0], 'starling-lm-7b-alpha': [1088.2909310158232, 0], 'vicuna-13b': [1041.790097967258, 0], 'zephyr-7b-beta': [1053.3444426537412, 0]}

def plot_preference_bars(dct):
    sorted_models = sorted(dct.keys(), key=lambda k: judge_variances[k][0], reverse=True)
    sorted_values = [dct[model] for model in sorted_models]

    # Prepare data for plotting
    components = ['First', 'Second', 'Tie']

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    bottoms = np.zeros(len(sorted_models))

    for i, component in enumerate(components):
        ax.bar(sorted_models, [v[i] for v in sorted_values], bottom=bottoms, label=component)
        bottoms += [v[i] for v in sorted_values]

    # Add labels inside the bars
    for idx, model in enumerate(sorted_models):
        total_height = bottoms[idx]
        ax.text(idx, 0.03, f"{model} ({round(judge_variances[model][0])})", ha='center', va='bottom', fontsize=14, rotation=90, color='yellow', fontweight='bold')  # Adjusted position

    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set y-axis ticks at intervals of 0.1

    # Customize plot
    ax.set_ylabel('Proportion of Contests', fontsize=14)
    ax.set_xlabel('Judge LLMs', fontsize=14)
    ax.set_title('Judge LLM Preference Results on Contests', fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis labels below the plot
    plt.tight_layout()

    # Show plot
    plt.show()


def count_preferences():
    for json_file in input_folder.glob('*_contest_data.json'):
        # Extract the judge name part of the filename
        name = json_file.stem.split('_contest_data')[0]
        
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        first = 0
        second = 0
        tie = 0
        items = len(data)

        for match in data:
            if match["winner"] == match["first"]:
                first += 1
            elif match["winner"] == match["second"]:
                second += 1
            else:
                tie += 1

        dct[name] = [round(first/items, sigfigs), round(second/items, sigfigs), round(tie/items, sigfigs)]
    return dct


data = count_preferences()
plot_preference_bars(data)



