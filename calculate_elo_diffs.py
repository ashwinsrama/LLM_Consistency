import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


chatbot_elos = { 'RWKV-4-Raven-14B': 921.7902833486501, 'chatglm-6b': 878.9863809080546, 'claude-1': 1148.8604494209847, 'claude-2.0': 1131.728698244663, 'claude-2.1': 1118.1627718231223, 'claude-instant-1': 1111.2746090544983, 'codellama-34b-instruct': 1042.4602761657923, 'dolly-v2-12b': 822.0815633556351, 'fastchat-t5-3b': 868.2928632213884, 'gpt-3.5-turbo-0314': 1106.2606294737388, 'gpt-3.5-turbo-0613': 1116.93208727063, 'gpt-3.5-turbo-1106': 1067.8682557045763, 'gpt-4-0314': 1186.1240136265737, 'gpt-4-0613': 1162.4537519222738, 'gpt-4-1106-preview': 1250.842729376194, 'koala-13b': 964.2133598283524, 'llama-13b': 798.9227074608199, 'llama-2-13b-chat': 1063.2550550891224, 'llama-2-70b-chat': 1093.0834852202363, 'llama-2-7b-chat': 1036.9451766295156, 'mistral-7b-instruct': 1008.1704862874974, 'mistral-medium': 1147.7158092095751, 'mixtral-8x7b-instruct-v0.1': 1114.0, 'mpt-7b-chat': 927.2972558743481, 'oasst-pythia-12b': 893.5040382808504, 'palm-2': 1003.4764664659062, 'pplx-70b-online': 1077.9880951202867, 'stablelm-tuned-alpha-7b': 839.685717789491, 'tulu-2-dpo-70b': 1099.3892429724285, 'vicuna-13b': 1041.790097967258, 'vicuna-33b': 1090.711881676661, 'vicuna-7b': 1005.0248613656842, 'wizardlm-13b': 1058.6029024510558, 'wizardlm-70b': 1106.1373952969384, 'zephyr-7b-beta': 1053.3444426537412 }
judges = ['chatgpt-4o-latest','claude-3-haiku-20240307','claude-3-opus-20240229','claude-3.5-sonnet-20240620','command-r','command-r-plus','gemini-1.5-flash-002','gemini-1.5-pro-002','gemma-7b-it','gpt-4o-mini-2024-07-18','llama-3-8b-instruct','llama-3-70b-instruct','llama-3.1-8b-instruct','llama-3.1-70b-instruct','llama-3.1-405b-instruct','mistral-7b-instruct','mistral-large-2407','mixtral-8x7b-instruct-v0.1','openchat-3.5-0106','phi-3-medium-4k-instruct','qwen1.5-14b-chat','starling-lm-7b-alpha','vicuna-13b','zephyr-7b-beta']
unique_pairs = [('vicuna-7b', 'gpt-3.5-turbo-0314'), ('gpt-3.5-turbo-0314', 'vicuna-7b'), ('chatglm-6b', 'vicuna-7b'), ('gpt-4-0314', 'chatglm-6b'), ('RWKV-4-Raven-14B', 'chatglm-6b'), ('gpt-3.5-turbo-0314', 'gpt-4-0314'), ('gpt-4-0314', 'gpt-3.5-turbo-0314'), ('chatglm-6b', 'gpt-4-0314'), ('RWKV-4-Raven-14B', 'vicuna-7b'), ('chatglm-6b', 'gpt-3.5-turbo-0314'), ('chatglm-6b', 'RWKV-4-Raven-14B'), ('vicuna-7b', 'RWKV-4-Raven-14B'), ('RWKV-4-Raven-14B', 'gpt-3.5-turbo-0314'), ('gpt-4-0314', 'vicuna-7b'), ('vicuna-7b', 'gpt-4-0314'), ('RWKV-4-Raven-14B', 'gpt-4-0314'), ('gpt-4-0314', 'RWKV-4-Raven-14B'), ('vicuna-7b', 'chatglm-6b'), ('gpt-3.5-turbo-0314', 'chatglm-6b'), ('gpt-3.5-turbo-0314', 'RWKV-4-Raven-14B'), ('gpt-4-1106-preview', 'claude-instant-1'), ('claude-instant-1', 'gpt-4-1106-preview'), ('claude-2.1', 'gpt-4-1106-preview'), ('tulu-2-dpo-70b', 'claude-2.1'), ('vicuna-33b', 'claude-2.1'), ('claude-instant-1', 'tulu-2-dpo-70b'), ('tulu-2-dpo-70b', 'claude-instant-1'), ('claude-2.1', 'tulu-2-dpo-70b'), ('vicuna-33b', 'gpt-4-1106-preview'), ('claude-2.1', 'claude-instant-1'), ('claude-2.1', 'vicuna-33b'), ('gpt-4-1106-preview', 'vicuna-33b'), ('vicuna-33b', 'claude-instant-1'), ('tulu-2-dpo-70b', 'gpt-4-1106-preview'), ('gpt-4-1106-preview', 'tulu-2-dpo-70b'), ('vicuna-33b', 'tulu-2-dpo-70b'), ('tulu-2-dpo-70b', 'vicuna-33b'), ('gpt-4-1106-preview', 'claude-2.1'), ('claude-instant-1', 'claude-2.1'), ('claude-instant-1', 'vicuna-33b'), ('gpt-3.5-turbo-1106', 'mixtral-8x7b-instruct-v0.1'), ('mixtral-8x7b-instruct-v0.1', 'gpt-3.5-turbo-1106'), ('mistral-medium', 'gpt-3.5-turbo-1106'), ('pplx-70b-online', 'mistral-medium'), ('gpt-4-0613', 'mistral-medium'), ('mixtral-8x7b-instruct-v0.1', 'pplx-70b-online'), ('pplx-70b-online', 'mixtral-8x7b-instruct-v0.1'), ('mistral-medium', 'pplx-70b-online'), ('gpt-4-0613', 'gpt-3.5-turbo-1106'), ('mistral-medium', 'mixtral-8x7b-instruct-v0.1'), ('mistral-medium', 'gpt-4-0613'), ('gpt-3.5-turbo-1106', 'gpt-4-0613'), ('gpt-4-0613', 'mixtral-8x7b-instruct-v0.1'), ('pplx-70b-online', 'gpt-3.5-turbo-1106'), ('gpt-3.5-turbo-1106', 'pplx-70b-online'), ('gpt-4-0613', 'pplx-70b-online'), ('pplx-70b-online', 'gpt-4-0613'), ('gpt-3.5-turbo-1106', 'mistral-medium'), ('mixtral-8x7b-instruct-v0.1', 'mistral-medium'), ('mixtral-8x7b-instruct-v0.1', 'gpt-4-0613'), ('stablelm-tuned-alpha-7b', 'oasst-pythia-12b'), ('oasst-pythia-12b', 'stablelm-tuned-alpha-7b'), ('llama-13b', 'stablelm-tuned-alpha-7b'), ('dolly-v2-12b', 'llama-13b'), ('koala-13b', 'llama-13b'), ('oasst-pythia-12b', 'dolly-v2-12b'), ('dolly-v2-12b', 'oasst-pythia-12b'), ('llama-13b', 'dolly-v2-12b'), ('koala-13b', 'stablelm-tuned-alpha-7b'), ('llama-13b', 'oasst-pythia-12b'), ('llama-13b', 'koala-13b'), ('stablelm-tuned-alpha-7b', 'koala-13b'), ('koala-13b', 'oasst-pythia-12b'), ('dolly-v2-12b', 'stablelm-tuned-alpha-7b'), ('stablelm-tuned-alpha-7b', 'dolly-v2-12b'), ('koala-13b', 'dolly-v2-12b'), ('dolly-v2-12b', 'koala-13b'), ('stablelm-tuned-alpha-7b', 'llama-13b'), ('oasst-pythia-12b', 'llama-13b'), ('oasst-pythia-12b', 'koala-13b'), ('llama-2-13b-chat', 'llama-2-7b-chat'), ('llama-2-7b-chat', 'llama-2-13b-chat'), ('mistral-7b-instruct', 'llama-2-13b-chat'), ('zephyr-7b-beta', 'mistral-7b-instruct'), ('gpt-3.5-turbo-0613', 'mistral-7b-instruct'), ('llama-2-7b-chat', 'zephyr-7b-beta'), ('zephyr-7b-beta', 'llama-2-7b-chat'), ('mistral-7b-instruct', 'zephyr-7b-beta'), ('gpt-3.5-turbo-0613', 'llama-2-13b-chat'), ('mistral-7b-instruct', 'llama-2-7b-chat'), ('mistral-7b-instruct', 'gpt-3.5-turbo-0613'), ('llama-2-13b-chat', 'gpt-3.5-turbo-0613'), ('gpt-3.5-turbo-0613', 'llama-2-7b-chat'), ('zephyr-7b-beta', 'llama-2-13b-chat'), ('llama-2-13b-chat', 'zephyr-7b-beta'), ('gpt-3.5-turbo-0613', 'zephyr-7b-beta'), ('zephyr-7b-beta', 'gpt-3.5-turbo-0613'), ('llama-2-13b-chat', 'mistral-7b-instruct'), ('llama-2-7b-chat', 'mistral-7b-instruct'), ('llama-2-7b-chat', 'gpt-3.5-turbo-0613'), ('claude-1', 'vicuna-13b'), ('vicuna-13b', 'claude-1'), ('mpt-7b-chat', 'claude-1'), ('palm-2', 'mpt-7b-chat'), ('fastchat-t5-3b', 'mpt-7b-chat'), ('vicuna-13b', 'palm-2'), ('palm-2', 'vicuna-13b'), ('mpt-7b-chat', 'palm-2'), ('fastchat-t5-3b', 'claude-1'), ('mpt-7b-chat', 'vicuna-13b'), ('mpt-7b-chat', 'fastchat-t5-3b'), ('claude-1', 'fastchat-t5-3b'), ('fastchat-t5-3b', 'vicuna-13b'), ('palm-2', 'claude-1'), ('claude-1', 'palm-2'), ('fastchat-t5-3b', 'palm-2'), ('palm-2', 'fastchat-t5-3b'), ('claude-1', 'mpt-7b-chat'), ('vicuna-13b', 'mpt-7b-chat'), ('vicuna-13b', 'fastchat-t5-3b'), ('claude-2.0', 'wizardlm-70b'), ('wizardlm-70b', 'claude-2.0'), ('wizardlm-13b', 'claude-2.0'), ('codellama-34b-instruct', 'wizardlm-13b'), ('llama-2-70b-chat', 'wizardlm-13b'), ('wizardlm-70b', 'codellama-34b-instruct'), ('codellama-34b-instruct', 'wizardlm-70b'), ('wizardlm-13b', 'codellama-34b-instruct'), ('llama-2-70b-chat', 'claude-2.0'), ('wizardlm-13b', 'wizardlm-70b'), ('wizardlm-13b', 'llama-2-70b-chat'), ('claude-2.0', 'llama-2-70b-chat'), ('llama-2-70b-chat', 'wizardlm-70b'), ('codellama-34b-instruct', 'claude-2.0'), ('claude-2.0', 'codellama-34b-instruct'), ('llama-2-70b-chat', 'codellama-34b-instruct'), ('codellama-34b-instruct', 'llama-2-70b-chat'), ('claude-2.0', 'wizardlm-13b'), ('wizardlm-70b', 'wizardlm-13b'), ('wizardlm-70b', 'llama-2-70b-chat')]


def import_jsons(input_folder:Path, mods_to_include):
    # Initialize an empty dictionary to hold the data
    data_dict = {}

    # Iterate over all json files in the input folder
    for json_file in input_folder.glob('*_contest_data.json'):
        # Extract the judge name part of the filename
        name = json_file.stem.split('_contest_data')[0]
        
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        new_data = []
        for mod in mods_to_include:
            for dct in data:
                if (dct["first"], dct["second"]) == mod:
                    new_data.append(dct)

        # Add the data to the dictionary with the judge name as the key
        data_dict[name] = new_data
    
    return data_dict


def compute_win_probabilities(contestants_index, contests):
    # Initialize a matrix for win counts and match counts
    num_contestants = len(contestants_index)
    win_matrix = np.zeros((num_contestants, num_contestants))
    match_matrix = np.zeros((num_contestants, num_contestants))

    # Populate the win and match matrices
    for match in contests:
        first_idx = contestants_index[match['first']]
        second_idx = contestants_index[match['second']]
        
        # Increment the match count
        match_matrix[first_idx, second_idx] += 1
        match_matrix[second_idx, first_idx] += 1
        
        # Increment the win count
        if match['winner'] == match['first']:
            win_matrix[first_idx, second_idx] += 1
        elif match['winner'] == match['second']:
            win_matrix[second_idx, first_idx] += 1
        else:
            win_matrix[first_idx, second_idx] += 0.5
            win_matrix[second_idx, first_idx] += 0.5

    # Calculate the win probability matrix
    prob_matrix = np.divide(win_matrix, match_matrix, out=np.zeros_like(win_matrix), where=match_matrix!=0)
    return match_matrix, prob_matrix


def compute_judges_win_probability(contestants_index, all_contests:Dict[str,Dict]):
    judges_nbmatch_matrices = dict()
    judges_probability_matrices = dict()
    for (judge,contests) in all_contests.items():
        match_matrix, prob_matrix = compute_win_probabilities(contestants_index, contests)
        judges_nbmatch_matrices[judge] = match_matrix
        judges_probability_matrices[judge] = prob_matrix
    return judges_nbmatch_matrices, judges_probability_matrices


def compute_variance(judge_pairs):
    variances = {}
    
    # Get the keys of the inner dictionary (pairs)
    inner_keys = next(iter(judge_pairs.values())).keys()
    
    # Calculate variance for each inner key
    for pair in inner_keys:
        # Collect all values corresponding to this pair across all outer dictionaries
        values = [judge_pairs[judge][pair] for judge in judge_pairs]
        # Compute the variance
        variances[pair] = np.var(values)
    
    sorted_variances = dict(sorted(variances.items(), key=lambda item: item[1], reverse=True))
    return sorted_variances


def scatter_plot_from_tuples(diffs, varis):
    plt.figure(figsize=(10, 8))
    plt.scatter(diffs, varis)

    # Calculate Pearson correlation
    r, _ = pearsonr(diffs, varis)

    # Calculate line of best fit
    slope, intercept = np.polyfit(diffs, varis, 1)
    line_x = np.linspace(min(diffs), max(diffs), 100)
    line_y = intercept + slope * line_x

    # Plot line of best fit in red
    plt.plot(line_x, line_y, 'r')

    # Add labels and title
    plt.xlabel("Pairwise Elo Score Difference", fontsize=14)
    plt.ylabel("Pair Variance Across All Judges", fontsize=14)
    plt.title("Correlation Between Elo Difference and Pair Variance", fontsize=14)

    # Adjust y-axis intervals to be reasonable
    min_y, max_y = min(varis), max(varis)
    interval = (max_y - min_y) / 5  # Divide range into 5 intervals
    plt.yticks([min_y + i * interval for i in range(6)])
    plt.grid(True)
    plt.tight_layout()

    legend_text = f"Pearson = {r:.2f}"
    plt.legend(handles=[], title=legend_text, loc="upper right", frameon=True, title_fontsize=10)

    plt.show()


#--------------------------------------------------------------------------------------------------
# GET RESULTS - ELO PAIRS DIFFS


judge_pairs = {judge_name : {pair : 0 for pair in unique_pairs} for judge_name in judges}
input_folder = Path("./Preference Data")
contests_data = import_jsons(input_folder, unique_pairs)
contestants_index = {name: i for i, name in enumerate(list(chatbot_elos.keys()))}
judges_nbmatch_matrices, judges_probability_matrices = compute_judges_win_probability(contestants_index, contests_data)


for judge_name in judge_pairs:
    prob_matrix = judges_probability_matrices.get(judge_name)
    var = prob_matrix * (1-prob_matrix)
    for pair in judge_pairs[judge_name]:
        first_idx = contestants_index[pair[0]]
        second_idx = contestants_index[pair[1]]
        judge_pairs[judge_name][pair] = var[first_idx, second_idx]


pairwise_variance = compute_variance(judge_pairs)
varis = []
diffs = []

for pair in pairwise_variance:
    p0 = pair[0]
    p1 = pair[1]
    diff = abs(round(chatbot_elos[p0] - chatbot_elos[p1]))
    var = pairwise_variance[pair]  # Keep it as float for plotting
    varis.append(var)
    diffs.append(diff)

scatter_plot_from_tuples(diffs, varis)