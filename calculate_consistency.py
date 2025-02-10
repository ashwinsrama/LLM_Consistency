import json
import numpy as np
from adjustText import adjust_text
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path


#Elo scores obtained from LMSYS dataset
judge_elos = {'chatgpt-4o-latest': 1315.1596096064077, 'claude-3-haiku-20240307': 1178.3537536536885, 'claude-3-opus-20240229': 1247.6312370612695, 'claude-3.5-sonnet-20240620': 1270.975339008378, 'command-r': 1148.546452040903, 'command-r-plus': 1189.8069202864776, 'gemini-1.5-flash-002': 1272, 'gemini-1.5-pro-002': 1301, 'gemma-7b-it': 1037.3011946180168, 'gpt-4o-mini-2024-07-18': 1274.8487742018067, 'llama-3-8b-instruct': 1151.8633874882728, 'llama-3-70b-instruct': 1206.116131996019, 'llama-3.1-8b-instruct': 1167.6097444004965, 'llama-3.1-70b-instruct': 1245.82198737406, 'llama-3.1-405b-instruct': 1263.494122234061, 'mistral-7b-instruct': 1008.1704862874974, 'mistral-large-2407': 1250.6308684722885, 'mixtral-8x7b-instruct-v0.1': 1114.0, 'openchat-3.5-0106': 1091.4297939395608, 'phi-3-medium-4k-instruct': 1122.7668502642346, 'qwen1.5-14b-chat': 1108.5413057460873, 'starling-lm-7b-alpha': 1088.2909310158232, 'vicuna-13b': 1041.790097967258, 'zephyr-7b-beta': 1053.3444426537412}

#independent (non-overlapping) groups of chatbots randomly sampled with at least 20 contests between all pairs
group1 = ['chatglm-6b', 'gpt-4-0314', 'vicuna-7b', 'gpt-3.5-turbo-0314', 'RWKV-4-Raven-14B']
group2 = ['claude-2.1', 'tulu-2-dpo-70b', 'gpt-4-1106-preview', 'claude-instant-1', 'vicuna-33b']
group3 = ['mistral-medium', 'pplx-70b-online', 'gpt-3.5-turbo-1106', 'mixtral-8x7b-instruct-v0.1', 'gpt-4-0613']
group4 = ['llama-13b', 'dolly-v2-12b', 'stablelm-tuned-alpha-7b', 'oasst-pythia-12b', 'koala-13b']
group5 = ['mistral-7b-instruct', 'zephyr-7b-beta', 'llama-2-13b-chat', 'llama-2-7b-chat', 'gpt-3.5-turbo-0613']
group6 = ['mpt-7b-chat', 'palm-2', 'claude-1', 'vicuna-13b', 'fastchat-t5-3b']
group7 = ['wizardlm-13b', 'codellama-34b-instruct', 'claude-2.0', 'wizardlm-70b', 'llama-2-70b-chat']

#the 10 matchups with the greatest Elo difference
high_diffs = [("gpt-4-0314", "chatglm-6b"), ("chatglm-6b", "gpt-4-0314"), ("fastchat-t5-3b", "claude-1"), ("claude-1", "fastchat-t5-3b"), ("RWKV-4-Raven-14B", "gpt-4-0314"), ("gpt-4-0314", "RWKV-4-Raven-14B"), ("chatglm-6b", "gpt-3.5-turbo-0314"), ("gpt-3.5-turbo-0314", "chatglm-6b"), ("mpt-7b-chat", "claude-1"), ("claude-1", "mpt-7b-chat")]

#clusters of Judge LLMs in the correlation plot
high_cluster = ['chatgpt-4o-latest','claude-3-opus-20240229','claude-3.5-sonnet-20240620','llama-3-70b-instruct','gemini-1.5-flash-002','gemini-1.5-pro-002','gpt-4o-mini-2024-07-18','llama-3.1-70b-instruct','llama-3.1-405b-instruct','mistral-large-2407']
mid_cluster = ['claude-3-haiku-20240307','command-r','command-r-plus','llama-3-8b-instruct','llama-3.1-8b-instruct','mixtral-8x7b-instruct-v0.1','phi-3-medium-4k-instruct','qwen1.5-14b-chat']
low_cluster = ['gemma-7b-it','mistral-7b-instruct','openchat-3.5-0106','starling-lm-7b-alpha','vicuna-13b','zephyr-7b-beta']

#--------------------------------------------------------------------------------------------------
# LOAD DATA FROM JSON FILES

def import_jsons(input_folder:Path):
    # Initialize an empty dictionary to hold the data
    data_dict = {}

    # Iterate over all json files in the input folder
    for json_file in input_folder.glob('*_contest_data.json'):
        # Extract the judge name part of the filename
        name = json_file.stem.split('_contest_data')[0]
        
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # Add the data to the dictionary with the judge name as the key
        filtered_data = data
        #filtered_data = [match for match in data if match["first"] in group7 and match["second"] in group7] #group-wise preference data
        #filtered_data = [match for match in data if (match["first"], match["second"]) in high_diffs] #cluster-wise preference data
        data_dict[name] = filtered_data
    
    return data_dict

# import the contests data to be analyzed
print("...Loading Preference Data...")
input_folder = Path("./Preference Data")
contests_data = import_jsons(input_folder)

#--------------------------------------------------------------------------------------------------
# COMPUTE CONTESTANTS LIST

def extract_contestants(contests_data):
    # Initialize a set to hold all contestant names
    contestants = set()

    # Collect contestant names from the contests_data
    for matches in contests_data.values():
        for match in matches:
            contestants.add(match['first'])
            contestants.add(match['second'])

    # Convert the set of contestants to a sorted list
    contestants = sorted(contestants)

    # Step 2: Create a mapping from contestant names to indices
    contestant_index = {name: i for i, name in enumerate(contestants)}
    return contestant_index

# extract contestants
print("...Extracting Contestant LLMs...")
contestants_index = extract_contestants(contests_data)


#--------------------------------------------------------------------------------------------------
# COMPUTE WIN PROBABILITY

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

# compute win probabilities
print("...Calculating Matchup Win Probabilities...")
judges_nbmatch_matrices, judges_probability_matrices = compute_judges_win_probability(contestants_index, contests_data)


#--------------------------------------------------------------------------------------------------
#CALCULATE WVAR SCORE

def calc_consistency(judge_name):
    prob_matrix = judges_probability_matrices.get(judge_name)
    num_matrix = judges_nbmatch_matrices.get(judge_name)
    var = prob_matrix * (1-prob_matrix)

    weights = num_matrix/np.sum(num_matrix)

    valid_mask = num_matrix > 0
    valid_var = var[valid_mask]
    valid_weights = weights[valid_mask]

    return 1 - 4 * np.sum(valid_var * valid_weights)


print("...Calculating Consistency Metric...")
judge_data = {}
for judge_name in judge_elos:
    judge_data[judge_name] = [judge_elos[judge_name], calc_consistency(judge_name)]

#--------------------------------------------------------------------------------------------------
# PLOT CORRELATION

def plot_points_with_labels(data_dict):
    plt.figure(figsize=(10, 8))

    x_vals = []
    y_vals = []
    texts = []

    for key, value in data_dict.items():
        x, y = value
        x_vals.append(x)
        y_vals.append(y)

        plt.scatter(x, y, label=key)

        # Collect text objects for adjustment
        if x < 1200:
            text = plt.text(x, y, key, fontsize=12, ha='left', va='bottom')
        else:
            text = plt.text(x, y, key, fontsize=12, ha='right', va='bottom')
        texts.append(text)

    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Calculate the line of best fit
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_range = np.linspace(min(x_vals), max(x_vals), 100)
    y_range = slope * x_range + intercept
    plt.plot(x_range, y_range, color="red", linestyle="--")

    # Calculate Pearson correlation (r)
    correlation_matrix = np.corrcoef(x_vals, y_vals)
    pearson_r = correlation_matrix[0, 1]

    # Calculate errors
    y_pred = [slope * x + intercept for x in x_vals]  # Predicted y values
    x_pred = [(y - intercept) / slope for y in y_vals]  # Predicted x values

    mae_y = np.mean(np.abs(np.array(y_vals) - np.array(y_pred)))  # Mean Absolute Error in y
    mae_x = np.mean(np.abs(np.array(x_vals) - np.array(x_pred)))  # Mean Absolute Error in x
    
    # Print error values
    print(f"Mean Absolute Error (MAE) in y-direction: {mae_y:.4f}")
    print(f"Mean Absolute Error (MAE) in x-direction: {mae_x:.4f}")

    # Add legend with Pearson correlation
    legend_text = f"Pearson = {pearson_r:.2f}"
    plt.legend(handles=[], title=legend_text, loc="upper left", frameon=True, title_fontsize=14)

    # Note the swapped axes labels
    plt.grid(True)
    plt.ylabel("Consistency Score", fontsize=14)
    plt.xlabel("LMSYS Elo Score of Judge LLM", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()


print("...Plotting Results...")
plot_points_with_labels(judge_data)