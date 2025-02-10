import openai
import os
import json
import csv
import concurrent.futures
from random import shuffle


def compare_pair(judge_name, i, idx, a, b, q, resp_a, resp_b, contest_data):
    if {"first": a, "second": b, "question": idx, "winner": a} in contest_data or {"first": a, "second": b, "question": idx, "winner": b} in contest_data or {"first": a, "second": b, "question": idx, "winner": "Tie"} in contest_data:
        return None  # Already compared, skip

    attempts = 0
    data = {"winner": None}

    while not data["winner"] and attempts < 20: #if connection not reliable, try at least 20 times.
        try:
            eval_prompt = load_markdown(f'reasoning_prompt_judge.md')
            eval_prompt = str(eval_prompt.format(PROMPT=q, ANSWER1=resp_a, ANSWER2=resp_b))
            reasoning_message = [{'role': 'user', 'content': eval_prompt}]
            reasoning = client.chat.completions.create(model=judge_name, messages=reasoning_message, temperature=0.0).choices[-1].message.content
            #print(reasoning, "\n")
            if reasoning:
                eval_prompt2 = load_markdown(f'preference_prompt_judge.md')
                eval_prompt2 = str(eval_prompt2.format(REASONING=str(reasoning)))
                preference_message = [{'role': 'user', 'content': eval_prompt2}]
                preference = client.chat.completions.create(model=judge_name, messages=preference_message, temperature=0.0).choices[-1].message.content
            else:
                preference = None
            print(f"Attempt: {attempts}, Preference: {preference}\n")

        except Exception as e:
            print(f"Error calling judge {judge_name}:\n{e}\n")
            return None  # Error, skip this pair

        if preference:
            data = {"first": a, "second": b, "question": idx, "winner": 
                    (
                        a if "1" in preference and "2" not in preference and "0" not in preference
                        else (b if "2" in preference and "1" not in preference and "0" not in preference
                        else ("Tie" if "0" in preference and "1" not in preference and "2" not in preference
                        else None))
                    )
            }

            if data["winner"]:
                print(f"{data}\t{i+1} of {len(chatbots_response)} pairs compared for '{judge_name}'.")
                return data  # Return the result to append later
            else:
                attempts += 1
        else:
            attempts += 1

    return None  # In case no winner determined


def get_prefs(judge_name, overwrite=False):
    # Load contest data
    if overwrite or not os.path.isfile(f'judge_data_agg/{judge_name}_contest_data.json'):
        contest_data = []
        write_list_to_json(contest_data, f'judge_data_agg/{judge_name}_contest_data.json')
    else:
        contest_data = read_json_to_list(f'judge_data_agg/{judge_name}_contest_data.json')

    print(f"Initiating Pairwise Comparison for '{judge_name}'.\n")

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, (idx, a, b, q, resp_a, resp_b) in enumerate(chatbots_response):
            futures.append(executor.submit(compare_pair, judge_name, i, idx, a, b, q, resp_a, resp_b, contest_data))

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:  # If result is not None, append it to the contest_data
                contest_data.append(result)
                write_list_to_json(contest_data, f'judge_data_agg/{judge_name}_contest_data.json')

    print(f"Completed Pairwise Comparison for '{judge_name}'.\n")


def read_json_to_list(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def write_list_to_json(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file, indent=4)


def load_markdown(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def csv_to_tuple_list(file_path):
    tuple_list = []
    
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row
        for row in csvreader:
            tuple_list.append(tuple(str(element) for element in row[:6]))
    shuffle(tuple_list)
    return tuple_list


client = openai.OpenAI(
api_key='<INSERT OpenAI API Key>',
)

chatbots_response = csv_to_tuple_list(f'lmsys_prefs_csv/group1.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group2.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group3.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group4.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group5.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group6.csv') + csv_to_tuple_list(f'lmsys_prefs_csv/group7.csv') 

os.makedirs(f'judge_data_agg', exist_ok=True)


if __name__ == '__main__':
    judge_name = "<judge LLM>"
    overwrite = False
    get_prefs(judge_name, overwrite)
    
