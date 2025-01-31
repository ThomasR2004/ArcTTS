import json
import os
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer
import ast

# Paths to models
MODEL_1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_2 = "ibm-granite/granite-3.1-8b-instruct"

ADAPTER_PATH = "/gpfs/home6/trietman/ArcTTS/checkpoint-43/" # load finetune
# Load the first model
model_1, tokenizer_1 = FastLanguageModel.from_pretrained(
    model_name=MODEL_1,
    max_seq_length=8192,
    load_in_4bit=True,
    dtype="auto",
    device_map="auto",
)
model_1.load_adapter(ADAPTER_PATH)
FastLanguageModel.for_inference(model_1)

# Load the second model
model_2, tokenizer_2 = FastLanguageModel.from_pretrained(
    model_name=MODEL_2,
    max_seq_length=8192,
    load_in_4bit=True,
    dtype="auto",
    device_map="auto",
)
FastLanguageModel.for_inference(model_2)

def run_first_llm(tasks_dict, system_prompt=None):
    output = {}
    removed_sections = {}
    
    # Remove all keys named "test" from task_data
    filtered_tasks_dict = {}
    for task_id, task_data in tasks_dict.items():
        removed_sections[task_id] = {k: v for k, v in task_data.items() if k == "test"}
        filtered_tasks_dict[task_id] = {k: v for k, v in task_data.items() if k != "test"}

    for task_id, task_data in filtered_tasks_dict.items():
        # Include the system prompt if provided
        prompt = f"{system_prompt}\n\nTask Data: {json.dumps(task_data)}"

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer_1.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer_1)
        generated_tokens = model_1.generate(
            input_ids=inputs, streamer=text_streamer, max_new_tokens=7000, use_cache=True, repetition_penalty = 1.4
        )

        # Decode the tensor output to a readable string
        output_text = tokenizer_1.decode(generated_tokens[0], skip_special_tokens=True)
        description = extract_after_think(output_text, system_prompt)

        output[task_id] = {"description": description}
    
    return output, removed_sections


def run_second_llm(intermediate_results, removed_sections, system_prompt=None):
    final_output = {}

    for task_id, description in intermediate_results.items():
        # Retrieve the removed "test" section
        modified_json = removed_sections.get(task_id, {})

        prompt = f"{system_prompt}\n\nDescription: {description}\n\nTask Data: {json.dumps(modified_json)}"

        # Fix message format for `apply_chat_template`
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer_2.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer_2)
        generated_tokens = model_2.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=2000, use_cache=True)

        # Decode the tensor output to a readable string
        generated_code = tokenizer_2.decode(generated_tokens[0], skip_special_tokens=True)

        # Extract JSON content using regex
        json_match = re.search(r'\{.*\}', generated_code, re.DOTALL)
        if json_match:
            generated_code = json_match.group(0)

        final_output[task_id] = {"generated_code": generated_code}
        print(task_id)

    return final_output

def process_output_to_dict(final_results, existing_dict=None):
    """
    Processes the final_results dictionary and extracts the SECOND valid list of lists
    from the "generated_code" field for each task_id. If the task_id already exists
    in the existing_dict, the new output is stored as "attempt_2".

    Args:
        final_results (dict): A dictionary where keys are task_ids and values are
                              dictionaries containing a "generated_code" field.
        existing_dict (dict): An existing dictionary containing previous attempts.
                              Defaults to None.

    Returns:
        dict: A dictionary mapping task_ids to dictionaries with "attempt_1" and
              (optionally) "attempt_2" as keys, containing the extracted lists.
    """
    if existing_dict is None:
        existing_dict = {}

    for task_id, result_data in final_results.items():
        # Extract the "generated_code" (which may contain extra text)
        generated_code = result_data.get("generated_code", "")

        # Use regex to find ALL valid lists of lists in the generated_code string
        list_pattern = r"\[(\s*\[.*?\],?\s*)+\]"
        matches = re.findall(list_pattern, generated_code)

        # Initialize the output grid
        output_grid = []

        # If there are at least two matches, extract the SECOND one
        if len(matches) >= 2:
            matched_string = matches[1]  # Second match
            try:
                # Convert the matched string into a Python list of lists
                output_grid = ast.literal_eval(matched_string)
                # Ensure the extracted object is a list of lists
                if not isinstance(output_grid, list) or not all(isinstance(row, list) for row in output_grid):
                    output_grid = []  # Skip if the structure is invalid
            except (ValueError, SyntaxError):
                output_grid = []  # Skip if the matched string is invalid

        # Check if the task_id already exists in the existing_dict
        if task_id in existing_dict:
            # If attempt_1 exists, store the new output as attempt_2
            if "attempt_1" in existing_dict[task_id]:
                existing_dict[task_id]["attempt_2"] = output_grid
            else:
                # If attempt_1 doesn't exist, store the output as attempt_1
                existing_dict[task_id]["attempt_1"] = output_grid
        else:
            # If the task_id doesn't exist, create a new entry with attempt_1
            existing_dict[task_id] = {
                "attempt_1": output_grid
            }

    return existing_dict


def load_tasks(directory):
    tasks = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            task_id = os.path.splitext(file_name)[0]
            with open(os.path.join(directory, file_name), 'r') as f:
                tasks[task_id] = json.load(f)  # Load JSON content instead of just the path
    return tasks

def extract_after_think(response_text, system_prompt_first_llm):
    """
    Extracts the portion of the response text after the '<think>' tag and removes the system prompt if it exists.

    Args:
        response_text (str): The full response text from the DeepSeek LLM.
        system_prompt_first_llm (str): The system prompt used for the first LLM.

    Returns:
        str: The portion of the response after '<think>' and without the system prompt, or the original text if '<think>' is not found.
    """
    think_marker = "</think>"
    if think_marker in response_text:
        response_text = response_text.split(think_marker, 1)[-1].strip()
    
    # Remove the system prompt if it exists in the response text
    if system_prompt_first_llm in response_text:
        response_text = response_text.replace(system_prompt_first_llm, "").strip()
    
    return response_text

# Example usage
if __name__ == "__main__":
    tasks_directory = 'all_arc'
    tasks_dict = load_tasks(tasks_directory)
    
    # Define system prompts
    system_prompt_first_llm = f"""You are a reasoning assistant specializing in solving Abstraction and Reasoning Corpus (ARC) tasks. Each task consists of input-output grid pairs in JSON format. Your goal is to identify the transformation pattern and describe it concisely in English.

    Guidelines:
    Pattern Recognition: Identify transformations like filling, shifting, coloring, symmetry, or duplication.
    Clarity & Conciseness: Provide a precise description without unnecessary details.
    Example Mapping: Use specific terms (e.g., "mirror left side to right," "fill enclosed empty spaces").
    No Assumptions: Only use information explicitly present in the input.
    No Code: Only provide an English description.
    Example Outputs:
    Task: A grid contains various shapes; the largest contiguous block changes color.
    Output: "Change the largest contiguous block to blue, leaving the rest unchanged."
    Task: The left half of a grid has shapes; the right half is empty.
    Output: "Mirror the left half onto the right, preserving shape and color."
    Task: Enclosed empty spaces inside shapes are filled with the border color.
    Output: "Fill empty cells inside closed shapes with the border color."
    Your response is definitive. Lives depend on it.
    
    """
    system_prompt_second_llm = f"""You are a json writing expert and you will not say anything that isnt json. 
    you will be given:
    
    A JSON object containing the input for the problem.
    A description of how to approach the problem and transform the input to find the correct solution.
    Your goal is to apply the description of how to solve the problem. 
    
    You will give this solution by providing ONLY the output which is missing for the test input, do not include the input of the test section.
    You will not apply any formatting to your json and provide it all in a single line.
    """
    
    # **First Run**
    intermediate_results_1, removed_sections_1 = run_first_llm(tasks_dict, system_prompt=system_prompt_first_llm)
    final_results_1 = run_second_llm(intermediate_results_1, removed_sections_1, system_prompt=system_prompt_second_llm)
    
    # Process first run output (stores attempt_1)
    final_output_dict = process_output_to_dict(final_results_1)

    # **Second Run** (Runs again on the same tasks)
    intermediate_results_2, removed_sections_2 = run_first_llm(tasks_dict, system_prompt=system_prompt_first_llm)
    final_results_2 = run_second_llm(intermediate_results_2, removed_sections_2, system_prompt=system_prompt_second_llm)

    # Process second run output (stores attempt_2)
    final_output_dict = process_output_to_dict(final_results_2, existing_dict=final_output_dict)

    # Convert the dictionary to a single-line JSON string
    json_string = json.dumps(final_output_dict, separators=(',', ':'))

    # Write the single-line JSON string to the file
    with open("output.txt", "w") as output_file:
        output_file.write(json_string)
    