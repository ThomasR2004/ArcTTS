import json
import os
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Paths to models
MODEL_1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_2 = "ibm-granite/granite-3.1-8b-instruct"

ADAPTER_PATH = "/gpfs/home6/trietman/ArcTTS/checkpoint-43/" # load finetune
# Load the first model
model_1, tokenizer_1 = FastLanguageModel.from_pretrained(
    model_name=MODEL_1,
    max_seq_length=16000,
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

    for task_id, task_data in tasks_dict.items():
        # Include the system prompt if provided
        prompt = f"{system_prompt}\n\nTask Data: {json.dumps(task_data)}"

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer_1.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(tokenizer_1)
        generated_tokens = model_1.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=20000, use_cache=True)

        # Decode the tensor output to a readable string
        output_text = tokenizer_1.decode(generated_tokens[0], skip_special_tokens=True)
        description = extract_after_think(output_text)

        output[task_id] = {"description": description, "original_json": task_data}

    return output


def run_second_llm(intermediate_results, system_prompt=None):
    final_output = {}

    for task_id, data in intermediate_results.items():
        description = data["description"]
        original_json = data["original_json"]
        

        # Ensure original_json is properly formatted
        if isinstance(original_json, str):
            original_json = json.loads(original_json)
            

        # Extract only the 'test' section
        modified_json = {key: value for key, value in original_json.items() if key == "test"}

        prompt = f"{system_prompt}\n\nDescription: {description}\n\nTask Data: {json.dumps(modified_json)}"

        # Fix message format for `apply_chat_template`
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer_2.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer_2)
        generated_tokens = model_2.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=1024, use_cache=True)

        # Decode the tensor output to a readable string
        generated_code = tokenizer_2.decode(generated_tokens[0], skip_special_tokens=True)

        # Extract JSON content using regex
        json_match = re.search(r'\{.*\}', generated_code, re.DOTALL)
        if json_match:
            generated_code = json_match.group(0)

        final_output[task_id] = {"generated_code": generated_code}

    return final_output


import json

def process_output_to_dict(final_results):
    """
    Processes the final results and stores them in a dictionary format with attempts.
    Extracts the "output" grid from the generated JSON, handling multiple cases.

    Args:
        final_results (dict): The final results from the second LLM.

    Returns:
        dict: A dictionary mapping task IDs to attempts, containing only the "output" grid.
    """
    task_dict = {}

    for task_id, result_data in final_results.items():
        output = result_data.get("generated_code")
        
        try:
            # Case 1: If the output is a JSON string, parse it
            if isinstance(output, str):
                output_json = json.loads(output)
            # Case 2: If the output is already a JSON object, use it directly
            elif isinstance(output, dict):
                output_json = output
            else:
                output_json = None

            # Extract the "output" grid
            if output_json:
                if "test" in output_json and isinstance(output_json["test"], list):
                    # Case 1: "test" key contains a list of items
                    for item in output_json["test"]:
                        if "output" in item:
                            output_grid = item["output"]
                            break
                    else:
                        output_grid = None
                elif "output" in output_json:
                    # Case 2: "output" key directly contains the grid
                    output_grid = output_json["output"]
                else:
                    output_grid = None
            else:
                output_grid = None
        except json.JSONDecodeError:
            output_grid = None

        # If task_id doesn't exist, initialize it with attempt_1
        if task_id not in task_dict:
            task_dict[task_id] = {"attempt_1": output_grid}
        else:
            # If task_id already exists, store the output as attempt_2
            if "attempt_1" in task_dict[task_id]:
                task_dict[task_id]["attempt_2"] = output_grid
            else:
                task_dict[task_id]["attempt_1"] = output_grid
                
    return task_dict

def load_tasks(directory):
    tasks = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            task_id = os.path.splitext(file_name)[0]
            with open(os.path.join(directory, file_name), 'r') as f:
                tasks[task_id] = json.load(f)  # Load JSON content instead of just the path
    return tasks

def extract_after_think(response_text):
    """
    Extracts the portion of the response text after the '<think>' tag.

    Args:
        response_text (str): The full response text from the DeepSeek LLM.

    Returns:
        str: The portion of the response after '<think>', or the original text if '<think>' is not found.
    """
    think_marker = "</think>"
    if think_marker in response_text:
        return response_text.split(think_marker, 1)[-1].strip()
    return response_text.strip()

# Example usage
if __name__ == "__main__":
    tasks_directory = 'onetask'
    tasks_dict = load_tasks(tasks_directory)
    
    # Define system prompts
    system_prompt_first_llm = f"""You are a reasoning assistant specializing in analyzing and solving Abstraction and Reasoning Corpus (ARC) tasks. Each ARC task is provided in a JSON format containing grids of input-output pairs, where the inputs are before-transformation states, and the outputs are after-transformation states. Your job is to examine the patterns and transformations between the input-output pairs and generate a concise English description of the process required to solve the task.

    Guidelines:
    Pattern Recognition: Focus on identifying patterns or rules applied to the input grids to produce the output grids. These may include transformations like filling, shifting, coloring, symmetry, duplication, etc.
    Conciseness: Your description should be as clear and concise as possible while fully explaining the solution process. Avoid unnecessary detail or repetition.
    Example Mapping: Use specific terms to describe the transformation (e.g., "replace all blue squares with yellow circles," "extend lines to form rectangles").
    Assumptions: Do not assume any information beyond what is explicitly presented in the JSON.
    Do not provide code about how to solve the problem. I only want a concise english description
    Example Output:
    Task: A grid contains several shapes of varying colors. In the output, the largest contiguous block of any color is replaced with blocks of a new color.
    Output Description: "Identify the largest contiguous block of connected cells of any color. Change the color of these cells to blue while leaving the rest of the grid unchanged."
    
    Task: A grid has shapes arranged in the left half, and the right half is empty. In the output, the left half is mirrored onto the right half.
    Output Description: "Copy the shapes from the left half of the grid and mirror them onto the right half, maintaining their orientation and color."
    
    Task: The input contains various closed shapes with some internal empty cells. In the output, all empty cells inside closed shapes are filled with the same color as the shape's border.
    Output Description: "For each closed shape, identify all internal empty cells and fill them with the color of the border. Ensure no external cells are affected."
    
    Task: The input grid contains lines of varying colors and lengths. In the output, all lines are extended horizontally until they reach the edge of the grid.
    Output Description: "Extend each horizontal line in the grid to the left and right edges while preserving the original color of the line. Do not alter vertical or diagonal lines."
    
    Task: Grids have isolated squares of varying colors, some marked with a smaller black dot inside. In the output, only squares with black dots are kept; others are removed.
    Output Description: "Remove all squares without black dots inside them. Retain the original color of squares with dots and preserve their positions."
    
    Input Format:
    You will receive JSON objects with input-output grids in a simplified format. Examine these grids to derive the solution pattern.
    
    Output Format:
    Provide a concise English description of the transformation needed to solve the ARC task. Ensure clarity and generality.
    
    This task is very important and a lot of lives depend on it, you dont ask if you did it well, you answer with certainty. You do not have to give any disclaimers.
    
    """
    system_prompt_second_llm = f"""You are a json writing expert and you will not say anything that isnt json. 
    you will be given:
    
    A JSON object containing the input for the problem.
    A description of how to approach the problem and transform the input to find the correct solution.
    Your goal is to apply the description of how to solve the problem. 
    
    You will give this solution by providing ONLY the output which is missing for the test input, do not include the input of the test section."
    """
    
    # Step 1: Process tasks with the first LLM
    intermediate_results = run_first_llm(tasks_dict, system_prompt=system_prompt_first_llm)
    
    # Step 2: Process intermediate results with the second LLM
    final_results = run_second_llm(intermediate_results, system_prompt=system_prompt_second_llm)
    
    # Step 3: Process final results into a dictionary format
    final_output_dict = process_output_to_dict(final_results)
    

    # Output the final results
    print(final_output_dict)

    
    
    output_file_path = "output.txt"
    with open(output_file_path, "w") as output_file:
        json.dump(final_output_dict, output_file, indent=4)
    
    print(f"Results written to {output_file_path}")