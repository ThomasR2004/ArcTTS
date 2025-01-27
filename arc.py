import json
import os
import openai
from openai import OpenAI


# Server and API key configuration
SERVER_1_URL = "http://localhost:8000/v1"
SERVER_1_API_KEY = "token-1"

SERVER_2_URL = "http://localhost:8001/v1"
SERVER_2_API_KEY = "token-2"

MODEL_1 = "deepseek-r1-distill-qwen-1.5b"
MODEL_2 = "granite-3.1-8b-instruct"


def run_first_llm(tasks_dict, system_prompt=None):
    """
    Process tasks using the first LLM and output the intermediate results.

    Args:
        tasks_dict (dict): A dictionary where keys are task IDs (strings) and values are file paths to JSON tasks.
        system_prompt (str): An optional system prompt to be used with the first LLM.

    Returns:
        dict: A dictionary mapping task IDs to intermediate results.
    """
    output = {}

    # Configure the OpenAI client for the first server
    openai.api_base = SERVER_1_URL
    openai.api_key = SERVER_1_API_KEY

    for task_id, task_file in tasks_dict.items():
        with open(task_file, 'r', encoding='utf-8') as file:
            task_data = json.load(file)

        # Include the system prompt if provided
        prompt = 'Name 10 locations that would be good for a skiing holiday but also for a swim'

        response = openai.completions.create(
            model=MODEL_1,
            prompt=prompt,
            temperature=0.5,
            max_tokens=10000
        )

        # Extract and store the generated result
        raw_result = response.choices[0].text.strip()
        clean_result = extract_after_think(raw_result)
        output[task_id] = {"description": clean_result, "original_json": task_data}

    return output


def run_second_llm(intermediate_results, system_prompt=None):
    """
    Process the output of the first LLM using a second LLM.

    Args:
        intermediate_results (dict): A dictionary where keys are task IDs and values are intermediate results.
        system_prompt (str): An optional system prompt to be used with the second LLM.

    Returns:
        dict: A dictionary mapping task IDs to final results.
    """
    final_output = {}

     # Configure the OpenAI client for the second server
    openai.api_base = SERVER_2_URL
    openai.api_key = SERVER_2_API_KEY

    for task_id, data in intermediate_results.items():
        description = data["description"]
        original_json = data["original_json"]

        # Only keep everything after and including 'test'
        modified_json = {key: value for key, value in original_json.items() if key == "test"}

        # Include the system prompt if provided
        prompt = f"{system_prompt}\n\n" if system_prompt else ""
        prompt += f"""
        Here is a JSON task: {json.dumps(modified_json)}.
        Based on this description of transformations: {description} 
        Return the modified JSON as the output, reflecting the solution to the task.
        """

        response = openai.completions.create(
            model=MODEL_2,
            prompt=prompt,
            temperature=0.4,
            max_tokens=1200
        )

        # Extract and store the final result
        result = response.choices[0].text.strip()
        final_output[task_id] = {"generated_code": result}

    return final_output

def process_output_to_dict(final_results):
    """
    Processes the final results and stores them in a dictionary format with attempts.

    Args:
        final_results (dict): The final results from the second LLM.

    Returns:
        dict: A dictionary mapping task IDs to attempts.
    """
    task_dict = {}

    for task_id, result_data in final_results.items():
        output = result_data.get("generated_code")
        
        # If task_id doesn't exist, initialize it with attempt_1
        if task_id not in task_dict:
            task_dict[task_id] = {"attempt_1": output}
        else:
            # If task_id already exists, store the output as attempt_2
            if "attempt_1" in task_dict[task_id]:
                task_dict[task_id]["attempt_2"] = output
            else:
                task_dict[task_id]["attempt_1"] = output
                
    return task_dict

def load_tasks(directory):
    """
    Load all JSON files from a directory and map them to task IDs.

    Args:
        directory (str): The path to the directory containing JSON task files.

    Returns:
        dict: A dictionary mapping task IDs to file paths.
    """
    tasks = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            task_id = os.path.splitext(file_name)[0]
            tasks[task_id] = os.path.join(directory, file_name)
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
    system_prompt_second_llm = system_prompt_2 = f"""You are a json writing expert and you will not say anything that isnt json. 
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
   # print(intermediate_results)