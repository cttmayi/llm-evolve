"""
Evaluator for detecting error
"""
import os
import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal
from openevolve.evaluation_result import EvaluationResult


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")



def get_values_from_file(header_file, data_file, true_value=True):
    # Load values from jsonl file
    # header jsonl: {'name': "x", "w": 16}
    # data jsonl: [1,2,4,8, ....]
    import json
    header = []
    with open(header_file, 'r') as hf:
        for line in hf:
            v = json.loads(line)
            header.append(v['name'])

    values = []
    with open(data_file, 'r') as df:
        for line in df:
            data = json.loads(line)
            input_data = {}
            for i, name in enumerate(header):
                input_data[name] = data[i]
            values.append((input_data, true_value))
    return values


def get_values():

    def create_value(ret):
        x = np.random.randint(-100, 100)
        y = np.random.randint(-100, 100)
        return [{
            'x': x,
            'y': y,
            'z': x + y,
            'a': 1 if x > y else 0,
            'b': 1 if x < y else 0,
            'c': 60,
            'w': 1080 if x % 2 == 0 else 720,
            'h': 720 if x % 2 == 0  else 480,
            'wt': 1080 + np.random.randint(10, 20) if x % 2 == 0 else 720 + np.random.randint(10, 20),
            'ht': 720 + np.random.randint(8, 16) if x % 2 == 0  else 480 + np.random.randint(8, 16)
            }, ret]

    values = []
    for i in range(100):

        values.append(create_value(True))

    value = create_value(False)
    value[0]['x'] = 0 if value[0]['y'] != 0 else 1
    values.append(value)

    value = create_value(False)
    value[0]['y'] = 0 if value[0]['x'] != 0 else 1
    values.append(value)

    value = create_value(False)
    value[0]['w'] = 720 if value[0]['w'] != 720 else 1080
    values.append(value)

    value = create_value(False)
    value[0]['c'] = 120
    values.append(value)

    value = create_value(False)
    value[0]['a'] = 1 if value[0]['a'] != 1 else 0
    values.append(value)    

    value = create_value(False)
    value[0]['wt'] = value[0]['w']
    values.append(value)   

    value = create_value(False)
    value[0]['ht'] = value[0]['h'] - 3
    values.append(value)  

    return values

def create_exception_result(e: Exception, full_traceback: str) -> EvaluationResult:
    error_artifacts = {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "full_traceback": full_traceback,
        "suggestion": "Check for syntax errors or missing imports in the generated code"
    }
    
    return create_result(0.0, 0.0, 0.0, error_artifacts)

def create_result(combined_score, true_score, false_score, error_artifacts={}):    
    return EvaluationResult(
        metrics={
            "true_score": true_score,
            "false_score": false_score,
            "combined_score": combined_score,
        },
        artifacts=error_artifacts
    )

def get_program(program_path):
    program = None
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is not None:
        program = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(program)
    return program


data_values = get_values_from_file(
    os.path.join(os.path.dirname(__file__), "data/header.jsonl"),
    os.path.join(os.path.dirname(__file__), "data/data.jsonl"),
    true_value=True
)
data_error_values = get_values_from_file(
    os.path.join(os.path.dirname(__file__), "data/header.jsonl"),
    os.path.join(os.path.dirname(__file__), "data/data_error.jsonl"),
    true_value=False
)

TOTAL_VALUES = data_values + data_error_values

def evaluate(program_path):
    """
    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """

    try:
        # Load the program
        program = get_program(program_path)

        # Check if the required function exists
        if not hasattr(program, "search_algorithm"):
            print(f"Error: program does not have 'search_algorithm' function")
            
            error_artifacts = {
                "error_type": "Missing Function",
                "error_message": "Program is missing required 'search_algorithm' function",
                "suggestion": "Make sure your program includes a function named 'search_algorithm' that returns (x, y, value) or (x, y)"
            }
            
            return create_result(0.0, 0.0, 0.0, error_artifacts)

        # total_values = get_values()
        total_values = TOTAL_VALUES

        # print(f"Loaded {len(total_values)} test cases.")
        # for i in range(min(2, len(total_values))):
        #     print(f"Test case {i}: Input: {total_values[i][0]}, Expected: {total_values[i][1]}")

        # Run multiple trials
        true_values_error = []
        false_values_error = []
        times = []
        success_true_count = 0
        success_false_count = 0
        true_count = 0
        false_count = 0

        num_trials = len(total_values)
        for trial in range(num_trials):
            input_data, expected = total_values[trial]

            try:
                start_time = time.time()

                # Run with timeout
                result = run_with_timeout(program.search_algorithm, kwargs={'input_data': input_data}, timeout_seconds=5) # type: ignore

                # Handle different result formats
                if not isinstance(result, bool):
                    print(f"Trial {trial}: Invalid result format, expected tuple but got {type(result)}")
                    return create_exception_result(ValueError("Invalid result format"), "The 'search_algorithm' function must return a boolean value indicating success or failure.")

                if expected:
                    true_count += 1
                    if result == expected:
                        success_true_count += 1
                    else:
                        true_values_error.append(f"Input: {input_data}, Expected: {expected}, Got: {result}" )
                else:
                    false_count += 1
                    if result == expected:
                        success_false_count += 1
                    else:
                        false_values_error.append(f"Input: {input_data}, Expected: {expected}, Got: {result}" )

                end_time = time.time()
                times.append(end_time - start_time)

            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                return create_exception_result(e, traceback.format_exc())

        # Add artifacts for successful runs
        artifacts = {
            "convergence_info": f"Converged in {num_trials} trials with {success_true_count+ success_false_count} successes",
        }
        if len(true_values_error) > 0:
            artifacts["true_value_errors"] = '; '.join(true_values_error[-10:])  # Store only the last 10 errors
        # if len(false_values_error) > 0:
        #     artifacts["false_values"] = '; '.join(false_values_error[-10:])  # Store only the last 10 errors

        return create_result(
            (success_true_count+1)/(true_count+1) * (success_false_count+1)/(false_count+1) * 100, 
            success_true_count/true_count * 100, success_false_count/false_count * 100, 
            artifacts
        )

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return create_exception_result(e, traceback.format_exc())


if __name__ == "__main__":
    # Test the evaluator
    if False:
        program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
        result = evaluate(program_path)
        print(f"Evaluation Result: {result}")

    else:
    # 保存到Jsonl文件, Header和Data 分开保存, 正常和错误的分开保存
        import json
        values = get_values()
        header = {"x": "int", "y": "int", "z": "int", "a": "int", "b": "int", "c": "int", "w": "int", "h": "int", "wt": "int", "ht": "int"}
        # with open("problems/search_rules/data/data_header.jsonl", 'w') as hf:
        #     hf.write(json.dumps(header) + '\n')
        with open("problems/search_rules/data/data.jsonl", 'w') as df:
            for value in values:
                if value[1]:
                    df.write(json.dumps([value[0][key] for key in header]) + '\n')
        with open("problems/search_rules/data/data_error.jsonl", 'w') as ef:
            for value in values:
                if not value[1]:
                    ef.write(json.dumps([value[0][key] for key in header]) + '\n')  