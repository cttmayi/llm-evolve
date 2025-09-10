"""
Evaluator for detecting error
"""
import os
import numpy as np
import json

from llm_evolve.helper.executor import run_python_with_timeout, import_python_program, check_result_type
from llm_evolve.helper.result import create_exception_result, create_eval_result, print_result
from llm_evolve.helper.executor import ExecutorException


def get_values_from_file(header_file, data_file, true_value=True):
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

def create_result(combined_score, true_score, false_score, error_artifacts={}):    
    return create_eval_result(
        combined_score,
        artifacts=error_artifacts,
        true_score=true_score,
        false_score=false_score
    )

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
        program = import_python_program(program_path, 'search_algorithm')

        total_values = TOTAL_VALUES
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
            # Run with timeout
            result = run_python_with_timeout(program.search_algorithm, kwargs={'input_data': input_data}, timeout_seconds=5) # type: ignore
            check_result_type('search_algorithm', result, bool)

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

        # Add artifacts for successful runs
        artifacts = {"convergence_info": f"Converged in {num_trials} trials with {success_true_count+ success_false_count} successes",}
        if len(true_values_error) > 0:
            artifacts["true_value_errors"] = '; '.join(true_values_error[-10:])  # Store only the last 10 errors

        return create_result(
            (success_true_count+1)/(true_count+1) * (success_false_count+1)/(false_count+1) * 100, 
            success_true_count/true_count * 100, success_false_count/false_count * 100, 
            artifacts
        )
    except ExecutorException as e:
        return create_result(0.0, 0.0, 0.0, e.artifacts)
    except Exception as e:
        return create_exception_result(e)


if __name__ == "__main__":
    # Test the evaluator
    if True:
        program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
        result = evaluate(program_path)
        print_result(result)

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