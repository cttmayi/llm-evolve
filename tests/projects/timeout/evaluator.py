"""
Evaluator for detecting error
"""
import os
import numpy as np
import json

from llm_evolve.helper.executor import run_python_with_timeout, import_python_program, check_result_type
from llm_evolve.helper.result import create_exception_result, create_eval_result, print_result
from llm_evolve.helper.executor import ExecutorException


def evaluate(program_path):
    """
    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """

    try:


        # Run the program
        program = import_python_program(program_path)

        # Check if the program has a search_algorithm function
        if not hasattr(program, "search_algorithm"):
            raise ExecutorException({"error": "Program does not have a search_algorithm function"})

        # Run the search_algorithm function
        import time
        start = time.time()
        result = program.search_algorithm()
        end = time.time()

        print("Done", result, end - start)
        return create_eval_result(end - start)
    except Exception as e:
        print("Error", e)
        return create_exception_result(e)


if __name__ == "__main__":
    # Test the evaluator
    program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")
    result = evaluate(program_path)
    # print_result(result)
