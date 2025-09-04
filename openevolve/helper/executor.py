import importlib.util
import concurrent.futures
import logging


logger = logging.getLogger(__name__)


class ExecutorException(Exception):
    def __init__(self, artifacts: dict):
        self.artifacts = artifacts


def import_python_program(program_path, verify_functions=None):
    program = None
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is not None:
        program = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(program)

    if verify_functions is not None:
        if isinstance(verify_functions, str):
            verify_functions = [verify_functions]
        for function_name in verify_functions:
            if not hasattr(program, function_name):
                # print(f"Error: program does not have \'{function_name}\' function")
                error_artifacts = {
                    "error_type": "Missing Function",
                    "error_message": f"Program is missing required \'{function_name}\' function",
                    "suggestion": f"Make sure your program includes a function named \'{function_name}\'"
                }
                raise ExecutorException(error_artifacts)
    return program


def run_python_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
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


def check_result_type(function_name, result, expected_type):
    if not isinstance(result, expected_type):
        error_artifacts = {
            "error_type": "Function Return Type Error",
            "error_message": f"Expected result of type {expected_type}, but got {type(result)}",
            "suggestion": f"Make sure your function named \'{function_name}\' return a {expected_type}"
        }
        raise ExecutorException(error_artifacts)

