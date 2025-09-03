import importlib.util
import concurrent.futures


def import_python_program(program_path):
    program = None
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is not None:
        program = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(program)
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


