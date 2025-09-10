import traceback
import logging

from ..evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)

class ExecutorException(Exception):
    def __init__(self, artifacts: dict):
        self.artifacts = artifacts


def create_eval_result(combined_score, artifacts={}, **kwargs):
    metrics={"combined_score": combined_score}
    metrics.update(kwargs)
    logger.debug(f"Metrics: {metrics}")
    logger.debug(f"Artifacts: {artifacts}")
    return EvaluationResult(
        metrics=metrics,
        artifacts=artifacts
    )


def create_exception_result(error:Exception, suggestion_message=None):
    full_traceback = traceback.format_exc()
    metrics = {"combined_score": 0.0}
    error_artifacts = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "full_traceback": str(full_traceback)
    }
    if suggestion_message is not None:
        error_artifacts["suggestion"] = str(suggestion_message)

    logger.error(f"Error: {error_artifacts['error_message']}")
    return EvaluationResult(
        metrics=metrics,
        artifacts=error_artifacts # type: ignore
    )


def print_result(result: EvaluationResult, limit=400):
    print("Metrics:")
    for key, value in result.metrics.items():
        print(f"    {key:<20}: {value:.4f}")
    print("Artifacts:")
    for key, value in result.artifacts.items():
        if len(str(value)) > limit:
            print(f"    {key:<20}: {str(value)[:limit]} ... ...")
        else:
            print(f"    {key:<20}: {str(value)}")