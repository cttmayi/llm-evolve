"""
Command-line interface
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from llm_evolve import llmEvolve
from llm_evolve.config import Config, load_config

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evolve - Evolutionary coding agent")

    parser.add_argument("problem", help="Path to the initial program directory", default=None)

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument("--iterations", "-i", help="Maximum number of iterations", type=int, default=None)

    parser.add_argument("--target-score", "-t", help="Target score to reach", type=float, default=None)

    parser.add_argument(
        "--log-level", "-l", help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=None,
    )

    parser.add_argument(
        "--checkpoint", "-p", help="Path to checkpoint directory to resume from (e.g., evolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    return parser.parse_args()


def _get_initial_program_path(problem_path: str) -> Optional[str]:
    """Get the path to the initial program file"""
    # 任意目录下的 initial_program.* 文件
    for filename in os.listdir(problem_path):
        if filename.startswith("initial_program."):
            return os.path.join(problem_path, filename)
    return None

def _get_evaluation_file_path(problem_path: str) -> Optional[str]:
    """Get the path to the evaluation file"""
    eval_path = os.path.join(problem_path, "evaluator.py")
    if os.path.exists(eval_path):
        return eval_path
    return None

def _get_config_file_path(problem_path: str) -> Optional[str]:
    """Get the path to the configuration file"""
    config_path = os.path.join(problem_path, "config.yaml")
    if os.path.exists(config_path):
        return config_path
    return None

def _get_prompt_dir_path(problem_path: str) -> Optional[str]:
    """Get the path to the prompt directory"""
    prompt_path = os.path.join(problem_path, "prompts")
    if os.path.exists(prompt_path) and os.path.isdir(prompt_path):
        return prompt_path
    return None
    

async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = _parse_args()

    args_problem = args.problem

    args_config = args.config
    if args_config is None:
        args_config = _get_config_file_path(args_problem)

    if args_config is None:
        print(f"Error: Configuration file not found")
        return 1

    # Load base config from file or defaults
    config = load_config(args_config)

    args_initial_program = _get_initial_program_path(args_problem)
    args_evaluation_file = _get_evaluation_file_path(args_problem)

    args_log_level = args.log_level
    args_iterations = args.iterations
    args_target_score = args.target_score
    args_output = args.output
    args_checkpoint = args.checkpoint

    #override config with command-line arguments
    config.log_level = args_log_level or config.log_level
    config.max_iterations = args_iterations or config.max_iterations
    config.prompt.template_dir = config.prompt.template_dir or _get_prompt_dir_path(args_problem)

    if args_initial_program is None:
        print(f"Error: Initial program file not found")
        return 1

    if args_evaluation_file is None:
        print(f"Error: Evaluation file not found")
        return 1

    if args_checkpoint:
        if not os.path.exists(args_checkpoint):
            print(f"Error: Checkpoint directory '{args_checkpoint}' not found")
            return 1

    # Initialize Evolve
    try:
        print("************* Evolve Starting ************")
        print(f"Loading configuration from: {args_config}")
        print(f"Initial program: {args_initial_program}")
        print(f"Evaluation file: {args_evaluation_file}")

        evolve = llmEvolve(
            initial_program_path=args_initial_program,
            evaluation_file=args_evaluation_file,
            config=config,
            output_dir=args_output,
        )

        # Load from checkpoint if specified
        if args_checkpoint:
            print(f"Loading checkpoint from {args_checkpoint}")
            evolve.database.load(args_checkpoint)
            print(f"Checkpoint loaded successfully (iteration {evolve.database.last_iteration})")    

        # Run evolution
        best_program = await evolve.run(
            # iterations=args_iterations,
            target_score=args_target_score,
            checkpoint_path=args_checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(evolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        if best_program:
            print(f"\nEvolution complete!")
            print(f"Best program metrics:")
            for name, value in best_program.metrics.items():
                # Handle mixed types: format numbers as floats, others as strings
                if isinstance(value, (int, float)):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
