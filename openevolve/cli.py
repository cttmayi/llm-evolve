"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("problem", help="Path to the initial program directory", default=None)

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument("--iterations", "-i", help="Maximum number of iterations", type=int, default=None)

    parser.add_argument("--target-score", "-t", help="Target score to reach", type=float, default=None)

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    return parser.parse_args()


def get_config_overrides(args: argparse.Namespace) -> Dict:
    """Get configuration overrides from command-line arguments"""
    overrides = {}
    if args.api_base:
        overrides["llm.api_base"] = args.api_base
    if args.primary_model:
        overrides["llm.primary_model"] = args.primary_model
    if args.secondary_model:
        overrides["llm.secondary_model"] = args.secondary_model
    if args.iterations is not None:
        overrides["evolution.iterations"] = args.iterations
    if args.target_score is not None:
        overrides["evolution.target_score"] = args.target_score
    return overrides



def get_initial_program_path(problem_path: str) -> Optional[str]:
    """Get the path to the initial program file"""
    # 任意目录下的 initial_program.* 文件
    for filename in os.listdir(problem_path):
        if filename.startswith("initial_program."):
            return os.path.join(problem_path, filename)
    return None

def get_evaluation_file_path(problem_path: str) -> Optional[str]:
    """Get the path to the evaluation file"""
    eval_path = os.path.join(problem_path, "evaluator.py")
    if os.path.exists(eval_path):
        return eval_path
    return None

def get_config_file_path(problem_path: str) -> Optional[str]:
    """Get the path to the configuration file"""
    config_path = os.path.join(problem_path, "config.yaml")
    if os.path.exists(config_path):
        return config_path
    return None

async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    problem = args.problem

    config_file = args.config
    if config_file is None:
        config_file = get_config_file_path(problem)

    # Load base config from file or defaults
    config = load_config(config_file)

    initial_program = get_initial_program_path(problem)

    if initial_program is None:
        print(f"Error: Initial program file not found")
        return 1

    evaluation_file = get_evaluation_file_path(problem)
    if evaluation_file is None:
        print(f"Error: Evaluation file not found")
        return 1

    # Initialize OpenEvolve
    try:
        print("************* OpenEvolve Starting ************")
        print(f"Loading configuration from: {config_file if config_file else 'defaults'}")
        print(f"Initial program: {initial_program}")
        print(f"Evaluation file: {evaluation_file}")

        openevolve = OpenEvolve(
            initial_program_path=initial_program,
            evaluation_file=evaluation_file,
            config=config,
            # config_path=config if config is None else None,
            output_dir=args.output,
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            openevolve.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {openevolve.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
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
