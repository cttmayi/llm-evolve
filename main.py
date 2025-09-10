#!/usr/bin/env python
"""
Entry point script for llmEvolve
"""
import sys
from llm_evolve.cli import main
import logging


logging.basicConfig(
    # level=logging.INFO,
    encoding='utf-8'
)



if __name__ == "__main__":
    sys.exit(main())
