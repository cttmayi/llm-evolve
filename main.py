#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import sys
from openevolve.cli import main
import logging


logging.basicConfig(
    # level=logging.INFO,
    encoding='utf-8'
)



if __name__ == "__main__":
    sys.exit(main())
