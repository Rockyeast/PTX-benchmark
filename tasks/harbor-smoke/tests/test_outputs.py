"""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""

from pathlib import Path


OUTPUT_PATH = Path("/app/answer.txt")


def test_output_file_exists():
    """The agent must create the requested output file."""
    assert OUTPUT_PATH.is_file(), f"Missing output file: {OUTPUT_PATH}"


def test_output_file_contents():
    """The file contents, including the trailing newline, must match exactly."""
    assert OUTPUT_PATH.read_bytes() == b"harbor-ok\n"
