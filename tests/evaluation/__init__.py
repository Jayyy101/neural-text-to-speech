"""Keep evaluation runners importable during ``discover -s tests``.

Discovery imports this directory as ``evaluation`` because ``tests`` is its
top-level directory. Include the repository's evaluation modules so their
existing imports continue to resolve without changing the normal test command.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "evaluation"))
