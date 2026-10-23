import os
import sys

# Allow `python -m bench` from the backend directory or `python backend/bench`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
