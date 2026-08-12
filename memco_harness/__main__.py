"""`uv run python -m memco_harness` is the same as the `memco-harness` script."""

from .cli import main

raise SystemExit(main())
