from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _emit_startup_failure(exc: Exception) -> int:
    error = str(exc).replace("\n", " ") or exc.__class__.__name__
    print("[START] task=startup env=rl_finance model=unknown", flush=True)
    print(
        f"[STEP] step=0 reward=0.00 action=StartupError done=true error={error}",
        flush=True,
    )
    print("[END] task=startup score=0.00 steps=0 success=false rewards=0.00", flush=True)
    return 0


def main() -> int:
    try:
        from rl_finance.inference import main as rl_main
    except Exception as exc:  # pragma: no cover - import failures are environment-specific
        return _emit_startup_failure(exc)
    return rl_main()


if __name__ == "__main__":
    raise SystemExit(main())
