"""发布回滚演练 CLI（argv-only，不发起外部调用）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.devmate.release_kit import ReleaseCandidate, RollbackDrill


def main() -> int:
    parser = argparse.ArgumentParser(description="devmate release rollback drill")
    parser.add_argument("--candidate-id", default="rel-1")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--target-commit", required=True)
    parser.add_argument("--rollback-commit", default="")
    parser.add_argument("--steps", nargs="*", default=("migrate", "deploy"))
    args = parser.parse_args()

    candidate = ReleaseCandidate(
        candidate_id=args.candidate_id,
        version=args.version,
        target_commit=args.target_commit,
        rollback_commit=args.rollback_commit,
        steps=tuple(args.steps),
    )
    result = RollbackDrill().run(candidate)
    print(f"passed={result.passed} rolled_back={result.rolled_back}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
