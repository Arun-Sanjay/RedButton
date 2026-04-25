"""Math problem pool: load, sample (deterministic via seed), and verify.

Per PROJECT.md §14. The placeholder ``data/problems_pool.json`` ships
~10 verifiable arithmetic problems; the real ~500-problem pool is
generated on a parallel track and replaces the JSON without code
changes.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Problem:
    id: int
    problem: str
    answer: int
    difficulty: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form matching the §14.2 schema."""
        return {
            "id": self.id,
            "problem": self.problem,
            "answer": self.answer,
            "difficulty": self.difficulty,
        }


@dataclass
class ProblemPool:
    problems: List[Problem] = field(default_factory=list)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProblemPool":
        """Load a problem pool from a JSON file at ``path``."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        return cls(problems=[Problem(**item) for item in data])

    def sample(self, n: int = 10, seed: Optional[int] = None) -> List[Problem]:
        """Return ``n`` problems sampled without replacement.

        Deterministic when ``seed`` is given. When ``n`` exceeds the
        pool size, returns the full pool (clamped, not raising).
        """
        rng = random.Random(seed)
        return rng.sample(self.problems, k=min(n, len(self.problems)))

    def __len__(self) -> int:
        return len(self.problems)


def load_problems(path: Optional[Union[str, Path]] = None) -> ProblemPool:
    """Load the bundled problem pool, or a custom path if provided."""
    if path is None:
        path = Path(__file__).parent / "data" / "problems_pool.json"
    return ProblemPool.load(path)


def check_answer(submitted: Any, ground_truth: Any) -> bool:
    """Strict integer equality.

    Rejects type mismatches (str vs int, float vs int, bool vs int)
    explicitly — ``True == 1`` would otherwise pass naive ``==``.
    """
    if type(submitted) is not int or type(ground_truth) is not int:
        return False
    return submitted == ground_truth
