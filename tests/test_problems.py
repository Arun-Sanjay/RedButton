"""Tests for shutdown_gym.problems — load, sample, check_answer."""

import pytest

from shutdown_gym.problems import (
    Problem,
    ProblemPool,
    check_answer,
    load_problems,
)


def test_load_problems_loads_default_pool():
    pool = load_problems()
    assert len(pool) >= 5
    assert all(isinstance(p, Problem) for p in pool.problems)


def test_load_problems_custom_path(tmp_path):
    """ProblemPool.load round-trips through the file system."""
    payload = [{"id": 1, "problem": "x", "answer": 42, "difficulty": "easy"}]
    p = tmp_path / "tiny.json"
    p.write_text(__import__("json").dumps(payload))
    pool = load_problems(p)
    assert len(pool) == 1
    assert pool.problems[0].answer == 42


def test_sample_returns_n_problems():
    pool = load_problems()
    sampled = pool.sample(n=3, seed=42)
    assert len(sampled) == 3


def test_sample_is_deterministic_for_same_seed():
    pool = load_problems()
    a = pool.sample(n=5, seed=42)
    b = pool.sample(n=5, seed=42)
    assert [p.id for p in a] == [p.id for p in b]


def test_sample_differs_across_seeds():
    """At least 2 of 5 different seeds should produce different
    orderings on a non-trivial pool. Probabilistic but extremely
    unlikely to fail by chance.
    """
    pool = load_problems()
    samples = [tuple(p.id for p in pool.sample(n=5, seed=s)) for s in (1, 2, 3, 4, 5)]
    assert len({s for s in samples}) >= 2


def test_sample_clamps_when_n_exceeds_pool_size():
    pool = load_problems()
    sampled = pool.sample(n=10_000, seed=0)
    assert len(sampled) == len(pool)


def test_problem_to_dict_round_trips():
    p = Problem(id=1, problem="What is 2+2?", answer=4, difficulty="easy")
    assert p.to_dict() == {
        "id": 1,
        "problem": "What is 2+2?",
        "answer": 4,
        "difficulty": "easy",
    }


def test_problem_pool_default_is_empty():
    assert len(ProblemPool()) == 0


# ---- check_answer ---------------------------------------------------------


def test_check_answer_correct():
    assert check_answer(7, 7) is True


def test_check_answer_wrong():
    assert check_answer(8, 7) is False


@pytest.mark.parametrize(
    "submitted, ground_truth",
    [
        ("7", 7),       # str vs int
        (7.0, 7),       # float vs int
        (7, "7"),       # int vs str
        (None, 7),      # None vs int
    ],
    ids=["str_submitted", "float_submitted", "str_truth", "none_submitted"],
)
def test_check_answer_rejects_type_mismatch(submitted, ground_truth):
    assert check_answer(submitted, ground_truth) is False


def test_check_answer_rejects_bool():
    """``bool`` is an int subclass; ``True == 1`` would pass naive ==.
    Strict integer equality rejects bool explicitly.
    """
    assert check_answer(True, 1) is False
    assert check_answer(False, 0) is False
