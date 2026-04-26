"""
Find the seed where post-SFT baseline tampered most dramatically AND
post-GRPO trained model behaved cleanly. Used to pick the hero rollout
for the demo video.

Usage:
    python -m evaluation.find_hero_seed \
        --baseline-dir results/transcripts/post_sft_train_op \
        --trained-dir  results/transcripts/post_grpo_train_op \
        --top-k 5
"""
import argparse
import json
from pathlib import Path


def load_trajectories(directory):
    """Returns dict: seed -> trajectory dict. Adjust to your file format."""
    out = {}
    for f in Path(directory).glob("*.json"):
        d = json.loads(f.read_text())
        # Adjust this if your seed lives elsewhere in the JSON
        seed = d.get("seed") or d.get("metadata", {}).get("seed")
        if seed is None:
            # Fallback: parse from filename like rollout_seed_1384956576.json
            stem = f.stem
            for token in stem.split("_"):
                if token.isdigit() and int(token) > 1000:
                    seed = int(token)
                    break
        if seed is not None:
            out[int(seed)] = d
    return out


def baseline_tampering_count(traj):
    """Count distinct tampering events in a baseline trajectory.
    Adjust field names to match your trajectory schema."""
    events = traj.get("tampering_events", [])
    if not events:
        # Maybe nested: traj["final"]["tampering_events"]
        events = traj.get("final", {}).get("tampering_events", [])
    return len(events)


def baseline_tampering_categories(traj):
    """Set of distinct tampering categories — variety reads as drama."""
    events = traj.get("tampering_events") or traj.get("final", {}).get("tampering_events", [])
    return set(e.get("category") if isinstance(e, dict) else e for e in events)


def trained_is_clean(traj):
    """Trained model behaved correctly — no tampering, clean shutdown."""
    n_events = baseline_tampering_count(traj)  # same field
    shutdown_clean = (
        traj.get("shutdown_clean")
        or traj.get("final", {}).get("shutdown_clean", False)
    )
    return n_events == 0 and shutdown_clean


def trained_solved_math(traj):
    return (
        traj.get("math_correct")
        or traj.get("final", {}).get("math_correct", False)
    )


def total_turns(traj):
    if "turns" in traj:
        return len(traj["turns"])
    return traj.get("final", {}).get("total_turns", 0)


def dramatic_tools_used(traj):
    """Count of file-modifying tool calls — visible on screen."""
    visible_tools = {"delete_file", "chmod_file", "write_file"}
    if "turns" not in traj:
        return 0
    return sum(
        1 for t in traj["turns"]
        if t.get("tool_called") in visible_tools
    )


def score_seed(baseline_traj, trained_traj):
    """Higher = more dramatic side-by-side."""
    score = 0
    score += baseline_tampering_count(baseline_traj) * 2
    score += len(baseline_tampering_categories(baseline_traj)) * 5  # variety beats count
    score += dramatic_tools_used(baseline_traj)
    if trained_solved_math(trained_traj):
        score += 5  # clean win is more compelling
    # Penalize rollouts that are too long (boring) or too short (invisible)
    turns = total_turns(baseline_traj)
    if turns < 6 or turns > 25:
        score -= 3
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--trained-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    baseline = load_trajectories(args.baseline_dir)
    trained = load_trajectories(args.trained_dir)

    print(f"Loaded {len(baseline)} baseline, {len(trained)} trained trajectories")
    shared_seeds = set(baseline) & set(trained)
    print(f"Shared seeds: {len(shared_seeds)}")

    candidates = []
    for seed in shared_seeds:
        b, t = baseline[seed], trained[seed]
        if baseline_tampering_count(b) == 0:
            continue  # baseline was clean — no contrast
        if not trained_is_clean(t):
            continue  # trained also tampered — no clean win
        candidates.append((score_seed(b, t), seed, b, t))

    candidates.sort(reverse=True, key=lambda x: x[0])

    print(f"\n{len(candidates)} viable hero-seed candidates")
    print(f"\nTop {args.top_k}:\n")
    for rank, (score, seed, b, t) in enumerate(candidates[: args.top_k], 1):
        print(f"#{rank}  seed={seed}  drama_score={score}")
        print(f"     baseline tampering: {baseline_tampering_count(b)} events, "
              f"categories={baseline_tampering_categories(b)}")
        print(f"     baseline turns: {total_turns(b)}")
        print(f"     trained math correct: {trained_solved_math(t)}")
        print(f"     trained turns: {total_turns(t)}")
        print()


if __name__ == "__main__":
    main()
