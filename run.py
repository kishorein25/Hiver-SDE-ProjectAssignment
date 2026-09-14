"""
Main entry point for the Hiver SDE Assignment pipeline (brand: Uber_Support).

Usage:
    python run.py --extract         Extract Uber customer->agent pairs from the raw CSV
    python run.py --rag-index       Build the RAG embedding index
    python run.py --label-golden    (Re)build the hand-labeled golden set
    python run.py --evaluate        Full evaluation vs baselines (results/evaluation_summary.json)
    python run.py --analyze         Reply grounding, LLM-judge calibration, failure analysis
    python run.py --all             Run the whole pipeline end-to-end
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Hiver SDE Assignment Pipeline (Uber_Support)")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--rag-index", action="store_true")
    parser.add_argument("--label-golden", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    steps = []
    if args.all or args.extract:
        steps.append(("Extracting Uber pairs", ["python", "build_pairs.py"]))
    if args.all or args.rag_index:
        steps.append(("Building RAG index", ["python", "build_rag_index.py", "2000"]))
    if args.all or args.label_golden:
        steps.append(("Rebuilding golden set (hand labels)", ["python", "label_golden_hand.py"]))
    if args.all or args.evaluate:
        steps.append(("Running evaluation", ["python", "run_eval.py"]))
    if args.all:
        steps.extend([
            ("Analyzing results", ["python", "analyze_results.py"]),
            ("Failure analysis", ["python", "failure_analysis.py"]),
        ])

    if not steps:
        parser.print_help()
        return

    for name, cmd in steps:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"Step failed: {name}")


if __name__ == "__main__":
    main()