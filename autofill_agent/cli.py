"""
Command line tools for the CV knowledge layer.

    python -m autofill_agent.cli ingest path/to/cv.pdf [--user ID]
    python -m autofill_agent.cli profile [--user ID]
    python -m autofill_agent.cli search "python experience" [--user ID]
    python -m autofill_agent.cli remember "notice period?" "Two weeks" [--user ID]
    python -m autofill_agent.cli recall "notice period?" [--user ID]
"""

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_USER_ID
from .cv_ingest import describe_hit, ingest_cv, search_cv
from .storage import Storage


def cmd_ingest(args) -> int:
    print(f"Ingesting {args.pdf} for user '{args.user}' (first run downloads Docling models)...")
    result = ingest_cv(Path(args.pdf), user_id=args.user)
    name = result.profile.full_name or "(name not found)"
    print(f"Stored {result.chunk_count} chunks and a profile for {name}.")
    return 0


def cmd_profile(args) -> int:
    profile = Storage().get_profile(args.user)
    if profile is None:
        print(f"No profile for user '{args.user}'. Run the ingest command first.")
        return 1
    print(profile.model_dump_json(indent=2))
    return 0


def cmd_search(args) -> int:
    hits = search_cv(args.query, user_id=args.user, k=args.k, kind=args.kind)
    if not hits:
        print(f"No CV chunks for user '{args.user}'. Run the ingest command first.")
        return 1
    for i, hit in enumerate(hits, 1):
        score = "" if args.kind else f" (score {hit.score:.3f})"
        print(f"--- {i}{score} [{hit.section or 'no section'}]")
        print(describe_hit(hit))
        print()
    return 0


def cmd_remember(args) -> int:
    Storage().save_answer(args.user, args.question, args.answer)
    print("Saved.")
    return 0


def cmd_recall(args) -> int:
    answer = Storage().get_answer(args.user, args.question)
    if answer is None:
        print("No saved answer.")
        return 1
    print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autofill_agent.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_user(p):
        p.add_argument("--user", default=DEFAULT_USER_ID, help="User id (default from AUTOFILL_USER_ID)")

    p = sub.add_parser("ingest", help="Read a CV PDF into the local store")
    p.add_argument("pdf")
    add_user(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("profile", help="Print the stored structured profile")
    add_user(p)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("search", help="Search CV chunks")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--kind", choices=["experience", "education", "project"],
                   help="list every entry of this kind in a fixed order instead of searching")
    add_user(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("remember", help="Save an answer for a question")
    p.add_argument("question")
    p.add_argument("answer")
    add_user(p)
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("recall", help="Look up a saved answer")
    p.add_argument("question")
    add_user(p)
    p.set_defaults(func=cmd_recall)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
