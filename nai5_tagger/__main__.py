from __future__ import annotations

import argparse
import json
import sys

from nai5_tagger.pipeline import PipelineError, run_pipeline
from nai5_tagger.render import render
from nai5_tagger.types import CompileOptions, nai5_prompt_to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nai5_tagger")
    parser.add_argument("image")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-nl", action="store_true")
    args = parser.parse_args(argv)

    options = CompileOptions(include_nl=False) if args.no_nl else CompileOptions()
    try:
        prompt = run_pipeline(args.image, options)
    except PipelineError as exc:
        print(exc.message, file=sys.stderr)
        return exc.exit_code

    if args.json:
        print(json.dumps(nai5_prompt_to_dict(prompt), ensure_ascii=False, indent=2))
    else:
        print(render(prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
