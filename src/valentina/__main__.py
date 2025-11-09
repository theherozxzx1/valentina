"""Command line entry point for the Valentina toolkit."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valentina",
        description=(
            "Utilities for working with the Valentina Moreau generative image workflows."
        ),
    )
    parser.add_argument("--version", action="version", version=f"valentina {__version__}")
    return parser


def cli() -> None:
    """Run the command line interface."""
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    cli()
