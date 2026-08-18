"""Typer CLI entry point for the autosampler command."""

import typer

app = typer.Typer(
    name="autosampler",
    help="Batch multisample instrument builder.",
    no_args_is_help=True,
)


def main() -> None:
    """Run the autosampler CLI."""
    app()


if __name__ == "__main__":
    main()
