"""Typer CLI entry point for the autoluthier command.

Every command is a thin wrapper over `pipeline`/`export`/`helpers` — no processing decisions
are made here, per the project's "CLI and server are both thin clients over pipeline + export"
rule (see ../../../CLAUDE.md). Each command lives in its own module under `autoluthier.cli`;
this module only assembles them onto one `Typer` app, matching the target layout's "app.py (+
one module per command)".

The CLI is a thin *alternative* surface, not the primary one — the web-first decision means
every one of these commands (plus the helpers) is also reachable through the browser once the
server (step 9) exists, calling the exact same underlying functions.
"""

from __future__ import annotations

import typer

from autoluthier.cli.concat import concat_command
from autoluthier.cli.init import init_command
from autoluthier.cli.midi import midi_command
from autoluthier.cli.prenorm import prenorm_command
from autoluthier.cli.preview import preview_command
from autoluthier.cli.run import run_command
from autoluthier.cli.ui import ui_command
from autoluthier.cli.validate import validate_command

app = typer.Typer(
    name="autoluthier",
    help="Batch multisample instrument builder.",
    no_args_is_help=True,
)

app.command("init")(init_command)
app.command("run")(run_command)
app.command("validate")(validate_command)
app.command("preview")(preview_command)
app.command("ui")(ui_command)
app.command("midi")(midi_command)
app.command("prenorm")(prenorm_command)
app.command("concat")(concat_command)


def main() -> None:
    """Run the autoluthier CLI."""
    app()


if __name__ == "__main__":
    main()
