"""UI rendering hints attached to config fields via Pydantic's ``json_schema_extra``.

Each hint rides along in ``ProjectConfig.model_json_schema()``, which the frontend
(step 10) uses to auto-render parameter rows without hand-writing a form per field.
"""

from __future__ import annotations

from typing import Any


def ui_hint(
    *,
    unit: str | None = None,
    step: float | None = None,
    fine_step: float | None = None,
    log: bool = False,
    group: str | None = None,
    order: int | None = None,
    help_text: str | None = None,
) -> dict[str, Any]:
    """Build a ``json_schema_extra`` dict of UI hints for one config field.

    Args:
        unit: Display unit suffix, e.g. ``"s"``, ``"ms"``, ``"dB"``, ``"%"``, ``"Hz"``.
        step: Drag/arrow-key increment for a draggable numeric field.
        fine_step: Increment used while shift-dragging for fine adjustment.
        log: Whether the field should use a logarithmic drag/scale response.
        group: Small-caps section header this field is rendered under.
        order: Sort position within its group (lower first).
        help_text: Short explanation shown as a tooltip.

    Returns:
        A dict containing only the hints that were actually provided, ready to pass as
        ``json_schema_extra=ui_hint(...)`` on a Pydantic ``Field``.
    """
    extra: dict[str, Any] = {}
    if unit is not None:
        extra["unit"] = unit
    if step is not None:
        extra["step"] = step
    if fine_step is not None:
        extra["fine_step"] = fine_step
    if log:
        extra["log"] = True
    if group is not None:
        extra["group"] = group
    if order is not None:
        extra["order"] = order
    if help_text is not None:
        extra["help"] = help_text
    return extra
