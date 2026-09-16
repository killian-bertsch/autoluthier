"""Human-readable notes written alongside the instrument.

Only one so far: velocity-mode normalize *measures* the dynamic range of the recording rather
than being told it, and the number it lands on is something the user has to type into their
sampler by hand if they want the same response there. Printing it into the SFZ alone would
hide it inside a linear ``amp_velcurve_1`` gain.
"""

from __future__ import annotations

VELOCITY_RANGE_FILENAME = "velocity_range.txt"


def velocity_range_report(instrument_name: str, dynamic_range_db: float) -> str:
    """Return the text of the measured-dynamic-range note for one instrument.

    Args:
        instrument_name: Name the instrument was exported under.
        dynamic_range_db: The range velocity-mode normalize measured, in dB.

    Returns:
        The report body, ending in a newline.
    """
    return (
        f"Velocity Dynamic Range Report - {instrument_name}\n"
        f"{'=' * 50}\n\n"
        f"Measured dynamic range: {dynamic_range_db:.1f} dB\n\n"
        f"Set the sampler's velocity curve minimum to:\n"
        f"  {-dynamic_range_db:.1f} dB\n\n"
        f"(This is the gain at velocity 1 relative to velocity 127, and matches the\n"
        f"amp_velcurve_1 opcode already written into the SFZ.)\n"
    )
