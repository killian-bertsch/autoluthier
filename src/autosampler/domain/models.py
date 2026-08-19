"""Float32 in-memory audio and note/velocity indexing shared across the pipeline.

Replaces V1's ``Sample``/``InstrumentData`` (``src/data.py``): same shape, float32 instead
of float64 (V2's memory-budget decision), and a dict index on ``SampleSet`` for O(1)
``(note, velocity)`` lookup instead of V1's linear scan in ``InstrumentData.get_sustain``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Sample:
    """One ``(note, velocity)`` audio slice.

    ``audio`` is float32, shape ``(n_frames,)`` for mono or ``(n_frames, n_channels)`` for
    multichannel — every stage from the DSP framework onward must handle both shapes.

    ``loop_crossfade_frames`` is the crossfade length `dsp.loop` settled on for this specific
    sample after clamping to the audio it actually had available. In ``baked`` mode that fade is
    already in ``audio`` and the value is informational (reports, UI); in ``sfz`` mode the audio
    is untouched and this is the length export writes into the ``loop_crossfade`` opcode. It is
    per-sample rather than one project-wide number because the clamp depends on each sample's
    own length and detected loop.
    """

    note: int
    velocity: int
    audio: NDArray[np.float32]
    sample_rate: int
    loop_start: int | None = None
    loop_end: int | None = None
    loop_crossfade_frames: int = 0

    @property
    def n_frames(self) -> int:
        """Number of audio frames in this sample."""
        return int(self.audio.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of audio channels: 1 for a mono (1-D) array."""
        return 1 if self.audio.ndim == 1 else int(self.audio.shape[1])


@dataclass
class SampleSet:
    """An ordered collection of `Sample`s with O(1) lookup by ``(note, velocity)``."""

    samples: list[Sample]
    _index: dict[tuple[int, int], Sample] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Build the ``(note, velocity)`` lookup index from ``samples``."""
        self._index = {(s.note, s.velocity): s for s in self.samples}

    def get(self, note: int, velocity: int) -> Sample | None:
        """Look up a sample by exact ``(note, velocity)``, or ``None`` if absent."""
        return self._index.get((note, velocity))

    def notes(self) -> list[int]:
        """Sorted unique MIDI notes present in this set."""
        return sorted({s.note for s in self.samples})

    def velocities(self) -> list[int]:
        """Sorted unique velocity values present in this set."""
        return sorted({s.velocity for s in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)


@dataclass
class InstrumentAudio:
    """Sliced sustain + release samples for one instrument, before any DSP is applied."""

    sustain: SampleSet
    release: SampleSet
