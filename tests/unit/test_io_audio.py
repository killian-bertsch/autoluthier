"""Tests for io.reader and io.writer: file discovery, round-tripping, resample, subtypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autoluthier.config.schema import OutputConfig
from autoluthier.io.reader import find_audio_file, read_audio
from autoluthier.io.writer import resample, resolve_subtype, write_audio

SR = 44_100


class TestFindAudioFile:
    def test_finds_wav(self, tmp_path: Path) -> None:
        (tmp_path / "sustain.wav").write_bytes(b"")
        assert find_audio_file(tmp_path, "sustain") == tmp_path / "sustain.wav"

    def test_finds_flac_when_no_wav(self, tmp_path: Path) -> None:
        (tmp_path / "sustain.flac").write_bytes(b"")
        assert find_audio_file(tmp_path, "sustain") == tmp_path / "sustain.flac"

    def test_prefers_wav_over_flac(self, tmp_path: Path) -> None:
        (tmp_path / "sustain.wav").write_bytes(b"")
        (tmp_path / "sustain.flac").write_bytes(b"")
        assert find_audio_file(tmp_path, "sustain") == tmp_path / "sustain.wav"

    def test_none_when_missing(self, tmp_path: Path) -> None:
        assert find_audio_file(tmp_path, "release") is None


class TestReadAudio:
    def test_mono_round_trip(self, tmp_path: Path) -> None:
        original = np.linspace(-0.5, 0.5, SR, dtype=np.float32)
        path = tmp_path / "mono.wav"
        sf.write(str(path), original, SR)

        audio, sample_rate = read_audio(path)
        assert sample_rate == SR
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, original, atol=1e-4)

    def test_stereo_round_trip(self, tmp_path: Path) -> None:
        original = np.stack(
            [np.full(1000, 0.25, dtype=np.float32), np.full(1000, -0.25, dtype=np.float32)],
            axis=1,
        )
        path = tmp_path / "stereo.wav"
        sf.write(str(path), original, SR)

        audio, _ = read_audio(path)
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        np.testing.assert_allclose(audio, original, atol=1e-4)


class TestResolveSubtype:
    @pytest.mark.parametrize(
        ("sample_format", "expected"),
        [("pcm16", "PCM_16"), ("pcm24", "PCM_24"), ("pcm32", "PCM_32"), ("float32", "FLOAT")],
    )
    def test_wav_supports_every_format(self, sample_format: str, expected: str) -> None:
        assert resolve_subtype("wav", sample_format) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize("sample_format", ["pcm16", "pcm24"])
    def test_flac_supports_pcm16_pcm24(self, sample_format: str) -> None:
        resolve_subtype("flac", sample_format)  # type: ignore[arg-type]  # must not raise

    @pytest.mark.parametrize("sample_format", ["pcm32", "float32"])
    def test_flac_rejects_pcm32_and_float32(self, sample_format: str) -> None:
        with pytest.raises(ValueError, match="FLAC cannot encode"):
            resolve_subtype("flac", sample_format)  # type: ignore[arg-type]


class TestResample:
    def test_same_rate_is_noop(self) -> None:
        audio = np.random.default_rng(0).uniform(-1, 1, 1000).astype(np.float32)
        assert resample(audio, SR, SR) is audio

    def test_upsample_doubles_frame_count(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        result = resample(audio, SR, SR * 2)
        assert result.shape[0] == 2000

    def test_downsample_halves_frame_count(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        result = resample(audio, SR, SR // 2)
        assert result.shape[0] == 500

    def test_dtype_stays_float32(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        assert resample(audio, SR, 22_050).dtype == np.float32


class TestWriteAudio:
    def test_writes_readable_flac(self, tmp_path: Path) -> None:
        audio = np.linspace(-0.5, 0.5, 4410, dtype=np.float32)
        output = OutputConfig(container="flac", sample_format="pcm24")
        path = write_audio(tmp_path / "out.flac", audio, SR, output)

        read_back, sample_rate = sf.read(str(path), dtype="float32")
        assert sample_rate == SR
        np.testing.assert_allclose(read_back, audio, atol=1e-4)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        audio = np.zeros(100, dtype=np.float32)
        output = OutputConfig()
        path = write_audio(tmp_path / "nested" / "dir" / "out.flac", audio, SR, output)
        assert path.exists()

    def test_clips_to_unit_range(self, tmp_path: Path) -> None:
        audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        output = OutputConfig(container="wav", sample_format="float32")
        path = write_audio(tmp_path / "out.wav", audio, SR, output)

        read_back, _ = sf.read(str(path), dtype="float32")
        assert read_back.max() <= 1.0
        assert read_back.min() >= -1.0

    def test_resamples_when_configured(self, tmp_path: Path) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        output = OutputConfig(sample_rate=22_050)
        path = write_audio(tmp_path / "out.flac", audio, SR, output)

        _, sample_rate = sf.read(str(path))
        assert sample_rate == 22_050

    def test_flac_pcm32_raises_before_writing(self, tmp_path: Path) -> None:
        audio = np.zeros(100, dtype=np.float32)
        output = OutputConfig(container="flac", sample_format="pcm32")
        path = tmp_path / "out.flac"
        with pytest.raises(ValueError, match="FLAC cannot encode"):
            write_audio(path, audio, SR, output)
        assert not path.exists()
