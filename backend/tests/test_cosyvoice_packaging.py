"""Regression coverage for CosyVoice dependencies in frozen server builds."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_binary import build_server


@pytest.fixture
def captured_build_args():
    """Capture PyInstaller options without creating a binary or fetching model files."""
    with (
        patch("build_binary.PyInstaller.__main__.run") as mock_run,
        patch("build_binary.platform.system", return_value="Darwin"),
        patch("build_binary.os.chdir"),
    ):
        build_server()
        return mock_run.call_args.args[0]


def _values_after(args: list[str], option: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args[:-1]) if value == option]


def test_frozen_cosyvoice_runtime_collects_modelscope_package(captured_build_args):
    """ModelScope lazy imports must be bundled with the CosyVoice desktop server."""
    assert "modelscope" in _values_after(captured_build_args, "--collect-all")
    assert "modelscope" in _values_after(captured_build_args, "--copy-metadata")
