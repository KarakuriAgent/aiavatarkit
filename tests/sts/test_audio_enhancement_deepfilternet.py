from pathlib import Path

from aiavatar.sts.audio_enhancement import DeepFilterNetAudioEnhancer


def test_deepfilternet_python_cli_command_shape():
    enhancer = DeepFilterNetAudioEnhancer(command="deepFilter", model="DeepFilterNet2")

    command = enhancer._build_deepfilter_command(Path("input.wav"), Path("out"))

    assert command == [
        "deepFilter",
        "input.wav",
        "--output-dir",
        "out",
        "--log-level",
        "error",
        "--model-base-dir",
        "DeepFilterNet2",
    ]


def test_deepfilternet_rust_cli_command_shape():
    enhancer = DeepFilterNetAudioEnhancer(
        command="/usr/local/bin/deep-filter",
        model="model.tar.gz",
    )

    command = enhancer._build_deepfilter_command(Path("input.wav"), Path("out"))

    assert command == [
        "/usr/local/bin/deep-filter",
        "--output-dir",
        "out",
        "--model",
        "model.tar.gz",
        "input.wav",
    ]
