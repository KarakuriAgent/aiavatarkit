from pathlib import Path
import json

import numpy as np

from wakeword_enrollment_server.enrollment import WakewordEnrollmentManager


class FakeSettings:
    def __init__(self, root):
        self.wakeword_enrollment_dir = str(root)
        self.audio_wakeword_threshold = 0.5
        self.audio_wakeword_frame_ms = 80
        self.audio_wakeword_inference_framework = "onnx"
        self.debug = False


def test_wakeword_manager_stores_candidates_only_when_reading(tmp_path):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))

    assert manager.add_candidate(
        audio_bytes=b"\x00\x00" * 160,
        sample_rate=16000,
        duration=0.01,
        session_id="s1",
    ) is None

    manager.start_reading("robo-kanon")
    candidate = manager.add_candidate(
        audio_bytes=b"\x01\x00" * 160,
        sample_rate=16000,
        duration=0.01,
        session_id="s1",
    )

    assert candidate is not None
    assert candidate.wakeword == "robo-kanon"
    assert candidate.role is None
    assert Path(candidate.path).exists()
    assert manager.list_candidates()[0]["audio_url"] == f"/api/candidates/{candidate.id}/audio"


def test_wakeword_manager_imports_and_deletes_onnx_model(tmp_path):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))

    model = manager.import_model(
        wakeword="robo-kanon",
        filename="robo.onnx",
        content=b"onnx-bytes",
        threshold=0.7,
    )

    assert model["wakeword"] == "robo-kanon"
    assert model["threshold"] == 0.7
    assert Path(model["path"]).read_bytes() == b"onnx-bytes"
    assert manager.list_models()[0]["id"] == model["id"]

    manager.delete_model(model["id"])

    assert manager.list_models() == []
    assert not Path(model["path"]).exists()


def test_wakeword_manager_training_job_registers_generated_model(tmp_path, monkeypatch):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))
    commands = []

    def fake_run_command(command, job):
        commands.append(command)
        if command[3] == "export":
            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.write_bytes(b"generated-onnx")

    monkeypatch.setattr(manager, "_run_command", fake_run_command)

    job = manager.start_training_job(
        wakeword="ロボ花音",
        target_phrases=["ロボ花音"],
        model_name="robo-kanon",
        threshold=0.61,
        tts_backend="voxcpm",
        n_samples=10,
        n_samples_val=2,
        steps=3,
        background=False,
    )

    assert job["status"] == "succeeded"
    assert job["model_id"]
    assert len(commands) == 5
    assert commands[0][3] == "setup"
    assert commands[0][-1] == "--skip-acav"
    assert commands[-1][3] == "export"

    config = json.loads(Path(job["config_path"]).read_text())
    assert config["target_phrases"] == ["ロボ花音"]
    assert config["tts_backend"] == "voxcpm"
    assert config["steps"] == 3

    model = manager.get_model(job["model_id"])
    assert model.wakeword == "ロボ花音"
    assert model.threshold == 0.61
    assert Path(model.path).read_bytes() == b"generated-onnx"


def test_wakeword_manager_completes_incomplete_voxcpm_download(tmp_path, monkeypatch):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))
    incomplete_dir = tmp_path / "training" / "data" / "voxcpm" / "VoxCPM2"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "config.json").write_text("{}")
    (incomplete_dir / ".incomplete").write_text("partial")

    def fake_download_voxcpm_snapshot(job, model_dir):
        (model_dir / "model.safetensors").write_bytes(b"model")
        (model_dir / "audiovae.pth").write_bytes(b"vae")

    def fake_run_command(command, job):
        if command[3] == "export":
            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.write_bytes(b"generated-onnx")

    monkeypatch.setattr(manager, "_download_voxcpm_snapshot", fake_download_voxcpm_snapshot)
    monkeypatch.setattr(manager, "_run_command", fake_run_command)

    job = manager.start_training_job(
        wakeword="ロボ花音",
        target_phrases=["ロボ花音"],
        tts_backend="voxcpm",
        n_samples=1,
        n_samples_val=1,
        steps=1,
        background=False,
    )

    assert job["status"] == "succeeded"
    assert (incomplete_dir / ".incomplete").exists()
    assert (incomplete_dir / "model.safetensors").exists()
    assert (incomplete_dir / "audiovae.pth").exists()


def test_wakeword_manager_configured_tts_generates_clips_without_livekit_generate(tmp_path, monkeypatch):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))
    commands = []

    def fake_synthesize(text):
        del text
        return np.sin(np.linspace(0, np.pi * 8, 16000, dtype=np.float32)) * 0.2

    def fake_run_command(command, job):
        commands.append(command)
        if command[3] == "export":
            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.write_bytes(b"generated-onnx")

    monkeypatch.setattr(manager, "_synthesize_configured_tts_audio", fake_synthesize)
    monkeypatch.setattr(manager, "_run_command", fake_run_command)

    job = manager.start_training_job(
        wakeword="ロボ花音",
        target_phrases=["ロボ花音"],
        model_name="robo-kanon",
        tts_backend="configured_tts",
        positive_source="configured_tts_only",
        negative_source="configured_tts_only",
        n_samples=3,
        n_samples_val=2,
        steps=1,
        background=False,
    )

    assert job["status"] == "succeeded"
    assert [command[3] for command in commands] == ["augment", "train", "export"]

    output_dir = Path(job["work_dir"]) / "output" / job["model_name"]
    assert len(list((output_dir / "positive_train").glob("clip_*.wav"))) == 3
    assert len(list((output_dir / "negative_train").glob("clip_*.wav"))) == 3
    assert len(list((output_dir / "background_train").glob("clip_*.wav"))) == 20

    config = json.loads(Path(job["config_path"]).read_text())
    assert config["tts_backend"] == "piper_vits"


def test_wakeword_manager_recorded_sources_use_labeled_candidates(tmp_path, monkeypatch):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))
    commands = []

    manager.start_reading("ロボ花音")
    positive = manager.add_candidate(
        audio_bytes=b"\x01\x00" * 16000,
        sample_rate=16000,
        duration=1.0,
        session_id="s1",
    )
    manager.start_reading("negative")
    negative = manager.add_candidate(
        audio_bytes=b"\x02\x00" * 16000,
        sample_rate=16000,
        duration=1.0,
        session_id="s1",
    )

    def fail_synthesize(text):
        raise AssertionError(f"TTS should not be used: {text}")

    def fake_run_command(command, job):
        commands.append(command)
        if command[3] == "export":
            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.write_bytes(b"generated-onnx")

    monkeypatch.setattr(manager, "_synthesize_configured_tts_audio", fail_synthesize)
    monkeypatch.setattr(manager, "_run_command", fake_run_command)

    job = manager.start_training_job(
        wakeword="ロボ花音",
        target_phrases=["ロボ花音"],
        model_name="recorded-only",
        tts_backend="configured_tts",
        positive_source="recorded_only",
        negative_source="recorded_only",
        n_samples=2,
        n_samples_val=1,
        steps=1,
        background=False,
    )

    assert positive is not None
    assert negative is not None
    assert job["status"] == "succeeded"
    assert [command[3] for command in commands] == ["augment", "train", "export"]

    output_dir = Path(job["work_dir"]) / "output" / job["model_name"]
    assert len(list((output_dir / "positive_train").glob("clip_*.wav"))) == 2
    assert len(list((output_dir / "negative_train").glob("clip_*.wav"))) == 2


def test_wakeword_manager_explicit_candidate_roles_override_legacy_labels(tmp_path, monkeypatch):
    manager = WakewordEnrollmentManager(FakeSettings(tmp_path))
    commands = []

    manager.start_reading("ロボ花音")
    positive = manager.add_candidate(
        audio_bytes=b"\x01\x00" * 16000,
        sample_rate=16000,
        duration=1.0,
        session_id="s1",
    )
    explicit_negative = manager.add_candidate(
        audio_bytes=b"\x02\x00" * 16000,
        sample_rate=16000,
        duration=1.0,
        session_id="s1",
    )
    ignored = manager.add_candidate(
        audio_bytes=b"\x03\x00" * 16000,
        sample_rate=16000,
        duration=1.0,
        session_id="s1",
    )

    assert positive is not None
    assert explicit_negative is not None
    assert ignored is not None

    manager.update_candidate_role(explicit_negative.id, "negative")
    manager.update_candidate_role(ignored.id, "ignore")

    reloaded = WakewordEnrollmentManager(FakeSettings(tmp_path))
    assert reloaded.get_candidate(explicit_negative.id).role == "negative"
    assert reloaded.get_candidate(ignored.id).role == "ignore"

    def fail_synthesize(text):
        raise AssertionError(f"TTS should not be used: {text}")

    def fake_run_command(command, job):
        commands.append(command)
        if command[3] == "export":
            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.write_bytes(b"generated-onnx")

    monkeypatch.setattr(reloaded, "_synthesize_configured_tts_audio", fail_synthesize)
    monkeypatch.setattr(reloaded, "_run_command", fake_run_command)

    job = reloaded.start_training_job(
        wakeword="ロボ花音",
        target_phrases=["ロボ花音"],
        model_name="explicit-roles",
        tts_backend="configured_tts",
        positive_source="recorded_only",
        negative_source="recorded_only",
        n_samples=2,
        n_samples_val=1,
        steps=1,
        background=False,
    )

    assert job["status"] == "succeeded"
    assert [command[3] for command in commands] == ["augment", "train", "export"]

    output_dir = Path(job["work_dir"]) / "output" / job["model_name"]
    assert len(list((output_dir / "positive_train").glob("clip_*.wav"))) == 2
    assert len(list((output_dir / "negative_train").glob("clip_*.wav"))) == 2
