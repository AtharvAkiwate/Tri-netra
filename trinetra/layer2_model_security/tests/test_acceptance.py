import os
import json
import pytest
from trinetra.layer2_model_security.api import scan_model, heal_model
from trinetra.layer2_model_security.attacks.make_backdoored_model import generate_backdoored_model

def validate_report(report):
    # Manual dependency-free validation
    assert report["module"] == "layer2_model_security"
    assert report["schema_version"] == "1.0"
    assert isinstance(report["scan_id"], str)
    assert isinstance(report["timestamp_utc"], str)
    
    m = report["model"]
    assert isinstance(m["path"], str)
    assert m["format"] in ["onnx", "pytorch"]
    assert isinstance(m["sha256"], str)
    assert isinstance(m["size_bytes"], int)
    assert isinstance(m["num_classes"], int)
    
    assert report["access_mode"] in ["white_box", "black_box"]
    assert report["verdict"] in ["CLEAN", "SUSPICIOUS", "BACKDOORED"]
    assert report["recommended_action"] in ["ACCEPT", "REVIEW", "QUARANTINE"]
    
    assert 0.0 <= report["confidence"] <= 1.0
    assert 0.0 <= report["risk_score"] <= 100.0
    
    sus = report["suspect"]
    assert isinstance(sus["target_class"], (int, type(None)))
    assert isinstance(sus["neuron_indices"], list)
    
    ev = report["evidence"]
    assert "activation_clustering" in ev
    assert "spectral_signature" in ev
    assert "abs_stimulation" in ev
    assert "trigger_reconstruction" in ev
    
    art = report["artifacts"]
    assert "gradcam_heatmap_png" in art
    assert "trigger_reconstruction_png" in art
    
    assert isinstance(report["limitations"], list)
    assert isinstance(report["timings_sec"], dict)
    
    return True

@pytest.fixture(scope="session")
def demo_models(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("models")
    out_dir_str = str(out_dir)
    meta = generate_backdoored_model(out_dir_str, poison_rate=0.1)
    return meta

def test_clean_model(demo_models, tmp_path):
    report = scan_model(demo_models["paths"]["clean_pt"], output_dir=str(tmp_path))
    validate_report(report)
    assert report["verdict"] == "CLEAN"
    assert report["recommended_action"] == "ACCEPT"

def test_backdoored_pt(demo_models, tmp_path):
    report = scan_model(demo_models["paths"]["backdoored_pt"], output_dir=str(tmp_path))
    validate_report(report)
    assert report["verdict"] == "BACKDOORED"
    assert report["recommended_action"] == "QUARANTINE"
    assert report["access_mode"] == "white_box"
    assert report["suspect"]["target_class"] == demo_models["target_class"]
    
    # Test Healing
    heal_rep = heal_model(demo_models["paths"]["backdoored_pt"], report, output_dir=str(tmp_path))
    assert heal_rep["repair_successful"] is True

def test_backdoored_onnx(demo_models, tmp_path):
    report = scan_model(demo_models["paths"]["backdoored_onnx"], output_dir=str(tmp_path))
    validate_report(report)
    assert report["verdict"] == "BACKDOORED"
    # We implemented ONNX mostly as black box or basic white box
    # It should still be detected

def test_locked_model(demo_models, tmp_path):
    report = scan_model(demo_models["paths"]["locked_bin"], mode="auto", output_dir=str(tmp_path))
    validate_report(report)
    assert report["access_mode"] == "black_box"
    assert report["fallback_reason"] is not None
    assert report["verdict"] == "BACKDOORED"

def test_determinism(demo_models, tmp_path):
    r1 = scan_model(demo_models["paths"]["backdoored_pt"], output_dir=str(tmp_path), seed=42)
    r2 = scan_model(demo_models["paths"]["backdoored_pt"], output_dir=str(tmp_path), seed=42)
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["verdict"] == r2["verdict"]
