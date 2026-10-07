"""Smoke tests: imports, model loading/inference, and docking-world composition."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if os.environ.get("AUV_SIM_ROOT"):
    sys.path.insert(0, str(Path(os.environ["AUV_SIM_ROOT"]) / "src/uuv_sim_physics"))


def _perception_server():
    spec = importlib.util.spec_from_file_location("perception_server", REPO / "tools/perception_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_package_imports():
    import docking_perception.models.transformer  # noqa: F401
    import docking_perception.integration.heavy_v5_protocol  # noqa: F401


@pytest.mark.parametrize("name", ["cleanH.pt", "rich_domain_transformer_v2.pt"])
def test_model_loads_and_infers(name):
    torch = pytest.importorskip("torch")
    from docking_perception.integration import heavy_v3_protocol as P
    srv = _perception_server()
    model, norm, cal = srv.load(str(REPO / "models" / name), "cpu")
    assert sum(p.numel() for p in model.parameters()) == 1_194_889
    camera = np.zeros(P.CAMERA_SHAPE, np.uint8)
    scan = np.zeros(P.SCAN_SHAPE, np.float32)
    se2, sigma = srv.infer(model, norm, cal, camera, scan, "cpu")
    assert np.all(np.isfinite(se2)) and np.all(np.isfinite(sigma))


def test_docking_world_composes():
    sim = os.environ.get("AUV_SIM_ROOT")
    if not sim:
        pytest.skip("AUV_SIM_ROOT not set")
    pytest.importorskip("yaml")
    from uuv_sim_physics import world_builder
    from docking_perception.integration.playground_world import build_playground_docking_world
    heavy = world_builder.build_world_sdf(validated=True, vehicle="bluerov2_heavy",
                                          arm_sensing="CR6_REALISTIC_FLS_DVL", lockstep_depth_sensor=True)
    text = build_playground_docking_world(Path(sim) / "src/uuv_mode_aware_navigation/worlds/mode_aware_survey.sdf",
                                          heavy, REPO / "worlds/underwater_docking_physics.sdf",
                                          world_name="mode_aware_survey")
    assert "<sdf" in text and "bluerov2_heavy" in text
