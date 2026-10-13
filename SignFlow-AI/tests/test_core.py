import importlib.util
from collections import deque
from pathlib import Path
import numpy as np

# Load the standalone source module without launching its CLI.
spec = importlib.util.spec_from_file_location("signflow_source", Path(__file__).parents[1] / "signflow_ai.py")
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

def test_empty_hand_feature_vector_is_fixed_size():
    class Result:
        hand_world_landmarks=[]
        handedness=[]
    values=mod.features(Result())
    assert values.shape==(126,)
    assert np.all(values==0)

def test_smoother_requires_multiple_votes():
    s=mod.Smoother(.5,3,2)
    s.update("hello",.8)
    label,score=s.update("hello",.9)
    assert label=="hello" and score>.8

def test_low_score_is_rejected():
    assert mod.Smoother(.7,3,2).update("gesture",.2)==("Uncertain",0.)

def test_invalid_smoother_settings():
    import pytest
    with pytest.raises(ValueError):
        mod.Smoother(.5,2,3)
