import pytest

from slam.features.learned_features import OptionalFeatureDependencyError, create_learned_matcher


def test_unknown_learned_matcher_is_rejected():
    with pytest.raises(ValueError, match="lightglue"):
        create_learned_matcher("unknown")


def test_missing_learned_matcher_dependency_has_install_guidance(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightglue":
            raise ImportError("missing lightglue for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalFeatureDependencyError, match=r"pip install -e \.\[modern\]"):
        create_learned_matcher("lightglue")
