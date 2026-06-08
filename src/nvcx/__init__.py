__all__ = [
    "AdaptiveMVScaleSelector",
    "HierarchicalContextModulation",
    "HierarchyOutput",
    "NeuralPostProcessor",
    "NeuralPreProcessor",
    "NeuralWrapper",
    "ReferenceState",
]


def __getattr__(name):
    if name == "AdaptiveMVScaleSelector":
        from .adaptive_inference import AdaptiveMVScaleSelector
        return AdaptiveMVScaleSelector
    if name in {"HierarchicalContextModulation", "HierarchyOutput", "ReferenceState"}:
        from .hierarchy import HierarchicalContextModulation, HierarchyOutput, ReferenceState
        return {
            "HierarchicalContextModulation": HierarchicalContextModulation,
            "HierarchyOutput": HierarchyOutput,
            "ReferenceState": ReferenceState,
        }[name]
    if name in {"NeuralPostProcessor", "NeuralPreProcessor", "NeuralWrapper"}:
        from .wrappers import NeuralPostProcessor, NeuralPreProcessor, NeuralWrapper
        return {
            "NeuralPostProcessor": NeuralPostProcessor,
            "NeuralPreProcessor": NeuralPreProcessor,
            "NeuralWrapper": NeuralWrapper,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
