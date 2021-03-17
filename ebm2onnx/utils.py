from onnx import defs


def get_latest_opset_version():
    """
    This module relies on *onnxruntime* to test every
    converter. The function returns the most recent
    target opset tested with *onnxruntime* or the opset
    version specified by *onnx* package if this one is lower
    (return by `onnx.defs.onnx_opset_version()`).
    """
    return min(13, defs.onnx_opset_version())
