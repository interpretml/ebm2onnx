import os
import tempfile
import pytest

import numpy as np
import onnx
from onnx.checker import check_model
import onnxruntime as rt
import ebm2onnx.graph as graph


def create_session(model):
    _, filename = tempfile.mkstemp()
    try:
        onnx.save_model(model, filename)
        #with open(filename, "wb") as f:
        #    f.write(model.SerializeToString())

        sess = rt.InferenceSession(filename)
        return sess
    finally:
        os.unlink(filename)


def infer_model(model, input):
    check_model(model)

    _, filename = tempfile.mkstemp()
    try:
        onnx.save_model(model, filename)
        #with open(filename, "wb") as f:
        #    f.write(model.SerializeToString())

        sess = rt.InferenceSession(filename)
        for o in sess.get_outputs():
            print(o)
        pred = sess.run(None, input)

        return pred

    finally:
        os.unlink(filename)


def assert_model_result(
    g, input,
    expected_result,
    exact_match=False,
    atol=1e-08,
    save_path=None
):
    model = graph.to_onnx(g)
    check_model(model)
    _, filename = tempfile.mkstemp()

    try:
        onnx.save_model(model, filename)
        if save_path:
            print("saving model...")
            onnx.save_model(model, save_path)
        sess = rt.InferenceSession(filename)
        pred = sess.run(None, input)

        print(pred)
        print(expected_result)
        for i, p in enumerate(pred):
            if exact_match:
                assert p.tolist() == expected_result[i]
            else:
                assert np.allclose(p, np.array(expected_result[i]))

    finally:
        os.unlink(filename)
