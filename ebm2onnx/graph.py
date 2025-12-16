import logging
from typing import NamedTuple, Optional, List, Dict, Union

import onnx
from onnx.helper import make_opsetid

from . import context as _context


class Graph(NamedTuple):
    context: _context.Context
    inputs: List[onnx.ValueInfoProto] = []
    outputs: List[onnx.ValueInfoProto] = []
    transients: List[onnx.ValueInfoProto] = []
    nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []
    opsets: Dict[str, int] = {}


def extend(i, val):
    """Extends a list as a copy
    """
    ii = list(i)
    ii.extend(val)
    return ii


def pipe(*args):
    pass


def create_graph(context=None) -> Graph:
    """Creates a new graph object.

    Returns:
        A Graph object.
    """
    if context is None:
        context = _context.create()
    return Graph(
        context=context
    )


def from_onnx(model) -> Graph:
    """Creates a graph object from an onnx model.

    Creating a graph from an existing model allows for editing it.

    Args:
        model: An ONNX model

    Returns:
        A Graph object.
    """
    opsets = {
        op.domain: op.version
        for op in model.opset_import
    }

    return Graph(
        context=_context.create(),
        inputs=[n for n in model.graph.input],
        outputs=[n for n in model.graph.output],
        nodes=[n for n in model.graph.node],
        initializers=[n for n in model.graph.initializer],
        opsets=opsets,
    )


def to_onnx(
    graph: Graph,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    name: Optional[str] = "ebm",
) -> Graph:
    """Converts a graph to an onnx model.

    If target_opset is an int, then is corresponds to the default domain
    'ai.onnx'. Using a dict allows to set opset versions for other domains
    like 'ai.onnx.ml'.

    Args:
        graph: The graph object
        target_opset: [Optional][Deprecated] the target opset to use when converting ot onnx, can be an int or a dict
        name: [Optional] An existing ONNX model

    Returns:
        A Graph object.
    """
    if target_opset:
        logging.warning("to_onnx: target_opset argument is deprecated")

    opset_imports = [
        make_opsetid(domain=domain, version=version)
        for domain,version in graph.opsets.items()
    ]

    graph = onnx.helper.make_graph(
        nodes=graph.nodes,
        name=name,
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializer=graph.initializers,
    )

    # create the onnx model from the graph.
    # The onnx library will set the ir version to the minimal required ir that
    # is compatible with the opset_imports provided.
    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name='ebm2onnx',
        opset_imports=opset_imports,
    )

    return model


def create_input(graph, name, type, shape):
    input = onnx.helper.make_tensor_value_info(name , type, shape)
    return Graph(
        context=graph.context,
        inputs=[input],
        transients=[input],
    )


def add_output(graph, name, type, shape):
    output = onnx.helper.make_tensor_value_info(name , type, shape)
    return graph._replace(
        outputs=extend(graph.outputs, [output]),
    )


def create_initializer(graph, name, type, shape, value):
    initializer = onnx.helper.make_tensor(graph.context.generate_variable_name(name) , type, shape, value)
    return Graph(
        context=graph.context,
        initializers=[initializer],
        transients=[initializer],
    )


def create_transient_by_name(graph, name, type, shape):
    input = onnx.helper.make_tensor_value_info(name, type, shape)
    return Graph(
        context=graph.context,
        transients=[input],
    )


def add_transient_by_name(graph, name, type=onnx.TensorProto.UNDEFINED, shape=[]):
    tname = [
        o
        for n in graph.nodes
        for o in n.output
        if o == name
    ]

    if len(tname) == 0:
        tname = [
            name
            for n in graph.initializers
            if n.name == name
        ]

    tname = tname[0]
    t = onnx.helper.make_tensor_value_info(tname, type, shape)
    return graph._replace(
        transients=extend(graph.transients, [t])
    )


def strip_to_transients(graph):
    """ Returns only the transients of a graph
    """
    return Graph(
        context=graph.context,
        transients=graph.transients,
    )


def clear_transients(graph):
    """ Removes all transients from a graph
    """
    return graph._replace(
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializers=graph.initializers,
        transients=[],
        nodes=graph.nodes,
    )


def merge(*args):
    g = None

    for graph in args:
        if g is None:
            g = graph
        else:
            g = g._replace(
                inputs=extend(g.inputs, graph.inputs),
                outputs=extend(g.outputs, graph.outputs),
                initializers=extend(g.initializers, graph.initializers),
                transients=extend(g.transients, graph.transients),
                nodes=extend(g.nodes, graph.nodes),
            )

            # merge opsets, keep higher version for each domain
            for domain,version in graph.opsets.items():
                cur_version = g.opsets.get(domain, -1)
                if version > cur_version:
                    g.opsets[domain] = version

    return g