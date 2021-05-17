from collections import namedtuple

import onnx


Graph = namedtuple('Graph', ['generate_name','inputs', 'outputs', 'transients', 'nodes', 'initializers'])
Graph.__new__.__defaults__ = ([], [], [], [], [])


def create_name_generator():
    state = {}
    def _generate_unique_name(name):
        if name in state:
            i = state[name]
            state[name] += 1
        else:
            state[name] = 0
            
        return "{}_{}".format(name, state[name])
        
        
    return _generate_unique_name


def extend(i, val):
    """Extends a list as a copy
    """
    ii = list(i)
    ii.extend(val)
    return ii


def pipe(*args):
    pass


def create_graph():
    return Graph(
        generate_name=create_name_generator()
    )

def compile(graph, target_opset, name="ebm"):
    #outputs = graph.transients

    graph = onnx.helper.make_graph(
        nodes=graph.nodes,
        name=name,
        inputs=graph.inputs,    
        outputs=graph.outputs,
        initializer=graph.initializers,
    )
    model = onnx.helper.make_model(graph, producer_name='ebm2onnx')
    model.opset_import[0].version = target_opset
    return model

def create_input(graph, name, type, shape):
    input = onnx.helper.make_tensor_value_info(name , type, shape)
    return Graph(
        generate_name=graph.generate_name,
        inputs=[input],
        transients=[input],
    )


def add_output(graph, name, type, shape):
    output = onnx.helper.make_tensor_value_info(name , type, shape)
    return graph._replace(
        outputs=extend(graph.outputs, [output]),
    )


def create_initializer(graph, name, type, shape, value):
    initializer = onnx.helper.make_tensor(graph.generate_name(name) , type, shape, value)
    return Graph(
        generate_name=graph.generate_name,
        initializers=[initializer],
        transients=[initializer],
    )


def strip_to_transients(graph):
    return Graph(
        generate_name=graph.generate_name,
        transients=graph.transients,
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

    return g