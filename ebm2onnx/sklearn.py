from . import context
from . import convert

import onnx


try:
    from skl2onnx.common.data_types import Int64TensorType, FloatTensorType, StringTensorType


    def ebm_output_shape_calculator(operator):
        op = operator.raw_operator

        operator.outputs[0].type = Int64TensorType([None])  # label
        operator.outputs[1].type = FloatTensorType([None, len(op.classes_)])  # probabilities


    def convert_ebm_classifier(scope, operator, container):
        """Converts an EBM model to ONNX with sklearn-onnx
        """
        op = operator.raw_operator

        input_name = operator.inputs[0].onnx_name
        ctx = context.create(
            generate_variable_name=scope.get_unique_variable_name,
            generate_operator_name=scope.get_unique_operator_name,
        )

        g = convert.to_graph(
            op, dtype=(input_name, 'float'),
            name="ebm",
            predict_proba=True,
            prediction_name="label",
            probabilities_name="probabilities",
            context=ctx
        )

        for node in g.nodes:
            v = container._get_op_version(node.domain, node.op_type)
            container.node_domain_version_pair_sets.add((node.domain, v))

        container.nodes.extend(g.nodes)

        for i in g.initializers:
            content = i.SerializeToString()
            container.initializers_strings[content] = i.name
            container.initializers.append(i)

except Exception:
    def ebm_output_shape_calculator(operator):
        raise ImportError('skl2onnx not found. Please install it to use serialize a model via scikit-learn')

    def convert_ebm_classifier(scope, operator, container):
        raise ImportError('skl2onnx not found. Please install it to use serialize a model via scikit-learn')
