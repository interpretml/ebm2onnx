from typing import NamedTuple, Callable


class Context(NamedTuple):
    generate_variable_name: Callable[[], str]
    generate_operator_name: Callable[[], str]


def create_name_generator() -> Callable[[str], str]:
    state = {}

    def _generate_unique_name(name: str) -> str:
        """ Generates a new globaly unique name in the graph
        """
        if name in state:
            state[name] += 1
        else:
            state[name] = 0

        return "{}_{}".format(name, state[name])

    return _generate_unique_name


def create(
    generate_variable_name=None,
    generate_operator_name=None,
) -> Context:
    generate_variable_name = generate_variable_name or create_name_generator()
    generate_operator_name = generate_operator_name or create_name_generator()

    return Context(
        generate_variable_name=generate_variable_name,
        generate_operator_name=generate_operator_name,
    )
