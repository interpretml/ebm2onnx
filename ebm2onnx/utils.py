

def opset(version: int, domain: str=""):
    def _operator(op):
        def _call(g):
            g = op(g)
            old_version = g.opsets.get(domain, -1)
            if version > old_version:
                g.opsets[domain] = version
            return g

        return _call

    return _operator
