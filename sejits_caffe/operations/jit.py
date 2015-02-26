from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions


def jit(fn):
    from ctree import browser_show_ast
    tree = get_ast(fn)
    PyBasicConversions().visit(tree)
    browser_show_ast(tree)
    return fn
