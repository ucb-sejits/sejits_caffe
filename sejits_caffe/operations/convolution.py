from ctree.frontend import get_ast
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from collections import namedtuple
from ctree.transformations import PyBasicConversions
import ctree.c.nodes as C
from ctree.nodes import Project
import ctypes as ct
import ast

from hindemith import hmarray
import numpy as np

arr_cfg = namedtuple('arr_cfg', ['shape', 'dtype'])
tuple_cfg = namedtuple('tuple_cfg', ['val'])


op_map = {
    C.Op.Add: lambda x, y: x + y,
    C.Op.Div: lambda x, y: x / y,
    C.Op.Lt: lambda x, y: x < y,
    C.Op.Sub: lambda x, y: x - y,
}


class Backend(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg
        self.tuple_table = {}
        self.cfg_dict = {}
        self.loop_shape_map = {}

    def visit_FunctionDecl(self, node):
        params = []
        for param, cfg in zip(node.params, self.arg_cfg):
            if type(cfg) == arr_cfg:
                param.type = np.ctypeslib.ndpointer(cfg.dtype, len(cfg.shape),
                                                    cfg.shape)()
                params.append(param)
                self.cfg_dict[param.name] = cfg
            else:
                self.tuple_table[param.name] = cfg
        node.params = params
        node.defn = list(map(self.visit, node.defn))
        return node

    def gen_loop_nest(self, loopvars, cfg):
        body = []
        node = C.For(C.Assign(C.SymbolRef(loopvars[0], ct.c_int()),
                              C.Constant(0)),
                     C.Lt(C.SymbolRef(loopvars[0]), C.Constant(cfg.shape[0])),
                     C.PostInc(C.SymbolRef(loopvars[0])),
                     body)
        curr_node = node
        for loopvar, dim in zip(loopvars[1:], cfg.shape[1:]):
            curr_node = C.For(C.Assign(C.SymbolRef(loopvar, ct.c_int()),
                                       C.Constant(0)),
                              C.Lt(C.SymbolRef(loopvar), C.Constant(dim)),
                              C.PostInc(C.SymbolRef(loopvar)),
                              [])
            body.append(curr_node)
            body = curr_node.body
        self.loop_shape_map[loopvars] = cfg.shape
        return node, curr_node

    def is_loop_by_index(self, node):
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Attribute):
                if node.iter.func.attr == 'indices':
                    return True
        return False

    def visit_For(self, node):
        if self.is_loop_by_index(node):
            cfg = self.cfg_dict[node.iter.func.value.id]
            loopvars = tuple(var.id for var in node.target.elts)
            outer, inner = self.gen_loop_nest(loopvars, cfg)
            inner.body = list(map(self.visit, node.body))
            return outer

        node.body = list(map(self.visit, node.body))
        return node

    def gen_loop_index(self, loopvars, shape):
        curr = C.SymbolRef(loopvars[-1])
        for i in reversed(range(len(loopvars) - 1)):
            curr = C.Add(
                C.Mul(C.SymbolRef(loopvars[i]),
                      C.Constant(np.prod(shape[i + 1:]))),
                curr
            )
        return curr

    def fold_add(self, node):
        if isinstance(node.left, C.Constant) and node.left.value == 0:
            return node.right
        elif isinstance(node.right, C.Constant) and node.right.value == 0:
            return node.left
        return node

    def fold_sub(self, node):
        if isinstance(node.left, C.Constant) and node.left.value == 0:
            return C.Op.SubUnary(node.right)
        elif isinstance(node.right, C.Constant) and node.right.value == 0:
            return node.left
        return node

    def fold_mul(self, node):
        if isinstance(node.left, C.Constant) and node.left.value == 1:
            return node.right
        elif isinstance(node.right, C.Constant) and node.right.value == 1:
            return node.left
        return node

    def constant_fold(self, node):
        if isinstance(node.left, C.Constant) and \
                isinstance(node.right, C.Constant):
            return C.Constant(op_map[node.op.__class__](
                node.left.value, node.right.value))
        elif isinstance(node.op, C.Op.Add):
            return self.fold_add(node)
        elif isinstance(node.op, C.Op.Sub):
            return self.fold_sub(node)
        elif isinstance(node.op, C.Op.Mul):
            return self.fold_mul(node)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if isinstance(node.left, C.SymbolRef):
                target = node.left.name
                if target in self.tuple_table:
                    val = self.tuple_table[target]
                    return C.Constant(val[node.right.value])
                elif target in self.cfg_dict:
                    loopvars = tuple(var.name for var in node.right.elts)
                    node.right = self.gen_loop_index(
                        loopvars, self.cfg_dict[target].shape)
                    return node
            if isinstance(node.left, ast.Attribute):
                if node.left.value.name in self.cfg_dict:
                    attr = getattr(self.cfg_dict[node.left.value.name],
                                   node.left.attr)
                    return C.Constant(attr[node.right.value])
                else:
                    raise NotImplementedError()
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return self.constant_fold(node)


class ConcreteStructuredGrid(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type):
        self._c_function = self._compile(entry_name, proj, entry_type)

    def __call__(self, *args, **kwargs):
        a = []
        for i in range(len(self._c_function.argtypes)):
            a.append(args[i])
        return self._c_function(*a)


class SpecializedStructuredGrid(LazySpecializedFunction):
    def args_to_subconfig(self, args, kwargs):
        arg_cfg = ()
        for arg in args:
            if isinstance(arg, hmarray):
                arg_cfg += (arr_cfg(arg.shape, arg.dtype), )
            elif isinstance(arg, tuple):
                arg_cfg += (arg, )
            else:
                raise Exception("Unsupport arg type {}".format(type(arg)))
        for key in kwargs:
            if isinstance(arg, tuple):
                arg_cfg += (kwargs[key], )
            else:
                raise Exception("Unsupport kwarg type {}".format(type(arg)))
        return arg_cfg

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        tree = PyBasicConversions().visit(tree)
        tree = Backend(arg_cfg).visit(tree)
        return tree

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        entry_type = (None, )
        for cfg in arg_cfg:
            if isinstance(cfg, arr_cfg):
                entry_type += (np.ctypeslib.ndpointer(cfg.dtype,
                                                      len(cfg.shape),
                                                      cfg.shape), )
        entry_type = ct.CFUNCTYPE(*entry_type)
        return ConcreteStructuredGrid('convolution_2d',
                                      Project(files),
                                      entry_type)


def structured_grid(fn):
    return SpecializedStructuredGrid(get_ast(fn))


@structured_grid
def convolution_2d(data, weights, output, padding=(0, 0), stride=(1, 1)):
    for y, x in output.indices():
        for j, i in weights.indices():
            yy = y * stride[0] - padding[0] + j
            xx = x * stride[1] - padding[1] + i
            if 0 <= yy < data.shape[0] and 0 <= xx < data.shape[1]:
                output[y, x] += weights[j, i] * data[yy, xx]
