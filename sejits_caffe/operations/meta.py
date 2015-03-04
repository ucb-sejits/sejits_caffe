import sys
import inspect
import copy
from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions
from ctree.transforms import ConstantFold
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project

import ast
import ctree.c.nodes as C
from ctree.types import get_ctype
from sejits_caffe.types import Array
from sejits_caffe.types.array import specialized_dispatch
import numpy as np
import ctypes as ct
from collections import namedtuple

arr_cfg = namedtuple('arr_cfg', ['shape', 'dtype'])


class Desugar(ast.NodeTransformer):
    """Desugar Python operators into the actual methods call"""
    def visit_Binop(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if node.op.__class__ in {ast.Add, ast.Mult, ast.Sub, ast.Div}:
            if isinstance(node.op, ast.Add):
                op = ast.Attribute(node.left, '__add__', ast.Load())
            elif isinstance(node.op, ast.Mult):
                op = ast.Attribute(node.left, '__mul__', ast.Load())
            elif isinstance(node.op, ast.Sub):
                op = ast.Attribute(node.left, '__sub__', ast.Load())
            elif isinstance(node.op, ast.Div):
                op = ast.Attribute(node.left, '__div__', ast.Load())
            return ast.Call(op, node.right)
        return node

    def visit_AugAssign(self, node):
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        if node.op.__class__ in {ast.Add, ast.Mult, ast.Sub, ast.Div}:
            if isinstance(node.op, ast.Add):
                op = ast.Attribute(node.target, '__add__', ast.Load())
            elif isinstance(node.op, ast.Mult):
                op = ast.Attribute(node.target, '__mul__', ast.Load())
            elif isinstance(node.op, ast.Sub):
                op = ast.Attribute(node.target, '__sub__', ast.Load())
            elif isinstance(node.op, ast.Div):
                op = ast.Attribute(node.target, '__div__', ast.Load())
            return ast.Assign([node.target],
                              ast.Call(op, [node.value], [], None, None))
        return node


class InlineEnvironment(ast.NodeTransformer):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self.decls = {}
        self.loop_vars = []
        self.files = []

    def visit_FunctionDef(self, node):
        self.decls = {}
        node.defn = [self.visit(s) for s in node.body]
        new_params = []
        for param in node.args.args:
            if sys.version_info > (3, 0):
                _id = param.arg
            else:
                _id = param.id
            if _id == 'self':
                continue
            value = self.symbol_table[_id]
            if isinstance(value, Array):
                _type = np.ctypeslib.ndpointer(
                    value.dtype, value.ndim, value.shape)()
            else:
                _type = get_ctype(value)
            new_params.append(
                C.SymbolRef(_id, _type))
        for name, value in self.decls.items():
            if isinstance(value, Array):
                type = np.ctypeslib.ndpointer(
                    value.dtype, value.ndim, value.shape)()
                value = value.ctypes.data
                new_params.append(
                    C.SymbolRef(name, type))
            else:
                if value is True:
                    value = 1
                    type = ct.c_int()
                elif value is False:
                    value = 0
                    type = ct.c_int()
                else:
                    type = get_ctype(value)
                node.body.insert(
                    0,
                    C.Assign(C.SymbolRef(name, type),
                             C.Constant(value)))
        node.args.args = new_params
        return node

    def visit_Subscript(self, node):
        node.slice = self.visit(node.slice)
        node.value = self.visit(node.value)
        if isinstance(node.value, C.SymbolRef):
            # Evaluate subscripts of constants immediately (i.e. tuple indices)
            if node.value.name in self.decls:
                if isinstance(node.slice.value, ast.Num):
                    value = self.decls.pop(node.value.name)
                    return C.Constant(value[node.slice.value.n])
        if isinstance(node.slice.value, ast.Tuple):
            if isinstance(node.value, C.SymbolRef):
                value = self.decls[node.value.name]
            else:
                value = self.eval_in_table(node.value)
            index = node.slice.value.elts
            c_index = ast.BinOp(index[-1], ast.Mult(),
                                ast.Num(np.prod(value.shape[len(index):])))
            for dim in reversed(range(len(index) - 1)):
                c_index = ast.BinOp(
                    ast.BinOp(index[dim], ast.Mult(),
                              ast.Num(np.prod(value.shape[dim+1:]))),
                    ast.Add(),
                    c_index)
            node.slice.value = c_index
        return node

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            # Lifts attributes that can be declared as constants in the
            # current scope
            value = node.value.id
            if value in self.symbol_table:
                name = "_".join((value, node.attr))
                self.decls[name] = getattr(self.symbol_table[value], node.attr)
                return C.SymbolRef(name)
        return node

    def eval_in_table(self, node):
        if isinstance(node, ast.Name):
            return self.symbol_table[node.id]
        elif isinstance(node, ast.Attribute):
            return getattr(self.eval_in_table(node.value), node.attr)
        elif isinstance(node, C.SymbolRef):
            return self.decls[node.name]
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice.value, ast.Tuple):
                index = self.eval_with_loop(node.slice.value.elts)
            else:
                index = self.eval_with_loop([node.slice.value])
            return self.eval_in_table(node.value)[index]
        raise NotImplementedError(node)

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            if len(node.iter.args) == 1:
                # Assume it starts at 0
                self.loop_vars.append((node.target.id, 0))
        node = super(InlineEnvironment, self).generic_visit(node)
        self.loop_vars.pop()
        return node

    def replace_loopvars_as_constants(self, node):
        if isinstance(node, ast.Name):
            for var in self.loop_vars:
                if var[0] == node.id:
                    return C.Constant(var[1])
        elif isinstance(node, ast.BinOp):
            node.left = self.replace_loopvars_as_constants(node.left)
            node.right = self.replace_loopvars_as_constants(node.right)
        return node

    def eval_with_loop(self, elts):
        new_elts = []
        for elt in elts:
            elt = self.replace_loopvars_as_constants(copy.deepcopy(elt))
            elt = PyBasicConversions().visit(elt)
            elt = ConstantFold().visit(elt)
            new_elts.append(elt.value)
        return tuple(new_elts)

    def table_contains(self, node):
        if isinstance(node, ast.Name):
            return node.id in self.symbol_table
        elif isinstance(node, ast.Attribute):
            return self.table_contains(node.value)
        elif isinstance(node, ast.Subscript):
            return self.table_contains(node.value)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'len':
            target = self.eval_in_table(node.args[0])
            return C.Constant(len(target))

        if self.table_contains(node.func):
            fn = self.eval_in_table(node.func)
            params = []
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Subscript):
                    value = self.eval_in_table(arg.value)
                    if isinstance(arg.slice.value, ast.Tuple):
                        index = self.eval_with_loop(arg.slice.value.elts)
                    else:
                        index = self.eval_with_loop([arg.slice.value])
                    params.append(value[index])
                    arg = self.visit(arg)
                    if isinstance(value[index], Array):
                        arg = C.Ref(arg)
                    args.append(arg)
                else:
                    arg = self.visit(arg)
                    if isinstance(arg, C.SymbolRef):
                        params.append(self.decls[arg.name])
                        args.append(arg)
                    elif isinstance(arg, ast.Tuple):
                        elts = ()
                        for elt in arg.elts:
                            if isinstance(elt, C.SymbolRef):
                                elts += (self.eval_in_table(elt), )
                            else:
                                elts += (elt, )
                        params.append(elts)
            if hasattr(fn, 'specialized_dispatch'):
                if fn.num_args:
                    trimmed = params[:fn.num_args]
                else:
                    trimmed = params
                fn = fn.fn(*params)
                params = trimmed
            cfg = fn._specializer.get_program_config(params, {})
            dir_name = fn._specializer.config_to_dirname(cfg)
            result = fn._specializer.get_transform_result(
                cfg, dir_name, cache=False)
            result[0].body[-1].static = True
            result[0].body[-1].inline = True
            self.files.extend(result)
            node.args = args
            node.func = ast.Name(result[0].body[-1].name, ast.Load())
        else:
            node.args = [self.visit(arg) for arg in node.args]
        return node


class ConcreteMeta(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, params, original_args):
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.params = params
        self.original_args = original_args

    def __call__(self, *args, **kwargs):
        a = []
        _self = args[0]
        for index, param in enumerate(self.params):
            if "self_" in param.name:
                a.append(getattr(_self, param.name[5:]))
            else:
                for i, arg in enumerate(self.original_args):
                    if sys.version_info > (3, 0):
                        _id = arg.arg
                    else:
                        _id = arg.id
                    if _id == param.name:
                        a.append(args[i])
                        break
        return self._c_function(*a)


class MetaSpecialized(LazySpecializedFunction):
    def __init__(self, tree, symbol_table):
        super(MetaSpecialized, self).__init__(tree)
        self.symbol_table = symbol_table

    def args_to_subconfig(self, args):
        cfg = ()
        for arg, name in zip(args, self.original_tree.body[0].args.args):
            if sys.version_info > (3, 0):
                self.symbol_table[name.arg] = arg
            else:
                self.symbol_table[name.id] = arg
            if isinstance(arg, Array):
                cfg += (arr_cfg(arg.shape, arg.dtype), )
            else:
                cfg += (arg, )
        return cfg

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        tree = Desugar().visit(tree)
        inliner = InlineEnvironment(self.symbol_table)
        tree = inliner.visit(tree)
        tree = PyBasicConversions().visit(tree)
        tree.find(C.For).pragma = 'omp parallel for'
        tree.name = self.original_tree.body[0].name
        body = []
        for file in inliner.files:
            body.extend(file.body)
        tree.body = body + tree.body
        print(tree)
        return [tree]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        proj = Project(files)
        entry_type = (None, )
        for param in files[0].body[-1].params:
            if "self_" in param.name:
                arg = getattr(arg_cfg[0], param.name[5:])
                arg = arr_cfg(arg.shape, arg.dtype)
            else:
                for index, p in enumerate(
                        self.original_tree.body[0].args.args):
                    if sys.version_info > (3, 0):
                        _id = p.arg
                    else:
                        _id = p.id
                    if _id == param.name:
                        arg = arg_cfg[index]
                        break

            if isinstance(arg, arr_cfg):
                entry_type += (np.ctypeslib.ndpointer(arg.dtype,
                                                      len(arg.shape),
                                                      arg.shape), )
            else:
                raise NotImplementedError()
        entry_type = ct.CFUNCTYPE(*entry_type)
        return ConcreteMeta(files[0].name, proj, entry_type,
                            files[0].body[-1].params,
                            self.original_tree.body[0].args.args)


def meta(fn):
    frame = inspect.stack()[1][0].f_back
    tree = get_ast(fn)
    symbol_table = frame.f_locals
    spec = MetaSpecialized(tree, symbol_table)

    def fn(*args, **kwargs):
        return spec(*args, **kwargs)
    return fn
