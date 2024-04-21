"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator
from typing import Any
CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'
ERR_FILLER = "Filler"


class Frame:
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value: tp.Any = None
        self.stored: tp.Any = {}
        self.last_exception = None
        self.flag_for_continue = False
        self.cells = None
        self.i: int = 0
        self.return_generator_flag: bool = False
        self.yielded_mas: list[tp.Any] = []
        self.kw_names: tp.Any = None
        self.flag_build_class: bool = False
        self.methods: tp.Any = {}

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def kw_names_op(self, arg: int) -> None:
        self.kw_names = self.code.co_consts[arg]

    def call_op(self, arg: int) -> None:
        arguments = self.popn(arg)

        d = {}
        if self.kw_names is not None:
            for name in self.kw_names:
                d[name] = arguments.pop()

        f = self.pop()
        obj = self.pop()
        if obj is None:
            res = f(*arguments, **d)
            if hasattr(f, '__name__'):
                if f.__name__ == '__build_class__':
                    for name, method in self.methods.items():
                        setattr(res, name, method)
                    self.methods = {}
            self.push(res)
        elif callable(getattr(f, obj.__name__, False)):
            self.push(obj(*arguments, **d))
        else:
            try:
                self.push(obj(f, *arguments, **d))
            except AttributeError:
                self.push(obj(*arguments, **d))
            except TypeError:
                self.push(obj(*arguments, **d))
        self.kw_names = None

    def nop_op(self, arg: int) -> None:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def jump_forward(self, arg: int, offset: int) -> None:
        j = self.i
        while self.instructions[j].offset != offset + 2 + 2 * arg:
            j += 1
        self.i = j
        self.flag_for_continue = True

    def jump_backward(self, arg: int, offset: int) -> None:
        j = self.i
        while self.instructions[j].offset != offset + 2 - 2 * arg:
            j -= 1
        self.i = j
        self.flag_for_continue = True

    def pop_jump_forward_if_true(self, arg: int, offset: int) -> None:
        if self.pop():
            j = self.i
            while self.instructions[j].offset != offset + 2 + 2 * arg:
                j += 1
            self.i = j
            self.flag_for_continue = True

    def pop_jump_backward_if_true(self, arg: int, offset: int) -> None:
        if self.pop():
            j = self.i
            while self.instructions[j].offset != offset + 2 - 2 * arg:
                j -= 1
            self.i = j
            self.flag_for_continue = True

    def pop_jump_forward_if_false(self, arg: int, offset: int) -> None:
        if not self.pop():
            j = self.i
            while self.instructions[j].offset != offset + 2 + 2 * arg:
                j += 1
            self.i = j
            self.flag_for_continue = True

    def pop_jump_backward_if_false(self, arg: int, offset: int) -> None:
        if not self.pop():
            j = self.i
            while self.instructions[j].offset != offset + 2 - 2 * arg:
                j -= 1
            self.i = j
            self.flag_for_continue = True

    def jump_if_true_or_pop(self, arg: int, offset: int) -> None:
        if top := self.pop():
            j = self.i
            while self.instructions[j].offset != offset + 2 + 2 * arg:
                j += 1
            self.i = j
            self.push(top)
            self.flag_for_continue = True

    def jump_if_false_or_pop(self, arg: int, offset: int) -> None:
        if not (top := self.pop()):
            j = self.i
            while self.instructions[j].offset != offset + 2 + 2 * arg:
                j += 1
            self.i = j
            self.push(top)
            self.flag_for_continue = True

    def pop_jump_forward_if_none(self, arg: int, offset: int) -> None:
        top = self.pop()
        if top is None:
            j = self.i
            while self.instructions[j].offset != offset + 2 + 2 * arg:
                j += 1
            self.i = j
            self.push(top)
            self.flag_for_continue = True

    def peek(self, n: int) -> tp.Any:
        """Get a value `n` entries down in the stack, without changing the stack."""
        return self.data_stack[-n]

    def run(self) -> tp.Any:
        instructions = [instr for instr in dis.get_instructions(self.code)]
        self.instructions: list[tp.Any] = instructions
        while self.i < len(instructions):
            instruction: tp.Any = instructions[self.i]
            op_name = instruction.opname
            if op_name.lower() == 'for_iter':
                iterator = self.pop()
                try:
                    self.push(iterator)
                    self.push(next(iterator))
                except StopIteration:
                    j = self.i
                    arg: int = instruction.arg
                    offset: int = instruction.offset
                    while self.instructions[j].offset != offset + 2 + 2 * arg:
                        j += 1
                    self.i = j
                    self.pop()
                    continue
            elif 'jump' in op_name.lower():
                self.flag_for_continue = False
                getattr(self, op_name.lower())(instruction.arg, instruction.offset)
                if self.flag_for_continue:
                    continue
            elif op_name.startswith('UNARY_'):
                self.unaryOperator(op_name[6:])
            elif op_name.startswith('BUILD_'):
                getattr(self, op_name.lower() + "_op")(instruction.arg)
            elif op_name.startswith('COMPARE_OP'):
                getattr(self, op_name.lower() + "_op")(instruction.arg)
            elif op_name.startswith('BINARY_OP'):
                self.binaryOperator(instruction.argrepr)
            elif op_name.startswith('LIST_EXTEND'):
                getattr(self, op_name.lower() + "_op")(instruction.arg)
            elif op_name.startswith('LOAD_GLOBAL'):
                getattr(self, op_name.lower() + "_op")(instruction)
            elif op_name.startswith('FORMAT_VALUE'):
                getattr(self, op_name.lower() + "_op")(instruction.arg)
            elif op_name.startswith('KW_NAMES'):
                getattr(self, op_name.lower() + "_op")(instruction.arg)
            else:
                getattr(self, op_name.lower() + "_op")(instruction.argval)
            self.i += 1
        return self.return_value

    def call_function_ex_op(self, arg: int) -> None:
        kwargs = {}
        if arg & 1:
            kwargs = self.pop()
        args = list(self.pop())
        f = self.pop()
        self.push(f(*args, **kwargs))

    def load_name_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, instruction: tp.Any) -> None:
        arg = instruction.argval
        if arg in self.builtins:
            if 'NULL' in instruction.argrepr:
                self.push(None)
            self.push(self.builtins[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        else:
            raise NameError

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def load_const_op(self, arg: tp.Any) -> None:
        self.push(arg)

    def load_fast_op(self, name: str) -> None:
        if name in self.locals:
            self.push(self.locals[name])
        else:
            raise UnboundLocalError

    def return_value_op(self, arg: tp.Any) -> None:
        if not self.return_generator_flag:
            self.return_value = self.pop()
        else:
            self.return_value = (j for j in self.yielded_mas)

    def pop_top_op(self, arg: tp.Any) -> None:
        self.pop()

    def make_function_op(self, arg: int) -> None:
        code = self.pop()

        kwdflts, dflts = None, None
        if bool(arg & 2):
            kwdflts = self.pop()
        if bool(arg & 1):
            dflts = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = \
                bind_args(code, dflts, kwdflts, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            ans = frame.run()

            if self.flag_build_class:
                self.flag_build_class = False
                self.methods = frame.locals
            return ans

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        const = self.pop()
        self.locals[arg] = const

    def store_fast_op(self, arg: tp.Any) -> None:
        const = self.pop()
        self.locals[arg] = const

# INPLACE

    def inplaceOperator(self, x: tp.Any, y: tp.Any, op: str) -> None:
        if op == '**=':
            x **= y
        elif op == '*=':
            x *= y
        elif op == '//=':
            x //= y
        elif op == '/=':
            x /= y
        elif op == '%=':
            x %= y
        elif op == '+=':
            x += y
        elif op == '-=':
            x -= y
        elif op == '<<=':
            x <<= y
        elif op == '>>=':
            x >>= y
        elif op == '&=':
            x &= y
        elif op == '^=':
            x ^= y
        elif op == '|=':
            x |= y
        else:
            raise NameError
        self.push(x)

    COMPARE_OPERATORS = [
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        lambda x, y: x in y,
        lambda x, y: x not in y,
        lambda x, y: x is y,
        lambda x, y: x is not y,
        lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    ]

    def compare_op_op(self, arg: int) -> None:
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[arg](x, y))

    def unpack_sequence_op(self, arg: int) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def load_assertion_error_op(self, error: tp.Any) -> None:
        self.push(error)

# BINARY

    BINARY_OPERATORS = {
        '**': pow,
        '*': operator.mul,
        '//': operator.floordiv,
        '/': operator.truediv,
        '%': operator.mod,
        '+': operator.add,
        '-': operator.sub,
        '[]': operator.getitem,
        '<<': operator.lshift,
        '>>': operator.rshift,
        '&': operator.and_,
        '^': operator.xor,
        '|': operator.or_,
    }

    def binaryOperator(self, operation_name: str) -> None:
        x, y = self.popn(2)
        if '=' in operation_name:
            self.inplaceOperator(x, y, operation_name)
        else:
            oper: tp.Any = self.BINARY_OPERATORS[operation_name](x, y)  # type: ignore
            self.push(oper)

    def binary_subscr_op(self, arg: int) -> None:
        key = self.pop()
        collection = self.pop()
        self.push(collection[key])

# STRING

    def build_string_op(self, arg: int) -> None:
        ans = ''
        for j in range(arg):
            ans = self.pop() + ans
        self.push(ans)

# GENERATORS

    def return_generator_op(self, arg: int) -> None:
        self.return_generator_flag = True
        self.push(None)

    def yield_value_op(self, arg: int) -> None:
        self.yielded_mas.append(self.peek(1))

# UNARY

    UNARY_OPERATORS = {
        'POSITIVE': operator.pos,
        'NEGATIVE': operator.neg,
        'NOT': operator.not_,
        'CONVERT': repr,
        'INVERT': operator.invert,
    }

    def unaryOperator(self, op: str) -> None:
        x = self.pop()
        oper: tp.Any = self.UNARY_OPERATORS[op](x)  # type: ignore
        self.push(oper)

# SLICE

    def build_slice_op(self, arg: int) -> None:
        if arg == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        else:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))

    def store_subscr_op(self, arg: int) -> None:
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def delete_subscr_op(self, arg: int) -> None:
        obj, subscr = self.popn(2)
        del obj[subscr]

    def delete_global_op(self, arg: tp.Any) -> None:
        del self.globals[arg]

# BUILDING

    def build_list_op(self, count: int) -> None:
        lst = []
        for i in range(count):
            lst.append(self.pop())
        self.push(lst)

    def build_map_op(self, size: int) -> None:
        d: tp.Any = {}
        for i in range(size):
            value = self.pop()
            key = self.pop()
            if key not in d:
                d[key] = value
        self.push(d)

    def list_extend_op(self, count: int) -> None:
        val = self.pop()
        the_list = self.peek(count)
        the_list.extend(val)

    def build_const_key_map_op(self, size: int) -> None:
        self.push(dict(zip(self.pop(), self.popn(size))))

    def build_set_op(self, count: int) -> None:
        elts = self.popn(count)
        self.push(set(elts))

    def set_update_op(self, count: int) -> None:
        val = self.pop()
        the_set = self.peek(count)
        the_set.update(val)

    def build_tuple_op(self, count: int) -> None:
        elts = self.popn(count)
        self.push(tuple(elts))

    def list_to_tuple_op(self, arg: int) -> None:
        self.push(tuple(self.pop()))

    def dict_update_op(self, count: int) -> None:
        val, key = self.popn(2)
        the_map = self.peek(count)
        the_map[key] = val

    def list_append_op(self, arg: int) -> None:
        s_0 = self.pop()
        self.peek(arg).append(s_0)

    def map_add_op(self, arg: int) -> None:
        value = self.pop()
        s_0 = self.pop()
        self.peek(arg).__setitem__(s_0, value)

    def set_add_op(self, arg: int) -> None:
        s_0 = self.pop()
        self.peek(arg).add(s_0)

# IMPORT

    def import_name_op(self, name: str) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(name, self.globals, self.locals, fromlist, level)
        )

    def import_from_op(self, name: str) -> None:
        mod = self.top()
        self.push(getattr(mod, name))

# OTHER

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def raise_varargs_op(self, argc: int) -> str:
        tb = None
        if argc == 3:
            tb = self.pop()
        if tb:
            return 'reraise'
        else:
            return 'exception'

    def extended_arg_op(self, arg: tp.Any) -> None:
        pass

    def format_value_op(self, arg: tp.Any) -> None:
        if (arg & 0x03) == 0x00:
            value = self.pop()
            self.push(value.format())
        if (arg & 0x03) == 0x01:
            value = self.pop()
            self.push(str(value).format())
        if (arg & 0x03) == 0x02:
            value = self.pop()
            self.push(repr(value).format())
        if (arg & 0x03) == 0x03:
            value = self.pop()
            self.push(ascii(value).format())
        if (arg & 0x04) == 0x04:
            pass

    def load_method_op(self, arg: str) -> None:
        obj = self.pop()
        method = getattr(obj, arg, None)
        if method is None:
            raise AttributeError
        if callable(method):
            self.push(method)
            self.push(obj)

    def is_op_op(self, arg: int) -> None:
        first = self.pop()
        sec = self.pop()
        if arg == 1:
            self.push(first is not sec)
        else:
            self.push(first is sec)

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def store_deref_op(self, name: tp.Any) -> None:
        pass

# ATTRIBUTES AND INDEXING

    def load_attr_op(self, attr: str) -> None:
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)

    def delete_fast_op(self, name: tp.Any) -> None:
        del self.locals[name]

    def delete_name_op(self, name: tp.Any) -> None:
        del self.locals[name]

    def contains_op_op(self, arg: int) -> None:
        l_ = self.pop()
        r_ = self.pop()
        is_in = r_ in l_
        if arg == 1:
            self.push(not is_in)
        else:
            self.push(is_in)

    def copy_op(self, arg: int) -> None:
        self.push(self.peek(arg))

    def swap_op(self, arg: int) -> None:
        items = []
        for j in range(arg):
            items.append(self.pop())
        items[0], items[-1] = items[-1], items[0]
        for j in range(-1, -arg - 1, -1):
            self.push(items[j])

# IMPORT .

    def import_star_op(self, arg: tp.Any) -> None:
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.locals[attr] = getattr(mod, attr)

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.flag_build_class = True
        self.push(self.builtins['__build_class__'])


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()


def bind_args(func: tp.Any, dflts: tp.Any, kwdflts: tp.Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    res: dict[str, tp.Any] = {}
    is_args = bin(func.co_flags // CO_VARARGS % 2) == bin(1)
    is_kwargs = bin(func.co_flags // CO_VARKEYWORDS % 2) == bin(1)
    args_cnt = func.co_argcount + func.co_kwonlyargcount + int(is_args) + int(
        is_kwargs) + func.co_posonlyargcount
    if is_args and is_kwargs:
        args_name = func.co_varnames[:args_cnt][-2]
        res[args_name] = list()
        kwargs_name = func.co_varnames[:args_cnt][-1]
        res[kwargs_name] = dict()
    elif is_args:
        args_name = func.co_varnames[:args_cnt][-1]
        res[args_name] = list()
    elif is_kwargs:
        kwargs_name = func.co_varnames[:args_cnt][-1]
        res[kwargs_name] = dict()

    kw = list(func.co_varnames[func.co_argcount:])
    pos = func.co_varnames[:func.co_argcount]
    arg_non_def = func.co_argcount - len(dflts) if dflts else 0
    arg_count = func.co_argcount
    pos_only = func.co_varnames[:func.co_posonlyargcount]

    if is_kwargs and is_args:
        kw = kw[:-2]
    elif is_kwargs or is_args:
        kw = kw[:-1]

    kwonly = [elem for i, elem in enumerate(func.co_varnames)
              if func.co_argcount <= i < func.co_argcount + func.co_kwonlyargcount]

    kw_in_pos = len([name for name in pos if name in kwargs])
    dflts_size = len(dflts) if dflts else 0
    if len(args) + dflts_size + kw_in_pos < len(pos):
        raise TypeError(ERR_MISSING_POS_ARGS)

    for k in range(len(args)):
        if k < len(pos):
            res[pos[k]] = args[k]
        elif not is_args:
            raise TypeError(ERR_TOO_MANY_POS_ARGS)
        else:
            res[str(args_name)].append(args[k])

    if is_args:
        res[str(args_name)] = tuple(res[str(args_name)])

    for elem in kwargs:
        if elem in pos_only and elem not in res:
            if is_kwargs:
                res[str(kwargs_name)][elem] = kwargs[elem]
            else:
                raise TypeError(ERR_POSONLY_PASSED_AS_KW)

        elif elem in res:
            if not is_kwargs:
                if elem in pos_only:
                    raise TypeError(ERR_POSONLY_PASSED_AS_KW)
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
            res[str(kwargs_name)][elem] = kwargs[elem]
            if is_kwargs and elem in pos and elem not in pos_only:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)

        elif elem not in kw and elem not in pos:
            if not is_kwargs:
                raise TypeError(ERR_TOO_MANY_KW_ARGS)
            res[str(kwargs_name)][elem] = kwargs[elem]
        else:
            res[elem] = kwargs[elem]

    if kwdflts:
        for kwarg in kwdflts:
            if kwarg not in res:
                res[kwarg] = kwdflts[kwarg]
    if arg_non_def >= 0 and dflts and func.co_varnames:
        for i in range(arg_non_def, arg_count):
            if func.co_varnames[i] not in res:
                res[func.co_varnames[i]] = dflts[i - arg_non_def]
    for elem in list(kwonly):
        if elem not in res:
            raise TypeError(ERR_MISSING_KWONLY_ARGS)
    for elem in pos:
        if elem not in res:
            raise TypeError(ERR_MISSING_POS_ARGS)

    return res
