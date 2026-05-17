"""
C codegen for CPU backend — walks transformed IR and emits scalar C with OpenMP.

Tensors become flat row-major pointers. Tiles become stack-allocated 2D arrays
(VLAs for runtime shapes). Elementwise tile ops are vectorized via polymorphic
vector types (FP32Vec1 / FP32Vec4 / FP32Vec8 / FP32Vec16).
"""

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    Abs,
    Add,
    And,
    AssignStmt,
    BinaryExpr,
    Call,
    ConstBool,
    ConstFloat,
    ConstInt,
    Eq,
    EvalStmt,
    Expr,
    FloorDiv,
    FloatDiv,
    ForKind,
    ForStmt,
    Function,
    Ge,
    Gt,
    IfStmt,
    InCoreScopeStmt,
    InlineStmt,
    Le,
    Lt,
    MakeTuple,
    Mul,
    Ne,
    Neg,
    Not,
    Or,
    ParamDirection,
    ReturnStmt,
    ScopeStmt,
    SeqStmts,
    Sub,
    UnaryExpr,
    Var,
    WhileStmt,
    YieldStmt,
)


# ── Type mapping ────────────────────────────────────────────────────────────

def _dtype_to_c(dtype: DataType | str) -> str:
    """Map a DataType to its C scalar type."""
    s = str(dtype).lower()
    mapping: dict[str, str] = {
        "fp32": "float",
        "fp64": "double",
        "fp16": "uint16_t",
        "bfloat16": "uint16_t",
        "bf16": "uint16_t",
        "int8": "int8_t",
        "int16": "int16_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "uint8": "uint8_t",
        "uint16": "uint16_t",
        "uint32": "uint32_t",
        "uint64": "uint64_t",
        "bool": "bool",
        "index": "int64_t",
        "float32": "float",
    }
    return mapping.get(s, "float")


def _c_type_for(expr: Expr) -> str:
    """Return the C type for an IR expression."""
    t = expr.type
    if hasattr(t, "dtype"):
        return _dtype_to_c(t.dtype)
    return "float"


def _is_shaped(expr: Expr) -> bool:
    """Check whether *expr* has a shaped type (tensor or tile)."""
    return hasattr(expr.type, "shape") and hasattr(expr.type, "dtype")


def _get_shape(expr: Expr) -> list[int | str]:
    """Resolve the shape of a shaped expression.

    Returns a list of ints (for compile-time constants) or strings (for
    runtime expressions that can't be resolved to a constant).
    """
    t = expr.type
    if not hasattr(t, "shape"):
        return []
    shape = []
    for d in t.shape:
        try:
            shape.append(int(str(d)))
        except (ValueError, TypeError):
            try:
                val = _eval_const(d)
                shape.append(val)
            except (TypeError, ValueError):
                shape.append(f"/*dynamic*/{_sanitize_name(str(d))}")
    return shape


def _eval_const(expr: Expr) -> int:
    """Evaluate a compile-time constant expression to an integer."""
    if isinstance(expr, ConstInt):
        return int(expr.value)
    if isinstance(expr, ConstFloat):
        return int(expr.value)
    raise TypeError(f"Expected constant expression, got {type(expr).__name__}")


# ── C codegen ───────────────────────────────────────────────────────────────


class CCodegen:
    """Walk a transformed IR Function and emit C source text."""

    # Exact-match tile ops → handler method name
    _TILE_OP_DISPATCH: dict[str, str] = {
        "tile.create": "_cg_tile_create_assign",
        "tile.full": "_cg_tile_full_assign",
        "tile.load": "_cg_tile_load_assign",
        "tile.store": "_cg_tile_store_assign",
        "tile.matmul": "_cg_matmul_assign",
        "tile.matmul_acc": "_cg_matmul_acc_assign",
        "tile.gemv": "_cg_structured_op_assign",
        "tile.reshape": "_cg_structured_op_assign",
        "tile.transpose": "_cg_structured_op_assign",
        "tile.cast": "_cg_structured_op_assign",
        "tile.sum": "_cg_structured_op_assign",
        "tile.max": "_cg_structured_op_assign",
    }

    def __init__(self, vec_type: str = "FP32Vec1", tile_type: str = "ScalarTile") -> None:
        self._vec = vec_type
        self._tile = tile_type
        self._lines: list[str] = []
        self._indent = 0
        self._temp_counter = 0
        # var name_hint → C variable name
        self._var_map: dict[str, str] = {}
        # var name_hint → C type (for shaped vars)
        self._var_types: dict[str, str] = {}
        # var name_hint → shapes (list of dim expressions)
        self._var_shapes: dict[str, list[int | str]] = {}

    # ── top-level entry ────────────────────────────────────────────────

    def generate_function(self, func: Function) -> str:
        """Generate a complete C source file for *func*."""
        self._lines = []
        self._indent = 0
        self._temp_counter = 0
        self._var_map = {}
        self._var_types = {}
        self._var_shapes = {}

        self._emit("#include <stdint.h>")
        self._emit("#include <stdbool.h>")
        self._emit("#include <stdlib.h>")
        self._emit("#include <string.h>")
        self._emit("#include <cmath>")
        self._emit("#include <omp.h>")
        self._emit(f'#include "cpu_vec_types.h"')
        self._emit(f'#include "cpu_tensor_types.h"')
        self._emit("")

        # Register parameter variables
        tensor_params: list[tuple[str, str, list[int | str]]] = []
        scalar_params: list[tuple[str, str]] = []
        out_params: list[tuple[str, str, list[int | str]]] = []

        params = func.params
        param_dirs = func.param_directions if hasattr(func, "param_directions") else []
        for i, param in enumerate(params):
            c_name = f"arg_{_sanitize_name(param.name_hint)}"
            direction = param_dirs[i] if i < len(param_dirs) else ParamDirection.In
            if _is_shaped(param):
                shape = _get_shape(param)
                ctype = _dtype_to_c(param.type.dtype)
                if direction == ParamDirection.Out:
                    out_params.append((c_name, ctype, shape))
                else:
                    tensor_params.append((c_name, ctype, shape))
                self._register_var(param, c_name)
                self._var_types[self._var_key(param)] = f"{ctype}*"
                self._var_shapes[self._var_key(param)] = shape
            else:
                ctype = _c_type_for(param)
                scalar_params.append((c_name, ctype))
                self._register_var(param, c_name)

        # Function signature
        sig_parts: list[str] = []
        for name, ctype, shape in tensor_params:
            sig_parts.append(f"{ctype}* {name}")
            for dim_i, dim_v in enumerate(shape):
                sig_parts.append(f"int64_t {name}_d{dim_i}")
        for name, ctype in scalar_params:
            sig_parts.append(f"{ctype} {name}")
        for name, ctype, shape in out_params:
            sig_parts.append(f"{ctype}* {name}")
            for dim_i, dim_v in enumerate(shape):
                sig_parts.append(f"int64_t {name}_d{dim_i}")

        sig = ", ".join(sig_parts)
        self._emit('extern "C" {')
        self._emit(f"void {_sanitize_name(func.name)}({sig}) {{")
        self._indent = 1

        # Walk the function body
        self._gen_stmt(func.body)

        self._indent = 0
        self._emit("}")
        self._emit("}")  # close extern "C"
        return "\n".join(self._lines)

    # ── helpers ────────────────────────────────────────────────────────

    def _emit(self, line: str) -> None:
        if line:
            self._lines.append("    " * self._indent + line)
        else:
            self._lines.append("")

    def _tmp(self, prefix: str = "t") -> str:
        self._temp_counter += 1
        return f"{prefix}_{self._temp_counter}"

    def _register_var(self, var: Var, c_name: str) -> None:
        self._var_map[self._var_key(var)] = c_name

    def _var_key(self, var: Var) -> str:
        """Return a stable key for *var* (uses name_hint, not id, because IR
        passes reconstruct Var objects)."""
        return var.name_hint

    def _var_name(self, expr: Expr) -> str:
        """Look up the C variable name for an IR variable."""
        if isinstance(expr, Var):
            name = self._var_map.get(self._var_key(expr))
            if name is not None:
                return name
            return _sanitize_name(expr.name_hint)
        raise TypeError(f"Expected Var, got {type(expr).__name__}")

    def _tile_element_type(self, expr: Expr) -> str:
        """Get the C element type for a tile expression."""
        if hasattr(expr.type, "dtype"):
            return _dtype_to_c(expr.type.dtype)
        return "float"

    # ── statement dispatch ─────────────────────────────────────────────

    def _gen_stmt(self, stmt) -> None:
        if stmt is None:
            return
        if isinstance(stmt, SeqStmts):
            self._gen_seq_stmts(stmt)
        elif isinstance(stmt, InCoreScopeStmt):
            self._emit("#pragma omp parallel")
            self._emit("{")
            self._indent += 1
            self._gen_stmt(stmt.body)
            self._indent -= 1
            self._emit("}")
        elif isinstance(stmt, AssignStmt):
            self._gen_assign(stmt)
        elif isinstance(stmt, ForStmt):
            self._gen_for(stmt)
        elif isinstance(stmt, IfStmt):
            self._gen_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._gen_while(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._gen_return(stmt)
        elif isinstance(stmt, ScopeStmt):
            self._emit("{")
            self._indent += 1
            self._gen_stmt(stmt.body)
            self._indent -= 1
            self._emit("}")
        elif isinstance(stmt, YieldStmt):
            pass  # Handled inline in loops
        elif isinstance(stmt, EvalStmt):
            self._emit(f"{self._gen_expr(stmt.expr)};")
        elif isinstance(stmt, InlineStmt):
            pass  # Skip inline stmts
        else:
            self._emit(f"/* unhandled stmt: {type(stmt).__name__} */")

    def _gen_seq_stmts(self, seq: SeqStmts) -> None:
        """Emit a sequence of statements, grouping adjacent InCoreScopeStmts
        into #pragma omp parallel sections for MPMD parallelism."""
        stmts = seq.stmts
        i = 0
        while i < len(stmts):
            s = stmts[i]
            if isinstance(s, InCoreScopeStmt):
                # Collect consecutive InCoreScopeStmts
                incore_stmts = [s]
                j = i + 1
                while j < len(stmts) and isinstance(stmts[j], InCoreScopeStmt):
                    incore_stmts.append(stmts[j])
                    j += 1
                if len(incore_stmts) >= 2:
                    self._emit("#pragma omp parallel sections")
                    self._emit("{")
                    self._indent += 1
                    for ics in incore_stmts:
                        self._emit("#pragma omp section")
                        self._emit("{")
                        self._indent += 1
                        self._gen_stmt(ics.body)
                        self._indent -= 1
                        self._emit("}")
                    self._indent -= 1
                    self._emit("}")
                else:
                    # Single InCoreScopeStmt — still use omp parallel
                    self._emit("#pragma omp parallel")
                    self._emit("{")
                    self._indent += 1
                    self._gen_stmt(incore_stmts[0].body)
                    self._indent -= 1
                    self._emit("}")
                i = j
            else:
                self._gen_stmt(s)
                i += 1

    # ── assignment ─────────────────────────────────────────────────────

    def _gen_assign(self, stmt: AssignStmt) -> None:
        value = stmt.value
        var = stmt.var

        if isinstance(value, Call):
            self._gen_tile_op_assign(var, value)
        elif _is_shaped(var) and isinstance(value, Var):
            # Tile aliasing — just register the var
            self._register_var(var, self._var_name(value))
        else:
            c_name = self._tmp("v")
            self._register_var(var, c_name)
            ctype = _c_type_for(var)
            expr_c = self._gen_expr(value)
            self._emit(f"{ctype} {c_name} = {expr_c};")

    # ── for / while / if ───────────────────────────────────────────────

    def _emit_iter_arg_inits(self, iter_args: list) -> None:
        """Emit declarations and initializers for loop-carried iter_args."""
        for ia in iter_args:
            ia_name = self._tmp("ia")
            self._register_var(ia, ia_name)
            ctype = _c_type_for(ia)
            init_val = self._gen_expr(ia.initValue) if ia.initValue is not None else "0"
            self._emit(f"{ctype} {ia_name} = {init_val};")

    def _resolve_offsets(self, args: list, idx: int) -> list:
        """Extract constant offsets from call args[idx], defaulting to [0,0]."""
        offset_elems = _extract_tuple_elements(args[idx]) if len(args) > idx else []
        try:
            return [_eval_const(e) for e in offset_elems]
        except TypeError:
            return [0, 0]

    def _gen_for(self, stmt: ForStmt) -> None:
        loop_var = stmt.loop_var
        start = self._gen_expr(stmt.start)
        stop = self._gen_expr(stmt.stop)
        step = self._gen_expr(stmt.step)
        lv_name = self._tmp("i")
        self._register_var(loop_var, lv_name)

        self._emit_iter_arg_inits(stmt.iter_args)

        kind = stmt.kind
        has_omp = kind == ForKind.Parallel
        if has_omp:
            self._emit("#pragma omp parallel for")

        self._emit(f"for (int64_t {lv_name} = {start}; {lv_name} < {stop}; {lv_name} += {step}) {{")
        self._indent += 1
        self._gen_stmt(stmt.body)
        self._indent -= 1
        self._emit("}")

        # Capture return_vars
        for i, rv in enumerate(stmt.return_vars):
            if i < len(stmt.iter_args):
                ia = stmt.iter_args[i]
                ia_name = self._var_map.get(self._var_key(ia), f"ia_{i}")
                rv_name = self._tmp("rv")
                self._register_var(rv, rv_name)
                ctype = _c_type_for(rv)
                self._emit(f"{ctype} {rv_name} = {ia_name};")

    def _gen_if(self, stmt: IfStmt) -> None:
        cond = self._gen_expr(stmt.condition)
        self._emit(f"if ({cond}) {{")
        self._indent += 1
        self._gen_stmt(stmt.then_body)
        self._indent -= 1
        if stmt.else_body is not None:
            self._emit("} else {")
            self._indent += 1
            self._gen_stmt(stmt.else_body)
            self._indent -= 1
        self._emit("}")

    def _gen_while(self, stmt: WhileStmt) -> None:
        self._emit_iter_arg_inits(stmt.iter_args)

        cond = self._gen_expr(stmt.condition)
        self._emit(f"while ({cond}) {{")
        self._indent += 1
        self._gen_stmt(stmt.body)
        self._indent -= 1
        self._emit("}")

    def _gen_return(self, stmt: ReturnStmt) -> None:
        # On CPU, return values are written to output tensor parameters.
        # A simple return is sufficient; outputs are written in-place during
        # tile.store.
        if stmt.value:
            for v in stmt.value:
                # Evaluate expression for side effects (e.g. tile.store calls)
                self._gen_expr(v)
        self._emit("return;")

    # ── expression dispatch ────────────────────────────────────────────

    def _gen_expr(self, expr: Expr) -> str:
        if isinstance(expr, ConstInt):
            return str(int(expr.value))
        if isinstance(expr, ConstFloat):
            return str(float(expr.value))
        if isinstance(expr, ConstBool):
            return "true" if expr.value else "false"
        if isinstance(expr, Var):
            return self._var_name(expr)
        if isinstance(expr, Call):
            return self._gen_call(expr)
        if isinstance(expr, BinaryExpr):
            return self._gen_binary(expr)
        if isinstance(expr, UnaryExpr):
            return self._gen_unary(expr)
        return f"/* unhandled expr: {type(expr).__name__} */"

    def _gen_binary(self, expr: BinaryExpr) -> str:
        left = self._gen_expr(expr.left)
        right = self._gen_expr(expr.right)
        op_map = {
            Add: "+",
            Sub: "-",
            Mul: "*",
            FloorDiv: "/",
            FloatDiv: "/",
            Eq: "==",
            Ne: "!=",
            Lt: "<",
            Le: "<=",
            Gt: ">",
            Ge: ">=",
            And: "&&",
            Or: "||",
        }
        op = op_map.get(type(expr), f"/*{type(expr).__name__}*/")
        return f"({left} {op} {right})"

    def _gen_unary(self, expr: UnaryExpr) -> str:
        operand = self._gen_expr(expr.operand)
        if isinstance(expr, Neg):
            return f"(-({operand}))"
        if isinstance(expr, Not):
            return f"(!({operand}))"
        if isinstance(expr, Abs):
            return f"fabsf({operand})"
        return f"/* unhandled unary: {type(expr).__name__} */({operand})"

    # ── call dispatch ──────────────────────────────────────────────────

    def _gen_call(self, call: Call) -> str:
        op_name = call.op.name

        # Fallback for unknown ops — generate comment + return 0
        handler = getattr(self, f"_cg_{_op_to_method(op_name)}", None)
        if handler is not None:
            result = handler(call)
            # Structured op handlers return (name, shapes); expression handlers return str
            if isinstance(result, tuple):
                return result[0]
            return result
        self._emit(f"/* unhandled op: {op_name} */")
        return "0"

    # ── tile op assignment ─────────────────────────────────────────────

    def _gen_tile_op_assign(self, var: Var, call: Call) -> None:
        """Handle a tile operation that produces a result bound to *var*."""
        op_name = call.op.name

        handler_name = self._TILE_OP_DISPATCH.get(op_name)
        if handler_name is not None:
            getattr(self, handler_name)(var, call)
        elif op_name.startswith("tile."):
            self._cg_elementwise_assign(var, call)
        elif op_name.startswith("tensor."):
            self._cg_tensor_op_assign(var, call)
        else:
            c_name = self._tmp("v")
            self._register_var(var, c_name)
            result = self._gen_call(call)
            ctype = _c_type_for(var)
            self._emit(f"{ctype} {c_name} = {result};")

    # ── structured tile op dispatch ──────────────────────────────────────

    def _cg_structured_op_assign(self, var: Var, call: Call) -> None:
        """Route non-elementwise tile ops (gemv, transpose, cast, reduce) to
        their dedicated handlers."""
        op_name = call.op.name
        handler_name = "_cg_" + _op_to_method(op_name)
        handler = getattr(self, handler_name)
        result_name, shapes = handler(call)
        self._register_var(var, result_name)
        key = self._var_key(var)
        self._var_shapes[key] = shapes
        self._var_types[key] = self._tile_element_type(call)

    # ── tile.create ────────────────────────────────────────────────────

    def _cg_tile_create_assign(self, var: Var, call: Call) -> None:
        shape = _get_shape(call)
        c_name = self._tmp("tile")
        self._register_var(var, c_name)

        if len(shape) == 2:
            M, N = shape[0], shape[1]
            self._emit(f"{self._tile}<{M}, {N}> {c_name};")
            self._emit(f"{c_name}.zero();")
        elif len(shape) == 1:
            N = shape[0]
            self._emit(f"{self._tile}<1, {N}> {c_name};")
            self._emit(f"{c_name}.zero();")
        else:
            self._emit(f"/* tile.create: unsupported rank {len(shape)} */")

        self._var_shapes[self._var_key(var)] = shape

    # ── tile.full ──────────────────────────────────────────────────────

    def _cg_tile_full_assign(self, var: Var, call: Call) -> None:
        shape = _get_shape(call)
        c_name = self._tmp("tile")
        self._register_var(var, c_name)

        value = "0.0f"
        if len(call.args) >= 2:
            value = self._gen_expr(call.args[1])
        elif "value" in (call.kwargs or {}):
            value = str(call.kwargs["value"])

        if len(shape) == 2:
            M, N = shape[0], shape[1]
            self._emit(f"{self._tile}<{M}, {N}> {c_name};")
            self._emit(f"{c_name}.fill({value});")

        self._var_shapes[self._var_key(var)] = shape

    # ── tile.matmul / matmul_acc ───────────────────────────────────────

    def _cg_matmul_assign(self, var: Var, call: Call) -> None:
        args = call.args
        a_name = self._var_name(args[0])
        b_name = self._var_name(args[1])
        a_shape = self._var_shapes.get(self._var_key(args[0]), ["M", "K"])
        b_shape = self._var_shapes.get(self._var_key(args[1]), ["K", "N"])
        c_name = self._tmp("tile")
        self._register_var(var, c_name)

        M, K = a_shape[0], a_shape[1]
        N = b_shape[1]

        self._emit(f"{self._tile}<{M}, {N}> {c_name};")
        self._emit(f"{c_name}.matmul({a_name}, {b_name});")
        self._var_shapes[self._var_key(var)] = [M, N]

    def _cg_matmul_acc_assign(self, var: Var, call: Call) -> None:
        args = call.args
        acc_name = self._var_name(args[0])
        a_name = self._var_name(args[1])
        b_name = self._var_name(args[2])
        self._emit(f"{acc_name}.matmul_acc({a_name}, {b_name});")
        self._register_var(var, acc_name)
        # Propagate accumulator's shape to the result var
        acc_key = self._var_key(args[0])
        self._var_shapes[self._var_key(var)] = self._var_shapes.get(acc_key, ["M", "N"])

    # ── tile elementwise ───────────────────────────────────────────────

    def _cg_elementwise_assign(self, var: Var, call: Call) -> None:
        """Handle elementwise tile ops: add, sub, mul, div, exp, sqrt, etc."""
        op_name = call.op.name
        args = call.args
        result_shape = _get_shape(call)
        c_name = self._tmp("tile")
        ctype = self._tile_element_type(call)

        if len(result_shape) == 2:
            M, N = result_shape[0], result_shape[1]

            if op_name.endswith("s") and len(args) >= 2 and not _is_shaped(args[1]):
                # Binary with scalar second arg — broadcast scalar to vector
                scalar_val = self._gen_expr(args[1])
                tile_name = self._var_name(args[0])
                self._emit(f"{self._tile}<{M}, {N}> {c_name};")
                c_op = _binary_op_symbol(op_name)
                self._emit(f"for (int64_t _i = 0; _i < {M}; _i++)")
                self._emit(f"    for (int64_t _j = 0; _j < {N}; _j += {self._vec}::VEC_ELEM_NUM) {{")
                self._emit(f"        {self._vec} _vt(&{tile_name}.data[_i][_j]);")
                self._emit(f"        {self._vec} _vs({scalar_val});")
                if c_op.startswith("."):
                    self._emit(f"        {self._vec} _vr = _vt{c_op}(_vs);")
                else:
                    self._emit(f"        {self._vec} _vr = _vt {c_op} _vs;")
                self._emit(f"        _vr.save(&{c_name}.data[_i][_j]);")
                self._emit(f"    }}")
            elif _is_unary_tile_op(op_name):
                tile_name = self._var_name(args[0])
                self._emit(f"{self._tile}<{M}, {N}> {c_name};")
                member = _eltwise_member_name(op_name, unary=True)
                self._emit(f"for (int64_t _i = 0; _i < {M}; _i++)")
                self._emit(f"    for (int64_t _j = 0; _j < {N}; _j += {self._vec}::VEC_ELEM_NUM) {{")
                self._emit(f"        {self._vec} _vt(&{tile_name}.data[_i][_j]);")
                self._emit(f"        {self._vec} _vr = _vt.{member}();")
                self._emit(f"        _vr.save(&{c_name}.data[_i][_j]);")
                self._emit(f"    }}")
            elif len(args) >= 2 and _is_shaped(args[0]) and _is_shaped(args[1]):
                # Tile-tile binary
                a_name = self._var_name(args[0])
                b_name = self._var_name(args[1])
                self._emit(f"{self._tile}<{M}, {N}> {c_name};")
                c_op = _binary_op_symbol(op_name)
                self._emit(f"for (int64_t _i = 0; _i < {M}; _i++)")
                self._emit(f"    for (int64_t _j = 0; _j < {N}; _j += {self._vec}::VEC_ELEM_NUM) {{")
                self._emit(f"        {self._vec} _va(&{a_name}.data[_i][_j]);")
                self._emit(f"        {self._vec} _vb(&{b_name}.data[_i][_j]);")
                if c_op.startswith("."):
                    self._emit(f"        {self._vec} _vr = _va{c_op}(_vb);")
                else:
                    self._emit(f"        {self._vec} _vr = _va {c_op} _vb;")
                self._emit(f"        _vr.save(&{c_name}.data[_i][_j]);")
                self._emit(f"    }}")
            else:
                self._emit(f"/* elementwise: unhandled pattern for {op_name} */")

        self._register_var(var, c_name)
        self._var_types[self._var_key(var)] = ctype
        self._var_shapes[self._var_key(var)] = result_shape

    # ── tensor ops ─────────────────────────────────────────────────────

    def _cg_tensor_op_assign(self, var: Var, call: Call) -> None:
        op_name = call.op.name
        args = call.args
        c_name = self._tmp("tensor")
        self._register_var(var, c_name)
        ctype = self._tile_element_type(call)
        shape = _get_shape(call)

        if op_name == "tensor.create":
            total = " * ".join(str(d) for d in shape) if shape else "1"
            self._emit(f"{ctype}* {c_name} = ({ctype}*)calloc(({total}), sizeof({ctype}));")
        elif op_name == "tensor.full":
            total = " * ".join(str(d) for d in shape) if shape else "1"
            value = self._gen_expr(args[1]) if len(args) >= 2 else "0.0f"
            self._emit(f"{ctype}* {c_name} = ({ctype}*)malloc(({total}) * sizeof({ctype}));")
            self._emit(f"for (int64_t _i = 0; _i < ({total}); _i++) {c_name}[_i] = {value};")
        else:
            self._emit(f"/* tensor op {op_name} */")

        self._var_types[self._var_key(var)] = f"{ctype}*"
        self._var_shapes[self._var_key(var)] = shape

    # ── tile.load / tile.store (assign variants) ───────────────────────

    def _cg_tile_load_assign(self, var: Var, call: Call) -> None:
        """tile.load produces a tile bound to *var*."""
        result_name, shapes = self._cg_tile_load(call)
        self._register_var(var, result_name)
        self._var_shapes[self._var_key(var)] = shapes

    def _cg_tile_store_assign(self, var: Var, call: Call) -> None:
        """tile.store: side effect + register result var."""
        result_name = self._cg_tile_store(call)
        self._register_var(var, result_name)

    # ── tile.load / tile.store ─────────────────────────────────────────

    def _cg_tile_load(self, call: Call) -> tuple[str, list]:
        args = call.args
        tensor_name = self._var_name(args[0])

        offsets = self._resolve_offsets(args, 1)
        shape_elems = _extract_tuple_elements(args[2]) if len(args) >= 3 else []
        try:
            shapes = [_eval_const(e) for e in shape_elems]
        except TypeError:
            shapes = [1, 1]

        result_name = self._tmp("tile")
        tensor_shape = self._var_shapes.get(self._var_key(args[0]), [1])

        if len(offsets) >= 2:
            off0, off1 = offsets[0], offsets[1]
        else:
            off0, off1 = 0, 0

        if len(shapes) == 2:
            M, N = shapes[0], shapes[1]
            stride = tensor_shape[1] if len(tensor_shape) >= 2 else 1
            base_off = f"({off0}) * ((int64_t)({stride})) + ({off1})"
            self._emit(f"{self._tile}<{M}, {N}> {result_name};")
            self._emit(f"{result_name}.load(&{tensor_name}[{base_off}], {stride});")

        return result_name, shapes

    def _cg_tile_store(self, call: Call) -> str:
        args = call.args
        tile_name = self._var_name(args[0])

        offsets = self._resolve_offsets(args, 1)
        tensor_name = self._var_name(args[2])

        tensor_shape = self._var_shapes.get(self._var_key(args[2]), [1])
        stride = tensor_shape[1] if len(tensor_shape) >= 2 else 1

        if len(offsets) >= 2:
            off0, off1 = offsets[0], offsets[1]
        else:
            off0, off1 = 0, 0

        base_off = f"({off0}) * ((int64_t)({stride})) + ({off1})"
        self._emit(f"{tile_name}.store(&{tensor_name}[{base_off}], {stride});")
        return tensor_name

    def _cg_tile_gemv(self, call: Call) -> tuple[str, list]:
        args = call.args
        a_name = self._var_name(args[0])
        b_name = self._var_name(args[1])
        result_name = self._tmp("tile")
        ctype = self._tile_element_type(call)
        a_shape = self._var_shapes.get(self._var_key(args[0]), [1, "K"])
        b_shape = self._var_shapes.get(self._var_key(args[1]), ["K", "N"])

        K, N = a_shape[1], b_shape[1]
        self._emit(f"{self._tile}<1, {N}> {result_name};")
        self._emit(f"for (int64_t _j = 0; _j < {N}; _j += {self._vec}::VEC_ELEM_NUM) {{")
        self._indent += 1
        self._emit(f"{self._vec} _vacc(0.0f);")
        self._emit(f"for (int64_t _k = 0; _k < {K}; _k++) {{")
        self._indent += 1
        self._emit(f"{self._vec} _va({a_name}.data[0][_k]);")
        self._emit(f"{self._vec} _vb(&{b_name}.data[_k][_j]);")
        self._emit(f"v_fma(_vacc, _va, _vb);")
        self._indent -= 1
        self._emit("}")
        self._emit(f"_vacc.save(&{result_name}.data[0][_j]);")
        self._indent -= 1
        self._emit("}")
        return result_name, [1, N]

    def _cg_tile_reshape(self, call: Call) -> tuple[str, list]:
        src_key = self._var_key(call.args[0])
        new_shape = _resolve_shape_from_args(call, idx=1)
        if new_shape and len(new_shape) == 2:
            src_name = self._var_name(call.args[0])
            result_name = self._tmp("tile")
            M, N = new_shape[0], new_shape[1]
            self._emit(f"{self._tile}<{M}, {N}> {result_name};")
            self._emit(f"memcpy(&{result_name}.data[0][0], &{src_name}.data[0][0], "
                       f"{M} * {N} * sizeof(float));")
            return result_name, new_shape
        return self._var_name(call.args[0]), self._var_shapes.get(src_key, [1, 1])

    def _cg_tile_transpose(self, call: Call) -> tuple[str, list]:
        args = call.args
        tile_name = self._var_name(args[0])
        tile_shape = self._var_shapes.get(self._var_key(args[0]), [1, 1])
        result_name = self._tmp("tile")
        result_shape = tile_shape

        if len(tile_shape) == 2:
            M, N = tile_shape[0], tile_shape[1]
            result_shape = [N, M]
            self._emit(f"{self._tile}<{N}, {M}> {result_name};")
            self._emit(f"for (int64_t _i = 0; _i < {M}; _i++)")
            self._emit(f"    for (int64_t _j = 0; _j < {N}; _j++)")
            self._emit(f"        {result_name}.data[_j][_i] = {tile_name}.data[_i][_j];")

        return result_name, result_shape

    def _cg_tile_cast(self, call: Call) -> tuple[str, list]:
        args = call.args
        tile_name = self._var_name(args[0])
        result_name = self._tmp("tile")
        ctype = self._tile_element_type(call)
        shape = _get_shape(args[0]) if _is_shaped(args[0]) else [1, 1]

        if len(shape) == 2:
            M, N = shape[0], shape[1]
            self._emit(f"{self._tile}<{M}, {N}> {result_name};")
            self._emit(f"for (int64_t _i = 0; _i < {M}; _i++)")
            self._emit(f"    for (int64_t _j = 0; _j < {N}; _j++)")
            self._emit(f"        {result_name}.data[_i][_j] = ({ctype})({tile_name}.data[_i][_j]);")

        return result_name, shape

    def _cg_tile_sum(self, call: Call) -> tuple[str, list]:
        return self._cg_tile_reduce(call, "+=", "0.0f")

    def _cg_tile_max(self, call: Call) -> tuple[str, list]:
        args = call.args
        tile_name = self._var_name(args[0])
        axis = _extract_axis(call)
        shape = _get_shape(args[0])
        result_name = self._tmp("tile")
        ctype = self._tile_element_type(call)
        result_shape: list = shape

        if axis == 1 and len(shape) == 2:
            M, N = shape[0], shape[1]
            result_shape = [M, 1]
            self._emit(f"{self._tile}<{M}, 1> {result_name};")
            self._emit(f"for (int64_t _i = 0; _i < {M}; _i++) {{")
            self._emit(f"    {ctype} _m = {tile_name}.data[_i][0];")
            self._emit(f"    for (int64_t _j = 1; _j < {N}; _j++)")
            self._emit(f"        _m = fmaxf(_m, {tile_name}.data[_i][_j]);")
            self._emit(f"    {result_name}.data[_i][0] = _m;")
            self._emit(f"}}")
        return result_name, result_shape

    def _cg_tile_reduce(self, call: Call, reduce_op: str, init_val: str) -> tuple[str, list]:
        args = call.args
        tile_name = self._var_name(args[0])
        axis = _extract_axis(call)
        shape = _get_shape(args[0])
        result_name = self._tmp("tile")
        ctype = self._tile_element_type(call)
        result_shape: list = shape

        if axis == 1 and len(shape) == 2:
            M, N = shape[0], shape[1]
            result_shape = [M, 1]
            self._emit(f"{self._tile}<{M}, 1> {result_name};")
            self._emit(f"for (int64_t _i = 0; _i < {M}; _i++) {{")
            self._emit(f"    {ctype} _acc = {init_val};")
            self._emit(f"    for (int64_t _j = 0; _j < {N}; _j++)")
            self._emit(f"        _acc {reduce_op} {tile_name}.data[_i][_j];")
            self._emit(f"    {result_name}.data[_i][0] = _acc;")
            self._emit(f"}}")
        elif axis == 0 and len(shape) == 2:
            M, N = shape[0], shape[1]
            result_shape = [1, N]
            self._emit(f"{self._tile}<1, {N}> {result_name};")
            self._emit(f"for (int64_t _j = 0; _j < {N}; _j++) {{")
            self._emit(f"    {ctype} _acc = {init_val};")
            self._emit(f"    for (int64_t _i = 0; _i < {M}; _i++)")
            self._emit(f"        _acc {reduce_op} {tile_name}.data[_i][_j];")
            self._emit(f"    {result_name}.data[0][_j] = _acc;")
            self._emit(f"}}")
        return result_name, result_shape


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _op_to_method(op_name: str) -> str:
    """Convert 'tile.matmul' → 'tile_matmul'."""
    return op_name.replace(".", "_")


def _binary_op_symbol(op_name: str) -> str:
    """Map tile binary op name to C++ operator symbol for vector types."""
    base = op_name.split(".")[-1].removesuffix("s")
    mapping = {
        "add": "+", "sub": "-", "mul": "*", "div": "/",
    }
    if base in ("max", "maximum"):
        return ".max"
    if base in ("min", "minimum"):
        return ".min"
    return mapping.get(base, "+")


def _is_unary_tile_op(op_name: str) -> bool:
    """Check if *op_name* is a unary tile elementwise op."""
    unary = {
        "tile.neg", "tile.abs", "tile.exp", "tile.sqrt", "tile.rsqrt",
        "tile.recip", "tile.log", "tile.sin", "tile.cos", "tile.relu",
        "tile.not_",
    }
    return op_name in unary


def _eltwise_member_name(op_name: str, unary: bool) -> str:
    """Map tile op name to vector type member function name."""
    base = op_name.split(".")[-1]
    if not unary:
        base = base.removesuffix("s")
    unary_map = {
        "neg": "neg", "abs": "abs", "exp": "exp", "sqrt": "sqrt",
        "rsqrt": "rsqrt", "recip": "recip", "log": "log", "sin": "sin",
        "cos": "cos", "relu": "relu", "not_": "not_",
    }
    binary_map = {
        "add": "operator+", "sub": "operator-", "mul": "operator*",
        "div": "operator/", "max": "max", "min": "min",
        "maximum": "max", "minimum": "min",
    }
    if unary:
        return unary_map.get(base, base)
    return binary_map.get(base, "operator+")


def _extract_axis(call: Call) -> int:
    """Extract the axis argument from a Call, checking kwargs first then args."""
    if call.kwargs and "axis" in call.kwargs:
        ax = call.kwargs["axis"]
        if isinstance(ax, int):
            return ax
        try:
            return int(_eval_const(ax))
        except (TypeError, ValueError):
            pass
    if len(call.args) >= 2:
        try:
            return int(_eval_const(call.args[1]))
        except (TypeError, ValueError):
            pass
    return -1


def _resolve_shape_from_args(call: Call, idx: int = 1) -> list[int] | None:
    """Extract a shape from call args[idx] or kwargs['shape']."""
    if call.kwargs and "shape" in call.kwargs:
        shape_expr = call.kwargs["shape"]
        try:
            return [_eval_const(e) for e in _extract_tuple_elements(shape_expr)]
        except TypeError:
            pass
    if len(call.args) > idx:
        try:
            return [_eval_const(e) for e in _extract_tuple_elements(call.args[idx])]
        except TypeError:
            pass
    return None


def _extract_tuple_elements(expr) -> list:
    """Extract elements from a MakeTuple or list expression."""
    if isinstance(expr, MakeTuple):
        return list(expr.elements)
    if hasattr(expr, "__getitem__") and hasattr(expr, "__len__"):
        try:
            return [expr[i] for i in range(len(expr))]
        except (TypeError, IndexError):
            pass
    return []

