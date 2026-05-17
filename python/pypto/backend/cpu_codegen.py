"""
C codegen for CPU backend — walks transformed IR and emits scalar C with OpenMP.

Tensors become flat row-major pointers. Tiles become stack-allocated 2D arrays.
Elementwise tile ops are vectorized via polymorphic vector types
(FP32Vec1 / FP32Vec4 / FP32Vec8 / FP32Vec16).
"""

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    AssignStmt,
    Call,
    ConstFloat,
    ConstInt,
    Expr,
    ForStmt,
    Function,
    IfStmt,
    MakeTuple,
    ParamDirection,
    ReturnStmt,
    ScopeStmt,
    SeqStmts,
    Var,
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
    """Resolve the shape of a shaped expression."""
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

    def __init__(self, vec_type: str = "FP32Vec1", tile_type: str = "ScalarTile") -> None:
        self._vec = vec_type
        self._tile = tile_type
        self._lines: list[str] = []
        self._indent = 0
        self._temp_counter = 0
        self._var_map: dict[str, str] = {}
        self._var_types: dict[str, str] = {}
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
        self._emit('#include "cpu_vec_types.h"')
        self._emit('#include "cpu_tensor_types.h"')
        self._emit("")

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

        self._gen_stmt(func.body)

        self._indent = 0
        self._emit("}")
        self._emit("}")
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
        return var.name_hint

    def _var_name(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            name = self._var_map.get(self._var_key(expr))
            if name is not None:
                return name
            return _sanitize_name(expr.name_hint)
        raise TypeError(f"Expected Var, got {type(expr).__name__}")

    def _tile_element_type(self, expr: Expr) -> str:
        if hasattr(expr.type, "dtype"):
            return _dtype_to_c(expr.type.dtype)
        return "float"

    def _resolve_offsets(self, args: list, idx: int) -> list:
        offset_elems = _extract_tuple_elements(args[idx]) if len(args) > idx else []
        try:
            return [_eval_const(e) for e in offset_elems]
        except TypeError:
            return [0, 0]

    # ── expression evaluation ──────────────────────────────────────────

    def _gen_expr(self, expr: Expr) -> str:
        if isinstance(expr, ConstInt):
            return str(int(expr.value))
        if isinstance(expr, ConstFloat):
            return str(float(expr.value))
        if isinstance(expr, Var):
            return self._var_name(expr)
        if isinstance(expr, Call):
            return self._gen_call(expr)
        return f"/* unhandled expr: {type(expr).__name__} */"

    def _gen_call(self, call: Call) -> str:
        op_name = call.op.name
        if op_name == "tile.store":
            return self._cg_tile_store(call)
        self._emit(f"/* unhandled call: {op_name} */")
        return "0"

    # ── statement dispatch ─────────────────────────────────────────────

    def _gen_stmt(self, stmt) -> None:
        if stmt is None:
            return
        if isinstance(stmt, SeqStmts):
            for s in stmt.stmts:
                self._gen_stmt(s)
        elif isinstance(stmt, AssignStmt):
            self._gen_assign(stmt)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                for v in stmt.value:
                    self._gen_expr(v)
            self._emit("return;")
        elif isinstance(stmt, (ScopeStmt,)):
            self._emit("{")
            self._indent += 1
            self._gen_stmt(stmt.body)
            self._indent -= 1
            self._emit("}")
        elif isinstance(stmt, (ForStmt, IfStmt)):
            self._emit(f"/* {type(stmt).__name__}: deferred to PR 2 */")
        else:
            self._emit(f"/* unhandled stmt: {type(stmt).__name__} */")

    # ── assignment ─────────────────────────────────────────────────────

    def _gen_assign(self, stmt: AssignStmt) -> None:
        value = stmt.value
        var = stmt.var
        if isinstance(value, Call):
            op_name = value.op.name
            if op_name == "tile.load":
                self._cg_tile_load_assign(var, value)
            elif op_name == "tile.store":
                self._cg_tile_store_assign(var, value)
            else:
                self._emit(f"/* tile op {op_name}: deferred to PR 2 */")
        else:
            self._emit(f"/* unhandled assignment: {type(value).__name__} */")

    # ── tile.load / tile.store ─────────────────────────────────────────

    def _cg_tile_load_assign(self, var: Var, call: Call) -> None:
        result_name, shapes = self._cg_tile_load(call)
        self._register_var(var, result_name)
        self._var_shapes[self._var_key(var)] = shapes

    def _cg_tile_store_assign(self, var: Var, call: Call) -> None:
        result_name = self._cg_tile_store(call)
        self._register_var(var, result_name)

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _extract_tuple_elements(expr) -> list:
    if isinstance(expr, MakeTuple):
        return list(expr.elements)
    if hasattr(expr, "__getitem__") and hasattr(expr, "__len__"):
        try:
            return [expr[i] for i in range(len(expr))]
        except (TypeError, IndexError):
            pass
    return []
