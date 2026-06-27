import ast
import logging
from collections import defaultdict
from itertools import combinations

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)

MAX = 2_000_000_000


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

def get_variables(expression: str) -> list[str]:
    """Return sorted list of variable names found in an expression string."""
    expression = expression.strip()
    tree = ast.parse(expression, mode="eval")
    variables = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    return sorted(list(variables))


def validate_equation(expr: str) -> list[str]:
    """
    Return human-readable errors for unsupported AST constructs.
    Call before passing an equation to the solver.
    """
    errors = []
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp):
            op_name = type(node.op).__name__
            errors.append(
                f"Unary operator '{op_name}' not supported in '{expr}'. "
                f"Rewrite e.g. '-x' as '(0 - x)'."
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            right = node.right
            if not (
                isinstance(right, ast.Constant)
                and isinstance(right.value, int)
                and right.value >= 1
            ):
                errors.append(
                    f"Only constant positive integer exponents are supported in '**', "
                    f"got '{ast.unparse(right)}' in '{expr}'."
                )
    return errors


def _as_intvar(val, model: cp_model.CpModel, tag: str):
    """Wrap a plain Python int into a constant IntVar so OR-Tools accepts it."""
    if isinstance(val, int):
        return model.NewIntVar(val, val, tag)
    return val


# ---------------------------------------------------------------------------
# AST → OR-Tools expression builder
# ---------------------------------------------------------------------------

def build_expr(node, model: cp_model.CpModel, vars: dict, tag: str):
    if isinstance(node, ast.Constant):
        return int(node.value)

    if isinstance(node, ast.Name):
        return vars[node.id]

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            if not (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and node.right.value >= 1
            ):
                raise ValueError(
                    f"Only constant positive integer exponents supported, "
                    f"got: {ast.unparse(node.right)}"
                )
            base = build_expr(node.left, model, vars, tag)
            exp = node.right.value
            if exp == 1:
                return base
            base_var = _as_intvar(base, model, f"{tag}_pow_base")
            result = base_var
            for i in range(1, exp):
                new_result = model.NewIntVar(-MAX, MAX, f"{tag}_pow_{i}")
                model.AddMultiplicationEquality(new_result, [result, base_var])
                result = new_result
            return result

        left = build_expr(node.left, model, vars, tag)
        right = build_expr(node.right, model, vars, tag)

        if isinstance(node.op, ast.Add):
            out = model.NewIntVar(-MAX, MAX, f"{tag}_add")
            model.Add(out == left + right)
            return out

        if isinstance(node.op, ast.Sub):
            out = model.NewIntVar(-MAX, MAX, f"{tag}_sub")
            model.Add(out == left - right)
            return out

        if isinstance(node.op, ast.Mult):
            out = model.NewIntVar(-MAX, MAX, f"{tag}_mul")
            model.AddMultiplicationEquality(out, [left, right])
            return out

        if isinstance(node.op, ast.Div):
            out = model.NewIntVar(-MAX, MAX, f"{tag}_div")
            left_var = _as_intvar(left, model, f"{tag}_div_left")
            right_var = _as_intvar(right, model, f"{tag}_div_right")
            model.Add(right_var != 0)
            model.AddDivisionEquality(out, left_var, right_var)
            return out

        if isinstance(node.op, ast.Mod):
            out = model.NewIntVar(0, MAX, f"{tag}_mod")
            left_var = _as_intvar(left, model, f"{tag}_mod_left")
            right_var = _as_intvar(right, model, f"{tag}_mod_right")
            model.AddModuloEquality(out, left_var, right_var)
            return out

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Equation → RHS division elimination
# ---------------------------------------------------------------------------

def _extract_divisors(node, divisors: list):
    """Collect denominators from RHS and strip divisions, recursing left."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        divisors.append(node.right)
        return _extract_divisors(node.left, divisors)
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=_extract_divisors(node.left, divisors),
            op=node.op,
            right=_extract_divisors(node.right, divisors),
        )
    return node


def _multiply_lhs(lhs_ast, divisors: list):
    result = lhs_ast
    for d in divisors:
        result = ast.BinOp(left=result, op=ast.Mult(), right=d)
    return result


def move_rhs_divisions_to_lhs(lhs_expr: str, rhs_expr: str) -> tuple[str, str]:
    """
    Transform  lhs = a / b  →  lhs * b = a
    so the solver never has division on the RHS (avoids integer-division loss).
    """
    lhs_ast = ast.parse(lhs_expr, mode="eval").body
    rhs_ast = ast.parse(rhs_expr, mode="eval").body
    divisors: list = []
    new_rhs = _extract_divisors(rhs_ast, divisors)
    new_lhs = _multiply_lhs(lhs_ast, divisors) if divisors else lhs_ast
    return ast.unparse(new_lhs), ast.unparse(new_rhs)


# ---------------------------------------------------------------------------
# Model constraint builders
# ---------------------------------------------------------------------------

def add_hard_equation(model: cp_model.CpModel, vars: dict, eq: dict):
    lhs = build_expr(ast.parse(eq["lhs"], mode="eval").body, model, vars, f"{eq['lhs']}_lhs")
    rhs = build_expr(ast.parse(eq["rhs"], mode="eval").body, model, vars, f"{eq['lhs']}_rhs")
    rel = eq["relation"]
    if rel == "==":
        model.Add(lhs == rhs)
    elif rel == "<=":
        model.Add(lhs <= rhs)
    elif rel == ">=":
        model.Add(lhs >= rhs)
    else:
        raise ValueError(f"Unsupported relation: {rel}")


def add_soft_constraint(
    model: cp_model.CpModel, vars: dict, eq: dict, penalties: list
):
    lhs = build_expr(ast.parse(eq["lhs"].strip(), mode="eval").body, model, vars, f"{eq['lhs']}_lhs")
    rhs = build_expr(ast.parse(eq["rhs"].strip(), mode="eval").body, model, vars, f"{eq['lhs']}_rhs")

    slack = model.NewIntVar(0, MAX, f"slack_{eq['lhs']}")
    tol = eq.get("tolerance", 0)
    rel = eq["relation"]

    if rel == "==":
        diff = model.NewIntVar(-MAX, MAX, f"diff_{eq['lhs']}")
        model.Add(diff == lhs - rhs)
        abs_diff = model.NewIntVar(0, MAX, f"abs_diff_{eq['lhs']}")
        model.AddAbsEquality(abs_diff, diff)
        model.Add(slack >= abs_diff - tol)
        model.Add(slack <= abs_diff + tol)
    elif rel == "<=":
        model.Add(lhs - rhs <= slack + tol)
    elif rel == ">=":
        model.Add(rhs - lhs <= slack + tol)
    else:
        raise ValueError(f"Unsupported relation: {rel}")

    penalties.append(eq["weight"] * slack)


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

_STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN: "UNKNOWN",
}


def solver(
    var_specs: dict,
    equations: list[dict],
    timeout_seconds: float = 5,
    num_workers: int = 8,
) -> dict:
    """
    Run CP-SAT to find values for non-fixed variables that satisfy equations.

    var_specs: {
        "x": {
            "min": 0, "max": 1000000,
            "obs": 520,          # observed (extracted) value
            "confidence": 0.9,   # how much to trust obs (0–1)
            "mult_factor": 100,  # scale float→int (divide result back on exit)
            "fixed": False       # if True, pins the variable to obs
        }
    }
    equations: [
        {"lhs": "x", "rhs": "y * z", "relation": "==", "weight": 1000, "tolerance": 0}
    ]

    Returns:
        {
            "status": "OPTIMAL",
            "corrected": {"x": 5.2, "z": 3.0}   # only non-fixed vars
        }
    """
    # --- validate equations up front ---
    for eq in equations:
        for side in ("lhs", "rhs"):
            errs = validate_equation(eq[side])
            if errs:
                return {"status": "INVALID_EQUATION", "errors": errs, "field": eq[side]}

    model = cp_model.CpModel()
    cp_vars: dict = {}

    for k, spec in var_specs.items():
        if spec.get("fixed", False):
            cp_vars[k] = model.NewConstant(int(spec["obs"]))
        else:
            cp_vars[k] = model.NewIntVar(int(spec.get("min", 0)), int(spec.get("max", MAX)), k)

    penalties: list = []

    # observation penalties for non-fixed vars
    for k, spec in var_specs.items():
        if spec.get("fixed", False):
            continue
        if "obs" not in spec:
            continue
        diff = model.NewIntVar(-MAX, MAX, f"diff_{k}")
        model.Add(diff == cp_vars[k] - int(spec["obs"]))
        slack = model.NewIntVar(0, MAX, f"obs_slack_{k}")
        model.AddAbsEquality(slack, diff)
        weight = int(spec.get("confidence", 0.0) * 500 + 400)
        penalties.append(weight * slack)

    for eq in equations:
        # pre-process: move RHS divisions to LHS
        new_lhs, new_rhs = move_rhs_divisions_to_lhs(eq["lhs"], eq["rhs"])
        processed_eq = {**eq, "lhs": new_lhs, "rhs": new_rhs}
        add_soft_constraint(model, cp_vars, processed_eq, penalties)

    model.Minimize(cp_model.LinearExpr.Sum(penalties))

    cp_solver = cp_model.CpSolver()
    cp_solver.parameters.num_search_workers = num_workers
    cp_solver.parameters.max_time_in_seconds = timeout_seconds

    status = cp_solver.Solve(model)
    status_name = _STATUS_NAMES.get(status, "UNKNOWN")
    logger.info("OR-Tools solver status: %s", status_name)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_name, "corrected": {}}

    corrected: dict = {}
    for k, spec in var_specs.items():
        if spec.get("fixed", False):
            continue
        raw = cp_solver.Value(cp_vars[k])
        mult_factor = spec.get("mult_factor", 1)
        corrected[k] = raw / mult_factor

    return {"status": status_name, "corrected": corrected}
