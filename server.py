#!/usr/bin/env python3
"""OR-Tools MCP server — constraint-based value correction via CP-SAT."""

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from solver import get_variables, validate_equation, solver as run_solver

logging.basicConfig(level=logging.INFO)

app = Server("ortools-solver")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="solve",
            description=(
                "Run OR-Tools CP-SAT to find corrected values for variables that best satisfy "
                "a set of equations. Fixed variables are pinned to their observed value; "
                "free variables are adjusted to minimise weighted equation violations. "
                "Equations are pre-processed to move RHS divisions to the LHS."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "object",
                        "description": (
                            "Map of variable name → spec. "
                            "Spec fields: min (int, default 0), max (int, default 2e9), "
                            "obs (observed value, required for penalty), "
                            "confidence (0–1, default 0.0 — obs trust level), "
                            "mult_factor (int, default 1 — scale float→int before solving, divide back after), "
                            "fixed (bool, default false — pin variable to obs)."
                        ),
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                                "obs": {"type": "number"},
                                "confidence": {"type": "number"},
                                "mult_factor": {"type": "integer"},
                                "fixed": {"type": "boolean"},
                            },
                        },
                    },
                    "equations": {
                        "type": "array",
                        "description": "List of equations the solver should satisfy.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lhs": {"type": "string", "description": "Left-hand side expression, e.g. 'gross_amount'"},
                                "rhs": {"type": "string", "description": "Right-hand side expression, e.g. 'quantity * rate_per_pc'"},
                                "relation": {"type": "string", "enum": ["==", "<=", ">="], "description": "Relation between lhs and rhs"},
                                "weight": {"type": "integer", "description": "Penalty weight for violations (default 1000)"},
                                "tolerance": {"type": "integer", "description": "Allowed slack before penalty kicks in (default 0, in scaled integer units)"},
                            },
                            "required": ["lhs", "rhs", "relation"],
                        },
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Solver time limit in seconds (default 5).",
                    },
                    "num_workers": {
                        "type": "integer",
                        "description": "Number of parallel search workers (default 8).",
                    },
                },
                "required": ["variables", "equations"],
            },
        ),
        Tool(
            name="validate_equations",
            description=(
                "Check equation strings for unsupported constructs (unary negation, "
                "non-constant exponents, syntax errors) before passing them to solve. "
                "Returns a list of errors per equation — empty list means the equation is valid."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "equations": {
                        "type": "array",
                        "description": "Equations to validate.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lhs": {"type": "string"},
                                "rhs": {"type": "string"},
                            },
                            "required": ["lhs", "rhs"],
                        },
                    },
                },
                "required": ["equations"],
            },
        ),
        Tool(
            name="evaluate_equations",
            description=(
                "Evaluate equations against a set of known values and return pass/fail "
                "for each equation. No OR-Tools involved — useful to check whether existing "
                "values already satisfy the system before deciding to run solve."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "equations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lhs": {"type": "string"},
                                "rhs": {"type": "string"},
                                "relation": {"type": "string", "enum": ["==", "<=", ">="]},
                                "tolerance": {
                                    "type": "number",
                                    "description": "Allowed absolute error before marking as failed (default 0).",
                                },
                            },
                            "required": ["lhs", "rhs", "relation"],
                        },
                    },
                    "values": {
                        "type": "object",
                        "description": "Map of variable name → numeric value.",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": ["equations", "values"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "solve":
        variables = arguments["variables"]
        equations = arguments["equations"]

        # apply defaults
        for eq in equations:
            eq.setdefault("weight", 1000)
            eq.setdefault("tolerance", 0)
        for spec in variables.values():
            spec.setdefault("mult_factor", 1)
            spec.setdefault("fixed", False)
            spec.setdefault("confidence", 0.0)
            mf = spec["mult_factor"]
            # user provides real-world values; scale to integer domain
            if "obs" in spec:
                spec["obs"] = int(spec["obs"] * mf)
            spec["min"] = int(spec.get("min", 0) * mf)
            spec["max"] = int(spec.get("max", 2_000_000_000 // mf) * mf)

        result = run_solver(
            var_specs=variables,
            equations=equations,
            timeout_seconds=arguments.get("timeout_seconds", 5),
            num_workers=arguments.get("num_workers", 8),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "validate_equations":
        equations = arguments["equations"]
        results = []
        for eq in equations:
            errors = validate_equation(eq["lhs"]) + validate_equation(eq["rhs"])
            results.append({"lhs": eq["lhs"], "rhs": eq["rhs"], "errors": errors})
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "evaluate_equations":
        equations = arguments["equations"]
        values = arguments["values"]
        results = []

        for eq in equations:
            lhs_expr = eq["lhs"]
            rhs_expr = eq["rhs"]
            relation = eq["relation"]
            tol = eq.get("tolerance", 0)

            lhs_vars = set(get_variables(lhs_expr))
            rhs_vars = set(get_variables(rhs_expr))
            all_vars = lhs_vars | rhs_vars

            missing = [v for v in all_vars if v not in values]
            if missing:
                results.append({
                    "lhs": lhs_expr,
                    "rhs": rhs_expr,
                    "status": "missing_values",
                    "missing": missing,
                })
                continue

            try:
                safe_ns = {k: values[k] for k in all_vars}
                lhs_val = eval(compile(lhs_expr, "<lhs>", "eval"), {"__builtins__": {}}, safe_ns)  # noqa: S307
                rhs_val = eval(compile(rhs_expr, "<rhs>", "eval"), {"__builtins__": {}}, safe_ns)  # noqa: S307

                if relation == "==":
                    passed = abs(lhs_val - rhs_val) <= tol
                elif relation == "<=":
                    passed = lhs_val <= rhs_val + tol
                elif relation == ">=":
                    passed = lhs_val >= rhs_val - tol
                else:
                    passed = False

                results.append({
                    "lhs": lhs_expr,
                    "rhs": rhs_expr,
                    "status": "passed" if passed else "failed",
                    "actual": lhs_val,
                    "computed": rhs_val,
                    "error": abs(lhs_val - rhs_val),
                })
            except Exception as e:
                results.append({
                    "lhs": lhs_expr,
                    "rhs": rhs_expr,
                    "status": "error",
                    "message": str(e),
                })

        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
