"""
Tests for server.py call_tool logic.

mcp is mocked so these tests run without installing the mcp package.
The mock makes @app.call_tool() a no-op decorator, leaving the function
callable directly.
"""
import asyncio
import json
import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock mcp before importing server
# ---------------------------------------------------------------------------

class _TextContent:
    def __init__(self, type, text):  # noqa: A002
        self.type = type
        self.text = text


class _Server:
    def __init__(self, *args, **kwargs):
        pass

    def call_tool(self):
        return lambda fn: fn

    def list_tools(self):
        return lambda fn: fn

    def create_initialization_options(self):
        return {}


_mcp_types = MagicMock()
_mcp_types.TextContent = _TextContent
_mcp_types.Tool = MagicMock()

_mcp_server = MagicMock()
_mcp_server.Server = _Server

_mcp_module = MagicMock()
_mcp_module.server = _mcp_server
_mcp_module.types = _mcp_types

sys.modules.setdefault("mcp", _mcp_module)
sys.modules.setdefault("mcp.server", _mcp_server)
sys.modules.setdefault("mcp.server.stdio", MagicMock())
sys.modules.setdefault("mcp.types", _mcp_types)

import server  # noqa: E402  (must come after sys.modules patching)


def _run(coro):
    return asyncio.run(coro)


def _parse(text_contents):
    return json.loads(text_contents[0].text)


# ---------------------------------------------------------------------------
# evaluate_equations
# ---------------------------------------------------------------------------

class TestEvaluateEquations:
    def _call(self, equations, values):
        return _parse(_run(server.call_tool("evaluate_equations", {
            "equations": equations,
            "values": values,
        })))

    def test_pass_exact(self):
        result = self._call(
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "=="}],
            values={"total": 100, "a": 60, "b": 40},
        )
        assert result[0]["status"] == "passed"
        assert result[0]["error"] == 0.0

    def test_fail_wrong_values(self):
        result = self._call(
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "=="}],
            values={"total": 110, "a": 60, "b": 40},
        )
        assert result[0]["status"] == "failed"
        assert result[0]["error"] == 10.0

    def test_missing_values(self):
        result = self._call(
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "=="}],
            values={"total": 100, "a": 60},  # b missing
        )
        assert result[0]["status"] == "missing_values"
        assert "b" in result[0]["missing"]

    def test_tolerance_within_passes(self):
        result = self._call(
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "==", "tolerance": 5}],
            values={"total": 103, "a": 60, "b": 40},  # error = 3 ≤ tol 5
        )
        assert result[0]["status"] == "passed"

    def test_tolerance_exceeded_fails(self):
        result = self._call(
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "==", "tolerance": 2}],
            values={"total": 106, "a": 60, "b": 40},  # error = 6 > tol 2
        )
        assert result[0]["status"] == "failed"

    def test_inequality_lte_pass(self):
        result = self._call(
            equations=[{"lhs": "price", "rhs": "cap", "relation": "<="}],
            values={"price": 80, "cap": 100},
        )
        assert result[0]["status"] == "passed"

    def test_inequality_lte_fail(self):
        result = self._call(
            equations=[{"lhs": "price", "rhs": "cap", "relation": "<="}],
            values={"price": 120, "cap": 100},
        )
        assert result[0]["status"] == "failed"

    def test_inequality_gte_pass(self):
        result = self._call(
            equations=[{"lhs": "margin", "rhs": "floor", "relation": ">="}],
            values={"margin": 15, "floor": 10},
        )
        assert result[0]["status"] == "passed"

    def test_multiple_equations(self):
        result = self._call(
            equations=[
                {"lhs": "total", "rhs": "taxable + tax", "relation": "=="},
                {"lhs": "tax", "rhs": "wrong_var", "relation": "=="},
            ],
            values={"total": 110, "taxable": 100, "tax": 10},
        )
        assert result[0]["status"] == "passed"
        assert result[1]["status"] == "missing_values"


# ---------------------------------------------------------------------------
# validate_equations
# ---------------------------------------------------------------------------

class TestValidateEquations:
    def _call(self, equations):
        return _parse(_run(server.call_tool("validate_equations", {
            "equations": equations,
        })))

    def test_valid_equation_no_errors(self):
        result = self._call([{"lhs": "gross", "rhs": "qty * rate"}])
        assert result[0]["errors"] == []

    def test_unary_negation_error(self):
        result = self._call([{"lhs": "discount", "rhs": "-base * 0.1"}])
        assert len(result[0]["errors"]) > 0
        assert "Unary" in result[0]["errors"][0]

    def test_invalid_lhs(self):
        result = self._call([{"lhs": "-x", "rhs": "y"}])
        assert len(result[0]["errors"]) > 0

    def test_mixed_valid_and_invalid(self):
        result = self._call([
            {"lhs": "a", "rhs": "b + c"},
            {"lhs": "x", "rhs": "-y"},
        ])
        assert result[0]["errors"] == []
        assert len(result[1]["errors"]) > 0


# ---------------------------------------------------------------------------
# solve tool — scaling via server
# ---------------------------------------------------------------------------

class TestSolveScaling:
    def _call(self, variables, equations, **kwargs):
        return _parse(_run(server.call_tool("solve", {
            "variables": variables,
            "equations": equations,
            **kwargs,
        })))

    def test_mult_factor_scales_and_divides_back(self):
        # gross_amount real=500 (wrong), should be 520 = qty(10) * rate(52)
        result = self._call(
            variables={
                "gross_amount": {"obs": 500, "confidence": 0.2, "mult_factor": 100,
                                 "min": 0, "max": 10000, "fixed": False},
                "quantity":     {"obs": 10,  "mult_factor": 1,   "fixed": True},
                "rate":         {"obs": 52,  "mult_factor": 100, "fixed": True},
            },
            equations=[{"lhs": "gross_amount", "rhs": "quantity * rate",
                        "relation": "==", "weight": 2000}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["gross_amount"] == pytest.approx(520.0, abs=0.1)

    def test_default_mult_factor_is_1(self):
        # no mult_factor provided — should default to 1 (integer scaling = no-op)
        result = self._call(
            variables={
                "total": {"obs": 90, "confidence": 0.2, "min": 0, "max": 200, "fixed": False},
                "a":     {"obs": 60, "fixed": True},
                "b":     {"obs": 40, "fixed": True},
            },
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "==", "weight": 2000}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["total"] == pytest.approx(100, abs=1)

    def test_invalid_equation_short_circuits(self):
        result = self._call(
            variables={"x": {"obs": 5, "min": 0, "max": 100, "fixed": False}},
            equations=[{"lhs": "-x", "rhs": "10", "relation": "==", "weight": 1000}],
        )
        assert result["status"] == "INVALID_EQUATION"


import pytest  # noqa: E402 — kept at bottom so the mock setup runs first


# ---------------------------------------------------------------------------
# optimize tool
# ---------------------------------------------------------------------------

class TestOptimizeTool:
    def _call(self, variables, objective, hard_constraints, soft_constraints=None, **kwargs):
        payload = {
            "variables": variables,
            "objective": objective,
            "hard_constraints": hard_constraints,
        }
        if soft_constraints is not None:
            payload["soft_constraints"] = soft_constraints
        payload.update(kwargs)
        return _parse(_run(server.call_tool("optimize", payload)))

    def test_maximize_via_server(self):
        result = self._call(
            variables={"a": {"min": 0, "max": 10}, "b": {"min": 0, "max": 20}},
            objective={"expr": "a + b", "direction": "maximize"},
            hard_constraints=[],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["objective_value"] == pytest.approx(30, abs=1)

    def test_minimize_via_server(self):
        result = self._call(
            variables={"cost": {"min": 0, "max": 1000}},
            objective={"expr": "cost", "direction": "minimize"},
            hard_constraints=[{"lhs": "cost", "rhs": "50", "relation": ">="}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["objective_value"] == pytest.approx(50, abs=1)

    def test_mult_factor_scaling(self):
        # price real domain 0..50 with mult_factor=100; maximize price
        result = self._call(
            variables={"price": {"min": 0, "max": 50, "mult_factor": 100}},
            objective={"expr": "price", "direction": "maximize"},
            hard_constraints=[],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["values"]["price"] == pytest.approx(50.0, abs=0.01)
        assert result["objective_value"] == pytest.approx(50.0, abs=0.01)

    def test_invalid_objective_short_circuits(self):
        result = self._call(
            variables={"x": {"min": 0, "max": 100}},
            objective={"expr": "-x", "direction": "maximize"},
            hard_constraints=[],
        )
        assert result["status"] == "INVALID_EQUATION"

    def test_infeasible_returns_infeasible(self):
        result = self._call(
            variables={"a": {"min": 0, "max": 200}},
            objective={"expr": "a", "direction": "maximize"},
            hard_constraints=[
                {"lhs": "a", "rhs": "100", "relation": ">="},
                {"lhs": "a", "rhs": "50",  "relation": "<="},
            ],
        )
        assert result["status"] == "INFEASIBLE"
