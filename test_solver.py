import pytest
from solver import (
    get_variables,
    validate_equation,
    move_rhs_divisions_to_lhs,
    solver,
)


# ---------------------------------------------------------------------------
# get_variables
# ---------------------------------------------------------------------------

class TestGetVariables:
    def test_simple(self):
        assert get_variables("a + b") == ["a", "b"]

    def test_constants_excluded(self):
        assert get_variables("a * 2 + b") == ["a", "b"]

    def test_single(self):
        assert get_variables("x") == ["x"]

    def test_no_variables(self):
        assert get_variables("1 + 2") == []

    def test_duplicates_deduplicated(self):
        assert get_variables("a * a + b") == ["a", "b"]

    def test_nested(self):
        assert get_variables("(a + b) * c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# validate_equation
# ---------------------------------------------------------------------------

class TestValidateEquation:
    def test_valid_arithmetic(self):
        assert validate_equation("a + b * c") == []

    def test_valid_division(self):
        assert validate_equation("a / b") == []

    def test_valid_modulo(self):
        assert validate_equation("a % b") == []

    def test_valid_constant_pow(self):
        assert validate_equation("a ** 2") == []

    def test_valid_constant_pow_1(self):
        assert validate_equation("a ** 1") == []

    def test_unary_negation(self):
        errors = validate_equation("-a")
        assert len(errors) == 1
        assert "Unary" in errors[0]
        assert "USub" in errors[0]

    def test_unary_plus(self):
        errors = validate_equation("+a")
        assert len(errors) == 1
        assert "Unary" in errors[0]

    def test_non_constant_pow(self):
        errors = validate_equation("a ** b")
        assert len(errors) == 1
        assert "**" in errors[0]

    def test_float_pow_rejected(self):
        errors = validate_equation("a ** 2.5")
        assert len(errors) == 1

    def test_zero_pow_rejected(self):
        errors = validate_equation("a ** 0")
        assert len(errors) == 1

    def test_syntax_error(self):
        errors = validate_equation("a +")
        assert len(errors) == 1
        assert "Syntax" in errors[0]

    def test_multiple_errors(self):
        # both unary and bad pow in one expression
        errors = validate_equation("-a ** b")
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# move_rhs_divisions_to_lhs
# ---------------------------------------------------------------------------

class TestMoveRhsDivisionsToLhs:
    def test_no_division_unchanged(self):
        lhs, rhs = move_rhs_divisions_to_lhs("gross", "quantity * rate")
        assert lhs == "gross"
        assert rhs == "quantity * rate"

    def test_simple_division(self):
        # gross = total / tax_rate  →  gross * tax_rate = total
        lhs, rhs = move_rhs_divisions_to_lhs("gross", "total / tax_rate")
        assert "tax_rate" in lhs
        assert rhs == "total"

    def test_chained_division(self):
        # a = b / c / d  →  a * c * d = b
        lhs, rhs = move_rhs_divisions_to_lhs("a", "b / c / d")
        assert rhs == "b"
        assert "c" in lhs
        assert "d" in lhs

    def test_addition_not_affected(self):
        lhs, rhs = move_rhs_divisions_to_lhs("total", "a + b")
        assert lhs == "total"
        assert rhs == "a + b"


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

def _fixed(obs, mult_factor=1):
    return {"obs": int(obs * mult_factor), "mult_factor": mult_factor, "fixed": True,
            "min": 0, "max": 2_000_000_000}

def _free(obs, confidence=0.5, mult_factor=1, min_=0, max_=2_000_000_000):
    return {"obs": int(obs * mult_factor), "confidence": confidence,
            "mult_factor": mult_factor, "fixed": False,
            "min": int(min_ * mult_factor), "max": int(max_ * mult_factor)}


class TestSolver:
    def test_additive_correction(self):
        # total = a + b; a and b fixed, total wrong
        result = solver(
            var_specs={
                "total": _free(90, confidence=0.3),
                "a": _fixed(60),
                "b": _fixed(40),
            },
            equations=[{"lhs": "total", "rhs": "a + b", "relation": "==", "weight": 2000, "tolerance": 0}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["total"] == pytest.approx(100, abs=1)

    def test_multiplicative_correction(self):
        # gross = quantity * rate; gross is wrong
        result = solver(
            var_specs={
                "gross": _free(500, confidence=0.3, mult_factor=100, max_=10_000_000),
                "quantity": _fixed(10),
                "rate": _fixed(52, mult_factor=100),
            },
            equations=[{"lhs": "gross", "rhs": "quantity * rate", "relation": "==", "weight": 2000, "tolerance": 0}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["gross"] == pytest.approx(520.0, abs=0.01)

    def test_mult_factor_scaling(self):
        # result should be divided back by mult_factor
        result = solver(
            var_specs={
                "price": _free(9950, confidence=0.1, mult_factor=100, max_=10_000_000),
                "base": _fixed(100, mult_factor=100),
            },
            equations=[{"lhs": "price", "rhs": "base", "relation": "==", "weight": 2000, "tolerance": 0}],
        )
        # price should be corrected to 100.0 (real), stored as 10000 internally
        assert result["corrected"]["price"] == pytest.approx(100.0, abs=0.01)

    def test_fixed_variable_not_in_output(self):
        result = solver(
            var_specs={
                "a": _free(50, confidence=0.5),
                "b": _fixed(30),
            },
            equations=[{"lhs": "a", "rhs": "b", "relation": "==", "weight": 1000, "tolerance": 0}],
        )
        assert "b" not in result["corrected"]
        assert "a" in result["corrected"]

    def test_high_confidence_wins(self):
        # a (high confidence) + b (low confidence) == 100
        # obs: a=60, b=50 → sum=110 off by 10; cheaper to move b
        result = solver(
            var_specs={
                "a": _free(60, confidence=0.9, max_=200),
                "b": _free(50, confidence=0.1, max_=200),
            },
            equations=[{"lhs": "a", "rhs": "100 - b", "relation": "==", "weight": 1000, "tolerance": 0}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        # a should stay near 60; b should move toward 40
        assert result["corrected"]["a"] == pytest.approx(60, abs=2)
        assert result["corrected"]["b"] == pytest.approx(40, abs=2)

    def test_tolerance_absorbs_small_error(self):
        # a == b with tolerance 10; obs error = 4 → no correction needed
        result = solver(
            var_specs={
                "a": _free(100, confidence=1.0, max_=200),
                "b": _fixed(104),
            },
            equations=[{"lhs": "a", "rhs": "b", "relation": "==", "weight": 5000, "tolerance": 4}],
        )
        # Within tolerance, obs penalty dominates — a stays near 100
        assert result["corrected"]["a"] == pytest.approx(100, abs=5)

    def test_pow_equation(self):
        # x**2 == 9; x free, obs=4 (wrong)
        result = solver(
            var_specs={
                "x": _free(4, confidence=0.1, max_=100),
                "nine": _fixed(9),
            },
            equations=[{"lhs": "x ** 2", "rhs": "nine", "relation": "==", "weight": 5000, "tolerance": 0}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["x"] == pytest.approx(3, abs=1)

    def test_inequality_gte(self):
        # a >= 50; obs a=30 (violates)
        result = solver(
            var_specs={
                "a": _free(30, confidence=0.5, max_=200),
                "threshold": _fixed(50),
            },
            equations=[{"lhs": "a", "rhs": "threshold", "relation": ">=", "weight": 5000, "tolerance": 0}],
        )
        assert result["corrected"]["a"] >= 49  # should be at or above threshold

    def test_inequality_lte(self):
        # a <= 50; obs a=80 (violates)
        result = solver(
            var_specs={
                "a": _free(80, confidence=0.5, max_=200),
                "cap": _fixed(50),
            },
            equations=[{"lhs": "a", "rhs": "cap", "relation": "<=", "weight": 5000, "tolerance": 0}],
        )
        assert result["corrected"]["a"] <= 51

    def test_invalid_equation_returns_error(self):
        result = solver(
            var_specs={"a": _free(10)},
            equations=[{"lhs": "-a", "rhs": "10", "relation": "==", "weight": 1000, "tolerance": 0}],
        )
        assert result["status"] == "INVALID_EQUATION"
        assert "errors" in result

    def test_no_obs_no_penalty(self):
        # variable with no obs — solver still finds feasible solution via equation
        result = solver(
            var_specs={
                "a": {"min": 0, "max": 100, "fixed": False, "mult_factor": 1},
                "b": _fixed(40),
                "c": _fixed(60),
            },
            equations=[{"lhs": "a", "rhs": "b + c", "relation": "==", "weight": 1000, "tolerance": 0}],
        )
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["corrected"]["a"] == pytest.approx(100, abs=1)

    def test_multi_equation_system(self):
        # taxable + tax == total; tax == taxable * 0  (0% tax placeholder)
        # real scenario: total=110, taxable=100, tax=10 (10% of taxable)
        # obs: total=110 (wrong→120), taxable=100 (correct), tax=10 (correct)
        result = solver(
            var_specs={
                "total": _free(120, confidence=0.2, max_=10000),
                "taxable": _fixed(100),
                "tax": _fixed(10),
            },
            equations=[
                {"lhs": "total", "rhs": "taxable + tax", "relation": "==", "weight": 2000, "tolerance": 0},
            ],
        )
        assert result["corrected"]["total"] == pytest.approx(110, abs=1)
