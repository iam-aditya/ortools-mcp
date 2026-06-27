# ortools-mcp

An MCP server that exposes [Google OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model) as tools for constraint-based value correction. Given a set of equations and observed variable values, it finds the minimal corrections needed to make everything consistent.

**Use case**: you have values extracted from a document (OCR, form parsing, etc.) that should satisfy known equations (e.g. `total = quantity × rate + tax`), but one or more values were misread. The solver finds which values to adjust and by how much, weighted by how much you trust each observation.

---

## Tools

### `solve`

Runs CP-SAT. Returns corrected values for free (non-fixed) variables that best satisfy the given equations.

**Input**

```json
{
  "variables": {
    "gross_amount": {
      "obs": 500,
      "confidence": 0.3,
      "mult_factor": 100,
      "min": 0,
      "max": 100000,
      "fixed": false
    },
    "quantity": {
      "obs": 10,
      "mult_factor": 1,
      "fixed": true
    },
    "rate_per_pc": {
      "obs": 52,
      "mult_factor": 100,
      "fixed": true
    }
  },
  "equations": [
    {
      "lhs": "gross_amount",
      "rhs": "quantity * rate_per_pc",
      "relation": "==",
      "weight": 1000,
      "tolerance": 0
    }
  ],
  "timeout_seconds": 5,
  "num_workers": 8
}
```

**Output**

```json
{
  "status": "OPTIMAL",
  "corrected": {
    "gross_amount": 520.0
  }
}
```

Only free (non-fixed) variables appear in `corrected`. Status values: `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNKNOWN`, `INVALID_EQUATION`.

**Variable spec fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `obs` | number | — | Observed (extracted) value in real-world units |
| `confidence` | 0–1 | `0.0` | How much to trust `obs`. Higher → stronger pull toward the observed value |
| `mult_factor` | int | `1` | Scales float values to integers for the solver (`obs × mult_factor`), result is divided back on return |
| `min` / `max` | number | `0` / `2e9` | Domain bounds in real-world units |
| `fixed` | bool | `false` | Pin this variable to `obs` — it won't appear in `corrected` |

**Equation fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `lhs` | string | required | Left-hand side expression |
| `rhs` | string | required | Right-hand side expression |
| `relation` | `==` \| `<=` \| `>=` | required | Relation between lhs and rhs |
| `weight` | int | `1000` | Penalty per unit of violation — higher means the solver tries harder to satisfy this equation |
| `tolerance` | int | `0` | Allowed slack (in scaled integer units) before penalty kicks in |

**Solver params**

| Field | Default | Description |
|---|---|---|
| `timeout_seconds` | `5` | Wall-clock limit for the solver |
| `num_workers` | `8` | Parallel search workers |

---

### `validate_equations`

Pure AST check — no solver invoked. Call this before `solve` to catch unsupported constructs early.

**Input**

```json
{
  "equations": [
    { "lhs": "total", "rhs": "taxable + tax" },
    { "lhs": "discount", "rhs": "-base * 0.1" }
  ]
}
```

**Output**

```json
[
  { "lhs": "total", "rhs": "taxable + tax", "errors": [] },
  {
    "lhs": "discount",
    "rhs": "-base * 0.1",
    "errors": [
      "Unary operator 'USub' not supported in '-base * 0.1'. Rewrite e.g. '-x' as '(0 - x)'."
    ]
  }
]
```

Empty `errors` list means the equation is valid. Catches:
- Unary negation (`-x`) and unary plus (`+x`) — rewrite as `(0 - x)`
- Non-constant or zero/negative exponents in `**` — only `x ** 2`, `x ** 3`, etc. are supported
- Syntax errors

---

### `evaluate_equations`

Evaluates equations against known values and returns pass/fail per equation. No OR-Tools involved — useful to check whether values already satisfy the system before deciding to call `solve`.

**Input**

```json
{
  "equations": [
    { "lhs": "gross_amount", "rhs": "quantity * rate_per_pc", "relation": "==", "tolerance": 0.5 }
  ],
  "values": {
    "gross_amount": 520,
    "quantity": 10,
    "rate_per_pc": 52
  }
}
```

**Output**

```json
[
  {
    "lhs": "gross_amount",
    "rhs": "quantity * rate_per_pc",
    "status": "passed",
    "actual": 520,
    "computed": 520,
    "error": 0.0
  }
]
```

Status values: `passed`, `failed`, `missing_values` (variable not in `values`), `error` (evaluation exception).

---

## Supported expression syntax

Both `lhs` and `rhs` accept arithmetic expressions over variable names and numeric constants:

| Operator | Example | Notes |
|---|---|---|
| `+` | `a + b` | |
| `-` | `a - b` | Binary only — `-a` (unary) is not supported |
| `*` | `qty * rate` | |
| `/` | `total / qty` | RHS divisions are automatically moved to LHS to avoid integer-division loss |
| `%` | `a % b` | Modulo |
| `**` | `x ** 2` | Exponent must be a constant positive integer |
| Parentheses | `(a + b) * c` | |

---

## `mult_factor` and integer scaling

OR-Tools CP-SAT works only with integers. `mult_factor` scales real-world float values to integers before solving and divides back on return.

- **Amount fields** (prices, totals): `mult_factor: 100` — values like `52.30` become `5230` internally
- **Quantity fields**: `mult_factor: 1` — must stay at 1 for multiplicative equations

**Why quantities must use `mult_factor: 1`**: for `gross = qty × rate`, scaling all three by 100 gives `gross×100 = (qty×100) × (rate×100)`, which is off by 100×. With qty at ×1: `gross×100 = (qty×1) × (rate×100)` — correct.

`tolerance` is expressed in scaled integer units (after mult_factor is applied). For `mult_factor: 100`, a tolerance of `200` means ±2.00 in real units.

---

## Setup

```bash
pip install -r requirements.txt
```

**Add to Claude Code** (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "ortools-solver": {
      "command": "python3",
      "args": ["/path/to/ortools-mcp/server.py"]
    }
  }
}
```

---

## Running tests

```bash
pip install pytest
pytest test_solver.py test_server.py -v
```

`test_server.py` mocks the `mcp` package so it runs without needing `mcp` installed. 50 tests total covering all three tools, expression validation, scaling, confidence weighting, tolerance, and inequality constraints.
