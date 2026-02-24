"""Validator for synthesized CWM code."""

import ast
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from io import StringIO
import math


@dataclass
class ValidationResult:
    """Result of CWM validation."""
    valid: bool
    syntax_ok: bool
    imports_ok: bool
    class_found: bool
    methods_ok: Dict[str, bool]
    tests_passed: Dict[str, bool]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_tests_passed(self) -> bool:
        return all(self.tests_passed.values()) if self.tests_passed else False

    def summary(self) -> str:
        """Get a summary of validation results."""
        lines = [
            f"Valid: {self.valid}",
            f"Syntax: {'OK' if self.syntax_ok else 'FAILED'}",
            f"Imports: {'OK' if self.imports_ok else 'FAILED'}",
            f"Class found: {'OK' if self.class_found else 'FAILED'}",
        ]
        if self.methods_ok:
            lines.append("Methods:")
            for method, ok in self.methods_ok.items():
                lines.append(f"  {method}: {'OK' if ok else 'MISSING'}")
        if self.tests_passed:
            lines.append("Tests:")
            for test, passed in self.tests_passed.items():
                lines.append(f"  {test}: {'PASSED' if passed else 'FAILED'}")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        return "\n".join(lines)


class CWMValidator:
    """Validates synthesized CWM code."""

    REQUIRED_METHODS = [
        "predict_next_state",
        "get_legal_actions",
        "is_terminal",
        "evaluate_state",
    ]

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def validate_structure(self, code: str) -> Tuple[bool, Dict[str, bool], Optional[str]]:
        """Check if code has required class and methods."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, {}, str(e)

        # Find SynthesizedCWM class
        cwm_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SynthesizedCWM":
                cwm_class = node
                break

        if cwm_class is None:
            return False, {}, "Class 'SynthesizedCWM' not found"

        # Check methods
        methods_found = {m: False for m in self.REQUIRED_METHODS}
        for node in ast.walk(cwm_class):
            if isinstance(node, ast.FunctionDef):
                if node.name in methods_found:
                    methods_found[node.name] = True

        all_methods = all(methods_found.values())
        return all_methods, methods_found, None if all_methods else "Missing required methods"

    def validate_execution(self, code: str) -> Tuple[bool, Dict[str, bool], List[str]]:
        """Execute code and run basic tests."""
        errors = []
        tests_passed = {
            "import": False,
            "instantiate": False,
            "predict_next_state": False,
            "get_legal_actions": False,
            "is_terminal": False,
            "evaluate_state": False,
            "numerical_stability": False,
        }

        # Create execution namespace with required imports
        namespace = {
            "math": math,
            "__builtins__": __builtins__,
        }

        # Add dataclass definitions
        setup_code = """
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class CWMState:
    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    improvement_rate: float
    stagnation: int
    fitness_std: float = 0.0
    population_size: int = 50
    dimension: int = 10

@dataclass
class CWMAction:
    action_type: str
    name: str
    params: Dict[str, Any]
"""

        try:
            exec(setup_code, namespace)
            tests_passed["import"] = True
        except Exception as e:
            errors.append(f"Setup failed: {e}")
            return False, tests_passed, errors

        # Execute the CWM code
        try:
            exec(code, namespace)
        except Exception as e:
            errors.append(f"Code execution failed: {e}")
            return False, tests_passed, errors

        # Instantiate
        try:
            cwm = namespace["SynthesizedCWM"]()
            tests_passed["instantiate"] = True
        except Exception as e:
            errors.append(f"Instantiation failed: {e}")
            return False, tests_passed, errors

        # Create test state and action
        CWMState = namespace["CWMState"]
        CWMAction = namespace["CWMAction"]

        test_state = CWMState(
            generation=10,
            best_fitness=1.0,
            mean_fitness=5.0,
            diversity=0.5,
            improvement_rate=0.1,
            stagnation=2,
            fitness_std=1.0,
            population_size=50,
            dimension=10,
        )

        test_action = CWMAction(
            action_type="mutation",
            name="gaussian_low",
            params={"rate": 0.1, "sigma": 0.1},
        )

        # Test predict_next_state
        try:
            next_state = cwm.predict_next_state(test_state, test_action)
            if isinstance(next_state, CWMState):
                if next_state.generation == test_state.generation + 1:
                    if next_state.best_fitness >= 0:
                        tests_passed["predict_next_state"] = True
                    else:
                        errors.append("predict_next_state: negative fitness")
                else:
                    errors.append("predict_next_state: generation not incremented")
            else:
                errors.append("predict_next_state: wrong return type")
        except Exception as e:
            errors.append(f"predict_next_state failed: {e}")

        # Test get_legal_actions
        try:
            actions = cwm.get_legal_actions(test_state)
            if isinstance(actions, list) and len(actions) > 0:
                if all(isinstance(a, CWMAction) for a in actions):
                    tests_passed["get_legal_actions"] = True
                else:
                    errors.append("get_legal_actions: wrong element types")
            else:
                errors.append("get_legal_actions: empty or wrong type")
        except Exception as e:
            errors.append(f"get_legal_actions failed: {e}")

        # Test is_terminal
        try:
            result = cwm.is_terminal(test_state)
            if isinstance(result, bool):
                tests_passed["is_terminal"] = True
            else:
                errors.append("is_terminal: wrong return type")
        except Exception as e:
            errors.append(f"is_terminal failed: {e}")

        # Test evaluate_state
        try:
            score = cwm.evaluate_state(test_state)
            if isinstance(score, (int, float)):
                if not math.isnan(score) and not math.isinf(score):
                    tests_passed["evaluate_state"] = True
                else:
                    errors.append("evaluate_state: NaN or Inf score")
            else:
                errors.append("evaluate_state: wrong return type")
        except Exception as e:
            errors.append(f"evaluate_state failed: {e}")

        # Test numerical stability with edge cases
        edge_state = CWMState(
            generation=1,
            best_fitness=0.0,
            mean_fitness=0.0,
            diversity=0.0,
            improvement_rate=0.0,
            stagnation=100,
            fitness_std=0.0,
            population_size=50,
            dimension=10,
        )

        try:
            next_state = cwm.predict_next_state(edge_state, test_action)
            score = cwm.evaluate_state(edge_state)
            terminal = cwm.is_terminal(edge_state)

            if not math.isnan(score) and not math.isinf(score):
                tests_passed["numerical_stability"] = True
            else:
                errors.append("numerical_stability: NaN/Inf with edge case")
        except Exception as e:
            errors.append(f"numerical_stability failed: {e}")

        all_passed = all(tests_passed.values())
        return all_passed, tests_passed, errors

    def validate(self, code: str) -> ValidationResult:
        """Full validation of CWM code."""
        result = ValidationResult(
            valid=False,
            syntax_ok=False,
            imports_ok=False,
            class_found=False,
            methods_ok={},
            tests_passed={},
        )

        # Check syntax
        syntax_ok, syntax_error = self.validate_syntax(code)
        result.syntax_ok = syntax_ok
        if not syntax_ok:
            result.errors.append(syntax_error)
            return result

        # Check structure
        structure_ok, methods_ok, structure_error = self.validate_structure(code)
        result.class_found = "SynthesizedCWM" in code
        result.methods_ok = methods_ok
        if not structure_ok:
            result.errors.append(structure_error)
            return result

        # Run execution tests
        exec_ok, tests_passed, exec_errors = self.validate_execution(code)
        result.imports_ok = tests_passed.get("import", False)
        result.tests_passed = tests_passed
        result.errors.extend(exec_errors)

        result.valid = exec_ok
        return result


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
    # Look for code blocks
    if "```python" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Return full response if no code blocks
    return response.strip()
