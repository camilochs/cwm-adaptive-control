"""CWM synthesis pipeline using LLMs."""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import time

from .validator import CWMValidator, ValidationResult, extract_code_from_response
from .templates.synthesis_prompt import (
    SYSTEM_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
    REFINEMENT_PROMPT_TEMPLATE,
    create_synthesis_prompt,
)


@dataclass
class SynthesisConfig:
    """Configuration for CWM synthesis."""
    # LLM settings
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Synthesis settings
    max_iterations: int = 10
    max_refinement_attempts: int = 3
    trajectory_samples: int = 15

    # Validation
    run_tests: bool = True
    timeout: float = 5.0


@dataclass
class SynthesisResult:
    """Result of CWM synthesis."""
    success: bool
    code: Optional[str]
    validation: Optional[ValidationResult]
    iterations: int
    total_time: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class LLMClient:
    """Unified interface for LLM APIs."""

    def __init__(self, provider: str = "anthropic", model: str = None):
        self.provider = provider
        self.model = model
        self._client = None

    def _init_client(self):
        """Initialize the appropriate client."""
        if self._client is not None:
            return

        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
                self.model = self.model or "claude-sonnet-4-20250514"
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI()
                self.model = self.model or "gpt-4-turbo"
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.provider == "ollama":
            try:
                import ollama
                self._client = ollama
                self.model = self.model or "codellama"
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from the LLM."""
        self._init_client()

        if self.provider == "anthropic":
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = self._client.messages.create(**kwargs)
            return response.content[0].text

        elif self.provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "ollama":
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                system=system or "",
            )
            return response["response"]


class CWMSynthesizer:
    """Synthesizes Code World Models from trajectory data."""

    def __init__(
        self,
        config: SynthesisConfig = None,
        llm_client: LLMClient = None,
    ):
        self.config = config or SynthesisConfig()
        self.llm_client = llm_client
        self.validator = CWMValidator(timeout=self.config.timeout)

    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self.llm_client is not None:
            return self.llm_client

        # Determine provider from model name
        model = self.config.model
        if "claude" in model.lower():
            provider = "anthropic"
        elif "gpt" in model.lower():
            provider = "openai"
        else:
            provider = "ollama"

        return LLMClient(provider=provider, model=model)

    def synthesize(
        self,
        trajectories: List[Any],  # List[Trajectory] - avoid circular import
        callback: Optional[Callable[[int, str, ValidationResult], None]] = None,
    ) -> SynthesisResult:
        """Synthesize a CWM from trajectory data.

        Args:
            trajectories: List of optimization trajectories
            callback: Optional callback(iteration, code, validation) for progress

        Returns:
            SynthesisResult with synthesized code or error
        """
        start_time = time.time()
        history = []

        # Create synthesis prompt
        prompt = create_synthesis_prompt(
            trajectories,
            max_samples=self.config.trajectory_samples,
        )

        client = self._get_llm_client()
        best_code = None
        best_validation = None

        for iteration in range(self.config.max_iterations):
            try:
                # Generate code
                response = client.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                # Extract code
                code = extract_code_from_response(response)

                # Validate
                validation = self.validator.validate(code)

                history.append({
                    "iteration": iteration,
                    "code_length": len(code),
                    "valid": validation.valid,
                    "tests_passed": validation.tests_passed,
                })

                if callback:
                    callback(iteration, code, validation)

                if validation.valid:
                    return SynthesisResult(
                        success=True,
                        code=code,
                        validation=validation,
                        iterations=iteration + 1,
                        total_time=time.time() - start_time,
                        history=history,
                    )

                # Track best attempt
                if best_validation is None or sum(validation.tests_passed.values()) > sum(best_validation.tests_passed.values()):
                    best_code = code
                    best_validation = validation

                # Refine with error feedback
                prompt = self._create_refinement_prompt(
                    code, validation, trajectories
                )

            except Exception as e:
                history.append({
                    "iteration": iteration,
                    "error": str(e),
                })

        # Return best attempt even if not fully valid
        return SynthesisResult(
            success=False,
            code=best_code,
            validation=best_validation,
            iterations=self.config.max_iterations,
            total_time=time.time() - start_time,
            history=history,
            error="Max iterations reached without valid CWM",
        )

    def _create_refinement_prompt(
        self,
        code: str,
        validation: ValidationResult,
        trajectories: List[Any],
    ) -> str:
        """Create refinement prompt from validation errors."""
        # Gather statistics from trajectories
        all_transitions = []
        for traj in trajectories:
            all_transitions.extend(traj.transitions)

        fitnesses = [t[0]["best_fitness"] for t in all_transitions]
        improvements = [
            t[0]["best_fitness"] - t[2]["best_fitness"]
            for t in all_transitions
        ]

        # Count action occurrences
        action_counts = {}
        for _, action, _ in all_transitions:
            if action:
                name = action["name"]
                action_counts[name] = action_counts.get(name, 0) + 1

        common_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
        common_actions_str = ", ".join(f"{name}({count})" for name, count in common_actions)

        # Format test failures
        failures = []
        for test, passed in validation.tests_passed.items():
            if not passed:
                failures.append(f"- {test}: FAILED")
        for error in validation.errors:
            failures.append(f"- Error: {error}")

        return REFINEMENT_PROMPT_TEMPLATE.format(
            current_code=code,
            test_failures="\n".join(failures),
            min_fitness=min(fitnesses),
            max_fitness=max(fitnesses),
            avg_improvement=sum(improvements) / len(improvements) if improvements else 0,
            common_actions=common_actions_str,
        )

    def synthesize_from_file(
        self,
        trajectory_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> SynthesisResult:
        """Synthesize CWM from trajectory file."""
        # Import here to avoid circular import
        from ..trajectory.storage import TrajectoryStorage

        trajectories = TrajectoryStorage.load_json(trajectory_path)
        result = self.synthesize(trajectories)

        if result.success and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result.code)

        return result

    @staticmethod
    def load_cwm(code_or_path: Union[str, Path]) -> Any:
        """Load a synthesized CWM from code or file path."""
        if Path(code_or_path).exists():
            with open(code_or_path) as f:
                code = f.read()
        else:
            code = code_or_path

        # Create namespace with required classes
        namespace = {}
        setup_code = """
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

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
        exec(setup_code, namespace)
        exec(code, namespace)

        return namespace["SynthesizedCWM"]()
