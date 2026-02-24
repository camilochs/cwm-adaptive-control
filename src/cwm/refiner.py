"""GIF-MCTS style refinement for CWM synthesis."""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

from .validator import CWMValidator, ValidationResult, extract_code_from_response
from .synthesizer import LLMClient, SynthesisConfig
from .templates.synthesis_prompt import SYSTEM_PROMPT, REFINEMENT_PROMPT_TEMPLATE


@dataclass
class RefinementNode:
    """Node in the refinement tree."""
    code: str
    validation: ValidationResult
    parent: Optional["RefinementNode"] = None
    children: List["RefinementNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    refinement_prompt: str = ""

    @property
    def score(self) -> float:
        """Compute score based on validation results."""
        if self.validation is None:
            return 0.0

        score = 0.0
        # Points for each passed test
        if self.validation.tests_passed:
            score += sum(self.validation.tests_passed.values()) * 10

        # Bonus for valid syntax and structure
        if self.validation.syntax_ok:
            score += 5
        if self.validation.class_found:
            score += 5
        if self.validation.methods_ok:
            score += sum(self.validation.methods_ok.values()) * 3

        # Penalty for errors
        score -= len(self.validation.errors) * 2

        return max(0, score)

    def ucb1(self, exploration: float = 1.414) -> float:
        """Upper Confidence Bound for tree search."""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term


@dataclass
class RefinementResult:
    """Result of refinement process."""
    success: bool
    best_code: Optional[str]
    best_validation: Optional[ValidationResult]
    iterations: int
    nodes_explored: int
    tree_depth: int


class CWMRefiner:
    """Tree-search based refinement for CWM code (GIF-MCTS style)."""

    def __init__(
        self,
        config: SynthesisConfig = None,
        llm_client: LLMClient = None,
        validator: CWMValidator = None,
    ):
        self.config = config or SynthesisConfig()
        self.llm_client = llm_client
        self.validator = validator or CWMValidator()

    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self.llm_client is not None:
            return self.llm_client

        model = self.config.model
        if "claude" in model.lower():
            provider = "anthropic"
        elif "gpt" in model.lower():
            provider = "openai"
        else:
            provider = "ollama"

        return LLMClient(provider=provider, model=model)

    def _select_node(self, root: RefinementNode) -> RefinementNode:
        """Select best node using UCB1."""
        node = root
        while node.children:
            # Select child with highest UCB1
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def _expand_node(
        self,
        node: RefinementNode,
        client: LLMClient,
        trajectory_stats: Dict[str, Any],
    ) -> Optional[RefinementNode]:
        """Expand a node by generating a refined version."""
        # Create refinement prompt
        prompt = self._create_refinement_prompt(node, trajectory_stats)

        try:
            response = client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=self.config.temperature + 0.1 * len(node.children),  # Increase diversity
                max_tokens=self.config.max_tokens,
            )

            code = extract_code_from_response(response)
            validation = self.validator.validate(code)

            child = RefinementNode(
                code=code,
                validation=validation,
                parent=node,
                refinement_prompt=prompt,
            )
            node.children.append(child)
            return child

        except Exception as e:
            return None

    def _backpropagate(self, node: RefinementNode, score: float):
        """Backpropagate score up the tree."""
        while node is not None:
            node.visits += 1
            node.value += score
            node = node.parent

    def _create_refinement_prompt(
        self,
        node: RefinementNode,
        stats: Dict[str, Any],
    ) -> str:
        """Create a refinement prompt for a node."""
        failures = []
        if node.validation:
            for test, passed in node.validation.tests_passed.items():
                if not passed:
                    failures.append(f"- {test}: FAILED")
            for error in node.validation.errors:
                failures.append(f"- Error: {error}")

        return REFINEMENT_PROMPT_TEMPLATE.format(
            current_code=node.code,
            test_failures="\n".join(failures) if failures else "None",
            min_fitness=stats.get("min_fitness", 0),
            max_fitness=stats.get("max_fitness", 100),
            avg_improvement=stats.get("avg_improvement", 0),
            common_actions=stats.get("common_actions", "N/A"),
        )

    def _compute_tree_depth(self, root: RefinementNode) -> int:
        """Compute maximum depth of tree."""
        if not root.children:
            return 0
        return 1 + max(self._compute_tree_depth(c) for c in root.children)

    def _count_nodes(self, root: RefinementNode) -> int:
        """Count total nodes in tree."""
        return 1 + sum(self._count_nodes(c) for c in root.children)

    def refine(
        self,
        initial_code: str,
        trajectory_stats: Dict[str, Any],
        max_iterations: int = None,
        callback: Optional[Callable[[int, RefinementNode], None]] = None,
    ) -> RefinementResult:
        """Refine CWM code using tree search.

        Args:
            initial_code: Starting code to refine
            trajectory_stats: Statistics from trajectory data
            max_iterations: Maximum refinement iterations
            callback: Optional callback(iteration, best_node)

        Returns:
            RefinementResult with best code found
        """
        max_iterations = max_iterations or self.config.max_refinement_attempts * 3

        # Initialize root
        initial_validation = self.validator.validate(initial_code)
        root = RefinementNode(
            code=initial_code,
            validation=initial_validation,
        )

        if initial_validation.valid:
            return RefinementResult(
                success=True,
                best_code=initial_code,
                best_validation=initial_validation,
                iterations=0,
                nodes_explored=1,
                tree_depth=0,
            )

        client = self._get_llm_client()
        best_node = root

        for iteration in range(max_iterations):
            # Selection
            node = self._select_node(root)

            # Expansion
            child = self._expand_node(node, client, trajectory_stats)

            if child is not None:
                # Backpropagation
                score = child.score
                self._backpropagate(child, score)

                # Track best
                if child.score > best_node.score:
                    best_node = child

                if callback:
                    callback(iteration, best_node)

                # Early termination if valid
                if child.validation and child.validation.valid:
                    return RefinementResult(
                        success=True,
                        best_code=child.code,
                        best_validation=child.validation,
                        iterations=iteration + 1,
                        nodes_explored=self._count_nodes(root),
                        tree_depth=self._compute_tree_depth(root),
                    )

        return RefinementResult(
            success=best_node.validation.valid if best_node.validation else False,
            best_code=best_node.code,
            best_validation=best_node.validation,
            iterations=max_iterations,
            nodes_explored=self._count_nodes(root),
            tree_depth=self._compute_tree_depth(root),
        )

    def refine_with_tests(
        self,
        initial_code: str,
        test_cases: List[Tuple[Dict, Dict, Dict]],  # (state, action, expected_next_state)
        max_iterations: int = None,
    ) -> RefinementResult:
        """Refine CWM to match specific test cases."""
        # This would extend validation to include custom test cases
        # For now, delegate to basic refine
        stats = {
            "min_fitness": 0,
            "max_fitness": 100,
            "avg_improvement": 0.01,
            "common_actions": "mutation, crossover",
        }
        return self.refine(initial_code, stats, max_iterations)
