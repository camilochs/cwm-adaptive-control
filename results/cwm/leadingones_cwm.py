class SynthesizedCWM:
    def predict_next_state(self, state: LOState, action: LOAction) -> LOState:
        """Predict the next state given current state and action."""
        import math
        
        i = state.fitness
        k = action.k
        n = state.n
        
        # If already at optimum, no improvement possible
        if i >= n:
            return LOState(
                fitness=i,
                n=n,
                step=state.step + 1,
                budget=state.budget,
                normalized_fitness=float(i) / n
            )
        
        # Probability of improvement: bit i must be flipped AND none of first i bits flipped
        # P(improve) = (k/n) * ((n-k)/(n-1))^i for i > 0
        # Special case for i=0: P(improve) = k/n
        if i == 0:
            p_improve = min(1.0, k / n) if n > 0 else 0.0
        else:
            if k >= n or n <= 1:
                p_improve = 0.0
            else:
                p_improve = (k / n) * pow((n - k) / (n - 1), i)
                p_improve = min(1.0, max(0.0, p_improve))
        
        # Expected fitness (continuous)
        expected_fitness = i + p_improve
        
        # Integer fitness (floor of expected)
        next_fitness = int(expected_fitness)
        
        return LOState(
            fitness=next_fitness,
            n=n,
            step=state.step + 1,
            budget=state.budget,
            normalized_fitness=expected_fitness / n
        )
    
    def get_legal_actions(self, state: LOState) -> list:
        """Return legal actions based on current state."""
        n = state.n
        i = state.fitness
        
        actions = []
        
        # Always include k=1 as it's safe and common
        actions.append(LOAction(k=1, name="flip_1"))
        
        # Include theoretically optimal k = floor(n / (i + 1))
        if i < n:
            optimal_k = max(1, n // (i + 1))
            if optimal_k != 1 and optimal_k <= n:
                actions.append(LOAction(k=optimal_k, name=f"flip_{optimal_k}"))
        
        # Include some other reasonable values based on patterns from data
        candidates = []
        
        # Small k values (2-5) are often good
        for k in [2, 3, 4, 5]:
            if k <= n and k not in [1, optimal_k if i < n else -1]:
                candidates.append(k)
        
        # Medium k values around n/4 to n/2
        quarter_n = max(1, n // 4)
        half_n = max(1, n // 2)
        
        for k in [quarter_n, half_n]:
            if k <= n and k not in [1, optimal_k if i < n else -1] and k not in candidates:
                candidates.append(k)
        
        # Add some exploration: larger k values
        if n >= 10:
            large_k = min(n, max(10, 2 * n // 3))
            if large_k not in [1, optimal_k if i < n else -1] and large_k not in candidates:
                candidates.append(large_k)
        
        # Sort candidates and take best ones
        candidates.sort()
        for k in candidates[:5]:  # Limit to avoid too many actions
            actions.append(LOAction(k=k, name=f"flip_{k}"))
        
        return actions
    
    def evaluate_state(self, state: LOState) -> float:
        """Evaluate state quality using normalized_fitness for fine-grained comparison."""
        # Base score from normalized fitness progress
        progress_score = state.normalized_fitness
        
        # Bonus for being close to optimum
        if state.fitness == state.n:
            return 1000.0  # Maximum reward for reaching optimum
        
        # Penalty for using too many steps
        step_penalty = 0.0
        if state.budget > 0:
            step_ratio = state.step / state.budget
            step_penalty = step_ratio * 0.1  # Small penalty for step usage
        
        # Bonus for high fitness values (getting close to optimum)
        fitness_bonus = (state.fitness / state.n) * 0.1
        
        return progress_score + fitness_bonus - step_penalty
    
    def is_terminal(self, state: LOState) -> bool:
        """Check if state is terminal."""
        # Terminal if optimum reached
        if state.fitness >= state.n:
            return True
        
        # Terminal if budget exhausted
        if state.budget > 0 and state.step >= state.budget:
            return True
        
        return False