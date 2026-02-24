class SynthesizedCWM:
    def predict_next_state(self, state: OMState, action: OMAction) -> OMState:
        i = state.fitness
        k = action.k
        n = state.n
        
        # Special case: if already at optimum, no change
        if i == n:
            return OMState(
                fitness=i,
                n=n,
                step=state.step + 1,
                budget=state.budget,
                normalized_fitness=1.0
            )
        
        # Special case: if k > n, treat as k = n
        k = min(k, n)
        
        # Expected value calculation using hypergeometric model
        # E[j] = k * i / n where j is number of 1-bits flipped
        # New fitness = i + k - 2j, so E[new_fitness] = i + k - 2*k*i/n = i + k*(1 - 2*i/n)
        expected_j = k * i / n
        expected_new_fitness = i + k - 2 * expected_j
        
        # For acceptance probability, we need P(j <= k/2)
        # This is complex to compute exactly, so we use approximation
        # When i < n/2, most flips are beneficial
        # When i > n/2, only small k values are likely to succeed
        
        if i < n // 2:
            # In the first phase, large k is generally good
            acceptance_prob = min(1.0, 0.8 + 0.2 * (n//2 - i) / (n//2))
        elif i > (n * 3) // 4:
            # Near optimum, only small k works well
            acceptance_prob = max(0.1, 1.0 - k / 10.0)
        else:
            # Transition region
            acceptance_prob = max(0.2, 0.6 - 0.4 * k / n)
        
        # If not accepted, fitness stays the same
        if acceptance_prob < 0.5:  # Simplified acceptance decision
            final_expected_fitness = i
        else:
            final_expected_fitness = max(i, expected_new_fitness)
        
        # Ensure we don't exceed maximum fitness
        final_expected_fitness = min(final_expected_fitness, n)
        
        return OMState(
            fitness=int(final_expected_fitness),
            n=n,
            step=state.step + 1,
            budget=state.budget,
            normalized_fitness=final_expected_fitness / n
        )
    
    def get_legal_actions(self, state: OMState) -> list:
        n = state.n
        i = state.fitness
        
        # Always include k=1 as it's safe near optimum
        actions = [OMAction(k=1, name="flip_1")]
        
        # Add other reasonable k values based on current fitness
        if i < n // 2:
            # First phase: include larger k values
            for k in [n//4, n//2, n-i, n//2 + n//4]:
                if 1 < k <= n and k not in [a.k for a in actions]:
                    actions.append(OMAction(k=k, name=f"flip_{k}"))
        elif i < (n * 3) // 4:
            # Transition phase: medium k values
            for k in [n//8, n//4, n//2]:
                if 1 < k <= n and k not in [a.k for a in actions]:
                    actions.append(OMAction(k=k, name=f"flip_{k}"))
        else:
            # Near optimum: only small k values
            for k in [2, 3, n//10]:
                if 1 < k <= n and k not in [a.k for a in actions]:
                    actions.append(OMAction(k=k, name=f"flip_{k}"))
        
        # Add some common values from the statistics
        for k in [10, 12, 14, 25]:
            if k <= n and k not in [a.k for a in actions]:
                actions.append(OMAction(k=k, name=f"flip_{k}"))
        
        return actions
    
    def evaluate_state(self, state: OMState) -> float:
        # Use normalized_fitness as the primary metric
        # This allows discrimination between states with same integer fitness
        # but different expected improvement potential
        
        base_score = state.normalized_fitness * 100
        
        # Add bonus for being close to optimum
        if state.normalized_fitness > 0.95:
            base_score += 10
        elif state.normalized_fitness > 0.9:
            base_score += 5
        
        # Small penalty for using up budget
        budget_factor = 1.0 - (state.step / state.budget) * 0.1
        
        return base_score * budget_factor
    
    def is_terminal(self, state: OMState) -> bool:
        # Terminal if we've reached the optimum or exhausted the budget
        return state.fitness >= state.n or state.step >= state.budget