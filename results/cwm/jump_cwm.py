class SynthesizedCWM:
    def __init__(self):
        pass
    
    def _binomial_coefficient(self, n, k):
        """Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        k = min(k, n - k)  # Take advantage of symmetry
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    def _hypergeometric_mean(self, N, K, n_draw):
        """Expected number of successes in hypergeometric distribution"""
        if n_draw == 0:
            return 0
        return (K * n_draw) / N
    
    def _region_from_fitness(self, fitness, n, jump_k):
        """Determine which region a fitness value is in"""
        if fitness == n + jump_k:
            return "optimum"
        elif fitness >= jump_k and fitness <= n:
            return "normal"
        else:
            return "valley"
    
    def _ones_from_fitness(self, fitness, n, jump_k):
        """Convert fitness to number of ones"""
        region = self._region_from_fitness(fitness, n, jump_k)
        if region == "optimum":
            return n
        elif region == "normal":
            return fitness - jump_k
        else:  # valley
            return n - fitness
    
    def predict_next_state(self, state, action):
        """Predict the next state given current state and action"""
        current_ones = self._ones_from_fitness(state.fitness, state.n, state.jump_k)
        current_zeros = state.n - current_ones
        k = action.k
        
        # If already at optimum, stay there
        if state.fitness == state.n + state.jump_k:
            return JumpState(
                fitness=state.fitness,
                n=state.n,
                jump_k=state.jump_k,
                step=state.step + 1,
                budget=state.budget,
                normalized_fitness=state.normalized_fitness
            )
        
        expected_fitness = state.fitness
        
        # Handle different regions
        region = self._region_from_fitness(state.fitness, state.n, state.jump_k)
        
        if region == "normal":
            # In normal region, compute expected change via hypergeometric
            if k <= current_ones and k <= state.n:
                # Expected number of 1-bits flipped (out of k total flips)
                expected_ones_flipped = self._hypergeometric_mean(state.n, current_ones, k)
                # New ones = current_ones - ones_flipped + (k - ones_flipped)
                # = current_ones + k - 2 * ones_flipped
                expected_new_ones = current_ones + k - 2 * expected_ones_flipped
                
                # Check if we might reach optimum
                if state.fitness == state.n and k == state.jump_k:
                    # At valley edge, k=jump_k can reach optimum
                    prob_optimum = 1.0 / self._binomial_coefficient(state.n, state.jump_k)
                    expected_fitness = (1 - prob_optimum) * state.fitness + prob_optimum * (state.n + state.jump_k)
                else:
                    # Normal case - stay in normal region most likely
                    if expected_new_ones <= state.n - state.jump_k:
                        expected_fitness = state.jump_k + expected_new_ones
                    elif expected_new_ones >= state.n:
                        expected_fitness = state.n + state.jump_k  # Optimum
                    else:
                        # Might fall into valley
                        expected_fitness = state.n - expected_new_ones
                        
                    # Only accept if improvement
                    if expected_fitness < state.fitness:
                        expected_fitness = state.fitness
        
        elif region == "valley":
            # In valley, try to escape
            if k <= current_ones and k <= state.n:
                expected_ones_flipped = self._hypergeometric_mean(state.n, current_ones, k)
                expected_new_ones = current_ones + k - 2 * expected_ones_flipped
                
                if expected_new_ones >= state.n:
                    expected_fitness = state.n + state.jump_k  # Optimum
                elif expected_new_ones <= state.n - state.jump_k:
                    expected_fitness = state.jump_k + expected_new_ones  # Back to normal
                else:
                    expected_fitness = state.n - expected_new_ones  # Still in valley
                
                # Only accept if improvement
                if expected_fitness < state.fitness:
                    expected_fitness = state.fitness
        
        # Compute normalized fitness for continuous evaluation
        normalized_fitness = expected_fitness / (state.n + state.jump_k)
        
        return JumpState(
            fitness=int(expected_fitness),
            n=state.n,
            jump_k=state.jump_k,
            step=state.step + 1,
            budget=state.budget,
            normalized_fitness=normalized_fitness
        )
    
    def get_legal_actions(self, state):
        """Get list of legal actions for current state"""
        actions = []
        
        # Always include k=1 (basic mutation)
        actions.append(JumpAction(k=1, name="flip_1"))
        
        # Always include k=jump_k (critical for escaping valley edge)
        if state.jump_k != 1:
            actions.append(JumpAction(k=state.jump_k, name=f"flip_{state.jump_k}"))
        
        # Add some intermediate values
        for k in [2, 3, 5]:
            if k != state.jump_k and k != 1 and k <= state.n:
                actions.append(JumpAction(k=k, name=f"flip_{k}"))
        
        # Add larger mutations for diversity
        for k in [state.n // 4, state.n // 2, state.n]:
            if k not in [1, state.jump_k] and k >= 2:
                actions.append(JumpAction(k=k, name=f"flip_{k}"))
        
        return actions
    
    def evaluate_state(self, state):
        """Evaluate state quality (higher = better)"""
        # Use normalized fitness as base score
        base_score = state.normalized_fitness
        
        # Bonus for being at optimum
        if state.fitness == state.n + state.jump_k:
            return 1000.0
        
        # Penalty for being in valley (fitness < jump_k)
        if state.fitness < state.jump_k:
            valley_penalty = (state.jump_k - state.fitness) * 0.1
            base_score -= valley_penalty
        
        # Small penalty for steps taken (encourage efficiency)
        efficiency_bonus = max(0, 1.0 - state.step / state.budget) * 0.1
        
        return base_score + efficiency_bonus
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        # Terminal if optimum reached or budget exhausted
        return state.fitness == state.n + state.jump_k or state.step >= state.budget