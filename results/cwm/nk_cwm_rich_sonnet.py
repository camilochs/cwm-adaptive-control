class SynthesizedCWM:
    def __init__(self):
        # Empirical transition table: (fitness_range_min, fitness_range_max) -> {k -> (p_improve, mean_delta_f)}
        self.transition_table = {
            (20.0, 25.0): {
                1: (0.508, 0.3067),
                5: (0.554, 0.6835),
                10: (0.500, 0.5725),
                15: (0.636, 1.3444),
                25: (0.865, 2.2194)
            },
            (25.0, 28.0): {
                1: (0.379, 0.2206),
                2: (0.381, 0.2260),
                3: (0.429, 0.3673),
                5: (0.279, 0.2485),
                7: (0.296, 0.1673),
                10: (0.339, 0.4979),
                15: (0.303, 0.2821),
                25: (0.182, 0.1577),
                50: (0.064, 0.0603)
            },
            (28.0, 30.0): {
                1: (0.251, 0.1373),
                2: (0.257, 0.1360),
                3: (0.222, 0.1372),
                5: (0.159, 0.1259),
                7: (0.117, 0.0871),
                10: (0.090, 0.0778),
                15: (0.031, 0.0242),
                25: (0.019, 0.0132),
                50: (0.005, 0.0023)
            },
            (30.0, 32.0): {
                1: (0.166, 0.0830),
                2: (0.145, 0.0901),
                3: (0.119, 0.0718),
                5: (0.065, 0.0460),
                7: (0.022, 0.0139),
                10: (0.017, 0.0088),
                15: (0.006, 0.0032),
                25: (0.001, 0.0012),
                50: (0.000, 0.0000)
            },
            (32.0, 34.0): {
                1: (0.012, 0.0053),
                2: (0.093, 0.0421),
                3: (0.051, 0.0223),
                5: (0.017, 0.0091),
                7: (0.006, 0.0038),
                10: (0.001, 0.0003),
                15: (0.001, 0.0006),
                25: (0.000, 0.0000),
                50: (0.000, 0.0000)
            },
            (34.0, 36.0): {
                1: (0.002, 0.0007),
                2: (0.033, 0.0117),
                3: (0.009, 0.0026),
                5: (0.002, 0.0008),
                7: (0.001, 0.0002),
                10: (0.000, 0.0000),
                15: (0.000, 0.0000),
                25: (0.000, 0.0000),
                50: (0.000, 0.0000)
            },
            (36.0, 38.0): {
                1: (0.001, 0.0004),
                2: (0.005, 0.0011),
                3: (0.001, 0.0002),
                5: (0.000, 0.0001),
                7: (0.000, 0.0000),
                10: (0.000, 0.0000),
                15: (0.000, 0.0000),
                25: (0.000, 0.0000),
                50: (0.000, 0.0000)
            },
            (38.0, 42.0): {
                1: (0.006, 0.0004),
                2: (0.000, 0.0000),
                3: (0.003, 0.0007),
                5: (0.000, 0.0000),
                7: (0.000, 0.0000),
                10: (0.000, 0.0000),
                15: (0.000, 0.0000),
                25: (0.000, 0.0000),
                50: (0.000, 0.0000)
            }
        }
    
    def _get_fitness_range(self, fitness):
        """Map fitness to appropriate range from transition table."""
        if fitness < 25.0:
            return (20.0, 25.0)
        elif fitness < 28.0:
            return (25.0, 28.0)
        elif fitness < 30.0:
            return (28.0, 30.0)
        elif fitness < 32.0:
            return (30.0, 32.0)
        elif fitness < 34.0:
            return (32.0, 34.0)
        elif fitness < 36.0:
            return (34.0, 36.0)
        elif fitness < 38.0:
            return (36.0, 38.0)
        else:
            return (38.0, 42.0)
    
    def _lookup_transition_stats(self, fitness, k):
        """Look up P(improve) and mean_delta_f for given fitness and k."""
        fitness_range = self._get_fitness_range(fitness)
        range_data = self.transition_table[fitness_range]
        
        # Find exact match or closest k
        if k in range_data:
            return range_data[k]
        
        # Find closest k value for interpolation
        available_ks = sorted(range_data.keys())
        if k < available_ks[0]:
            return range_data[available_ks[0]]
        elif k > available_ks[-1]:
            return range_data[available_ks[-1]]
        else:
            # Simple interpolation between closest k values
            lower_k = max([kk for kk in available_ks if kk <= k])
            upper_k = min([kk for kk in available_ks if kk > k])
            
            lower_stats = range_data[lower_k]
            upper_stats = range_data[upper_k]
            
            # Linear interpolation
            weight = (k - lower_k) / (upper_k - lower_k)
            p_improve = lower_stats[0] + weight * (upper_stats[0] - lower_stats[0])
            mean_delta = lower_stats[1] + weight * (upper_stats[1] - lower_stats[1])
            
            return (p_improve, mean_delta)
    
    def predict_next_state(self, state, action):
        """Predict next state based on empirical transition statistics."""
        p_improve, mean_delta = self._lookup_transition_stats(state.fitness, action.k)
        
        # Expected fitness change
        expected_fitness = state.fitness + mean_delta
        
        # Determine if we expect improvement (stochastic decision based on probability)
        # For prediction, we use the expected value
        will_improve = p_improve > 0.5  # Simple threshold-based prediction
        
        # Update stagnation counter
        new_stagnation = 0 if will_improve else state.stagnation_counter + 1
        
        # Ensure fitness stays within reasonable bounds
        expected_fitness = max(0.0, min(expected_fitness, 50.0))
        
        # Recalculate normalized fitness (assuming max possible is around 44)
        new_normalized_fitness = expected_fitness / 44.0
        
        return NKState(
            fitness=expected_fitness,
            n=state.n,
            K=state.K,
            step=state.step + 1,
            budget=state.budget,
            normalized_fitness=new_normalized_fitness,
            stagnation_counter=new_stagnation
        )
    
    def get_legal_actions(self, state):
        """Get legal actions, adapting to stagnation level."""
        actions = []
        
        # Always include k=1
        actions.append(NKAction(k=1, name="flip_1"))
        
        # Add more disruptive mutations based on stagnation
        if state.stagnation_counter < 100:
            # Low stagnation: conservative mutations
            actions.extend([
                NKAction(k=2, name="flip_2"),
                NKAction(k=3, name="flip_3"),
                NKAction(k=5, name="flip_5")
            ])
        elif state.stagnation_counter < 1000:
            # Medium stagnation: moderate mutations
            actions.extend([
                NKAction(k=5, name="flip_5"),
                NKAction(k=7, name="flip_7"),
                NKAction(k=10, name="flip_10")
            ])
        else:
            # High stagnation: aggressive mutations
            actions.extend([
                NKAction(k=10, name="flip_10"),
                NKAction(k=15, name="flip_15"),
                NKAction(k=25, name="flip_25")
            ])
        
        # Ensure k values don't exceed n
        actions = [a for a in actions if a.k <= state.n]
        
        return actions
    
    def evaluate_state(self, state):
        """Evaluate state quality: reward high fitness, penalize stagnation."""
        # Base score from normalized fitness
        fitness_score = state.normalized_fitness
        
        # Stagnation penalty: exponential decay based on stagnation counter
        stagnation_penalty = 0.1 * (1.0 - 1.0 / (1.0 + state.stagnation_counter / 1000.0))
        
        # Budget penalty: slight penalty as we approach budget limit
        budget_remaining = (state.budget - state.step) / state.budget
        budget_penalty = 0.05 * (1.0 - budget_remaining)
        
        return fitness_score - stagnation_penalty - budget_penalty
    
    def is_terminal(self, state):
        """Check if state is terminal (budget exhausted)."""
        return state.step >= state.budget