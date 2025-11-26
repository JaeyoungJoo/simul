from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import numpy as np
import matplotlib.pyplot as plt

def run_convergence_test():
    print("Starting Convergence Test...")
    
    # 1. Config: 1000 users, single segment, wide skill range
    # Mean 1500, Std 300 -> Range roughly 600 to 2400
    segments = [
        SegmentConfig("Test", 1.0, 1.0, 10.0, 2.0, 1500.0, 300.0, 0, 24)
    ]
    
    # Standard ELO Config
    elo_config = ELOConfig(base_k=32, placement_matches=10, placement_bonus=2.0)
    match_config = MatchConfig()
    
    sim = FastSimulation(1000, segments, elo_config, match_config)
    sim.initialize_users()
    
    # FORCE START AT 1500 to test hypothesis
    sim.mmr = np.full(1000, 1500.0)
    
    # Track Mean Absolute Error (MAE) between True Skill and MMR
    mae_history = []
    days = 100
    
    print(f"Simulating {days} days...")
    for d in range(days):
        sim.run_day()
        
        # Calculate MAE
        # Note: MMR starts at 1000, True Skill centers at 1500.
        # We expect MMR to drift towards True Skill.
        mae = np.mean(np.abs(sim.mmr - sim.true_skill))
        mae_history.append(mae)
        
        if d % 10 == 0:
            print(f"Day {d}: MAE = {mae:.2f} (Avg MMR: {np.mean(sim.mmr):.2f})")
            
    print(f"Final MAE: {mae_history[-1]:.2f}")
    
    # Check correlation
    correlation = np.corrcoef(sim.mmr, sim.true_skill)[0, 1]
    print(f"Final Correlation (MMR vs TrueSkill): {correlation:.4f}")
    
    if correlation > 0.8:
        print("SUCCESS: High correlation, system is converging.")
    else:
        print("FAILURE: Low correlation, convergence issues.")

if __name__ == "__main__":
    run_convergence_test()
