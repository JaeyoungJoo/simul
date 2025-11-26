from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import numpy as np

def run_verification():
    print("Starting verification (Multi-Match Logic)...")
    
    # 1. Configs with HIGH match count
    segments = [
        SegmentConfig("HighVolume", 1.0, 1.0, 10.0, 0.0, 1500.0, 0.0, 0, 24)
    ]
    elo_config = ELOConfig()
    match_config = MatchConfig()
    
    # 2. Initialize
    sim = FastSimulation(num_users=1000, segment_configs=segments, elo_config=elo_config, match_config=match_config)
    sim.initialize_users()
    
    # 3. Run Day
    sim.run_day()
    
    # 4. Check Match Counts
    avg_matches = np.mean(sim.matches_played)
    max_matches = np.max(sim.matches_played)
    min_matches = np.min(sim.matches_played)
    
    print(f"Day 1 Stats:")
    print(f"  Avg Matches/User: {avg_matches:.2f}")
    print(f"  Max Matches/User: {max_matches}")
    print(f"  Min Matches/User: {min_matches}")
    
    if avg_matches > 1.5:
        print("SUCCESS: Multiple matches per day verified.")
    else:
        print("FAILURE: Matches per day too low (expected ~10).")

if __name__ == "__main__":
    run_verification()
