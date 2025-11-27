
import numpy as np
import pandas as pd
from simulation_core import FastSimulation, TierConfig, TierType, SegmentConfig, ELOConfig, MatchConfig

def verify_reset_logic():
    print("=== Verifying Season Reset Logic ===")
    
    # 1. Setup Configuration
    num_users = 100
    placement_matches = 5
    
    tier_configs = [
        TierConfig("Bronze", TierType.MMR, 0, 1200, placement_min_mmr=0, placement_max_mmr=1200),
        TierConfig("Silver", TierType.MMR, 1200, 1400, placement_min_mmr=1200, placement_max_mmr=1400),
        TierConfig("Gold", TierType.MMR, 1400, 1600, placement_min_mmr=1400, placement_max_mmr=1600)
    ]
    
    elo_config = ELOConfig(base_k=32, placement_matches=placement_matches, placement_bonus=2.0)
    match_config = MatchConfig()
    segment_configs = [SegmentConfig("Test", 1.0, 1.0, 10, 10, 1000, 1000, 0, 24)]
    
    # 2. Initialize Simulation
    sim = FastSimulation(num_users, segment_configs, elo_config, match_config, tier_configs, initial_mmr=1000)
    sim.initialize_users()
    
    # 3. Run Matches (Season 1)
    print("\nRunning Season 1...")
    sim.run_day() # 10 matches per user -> Placement done
    
    # Verify everyone is ranked
    unranked_count = np.sum(sim.user_tier_index == -1)
    if unranked_count > 0:
        print(f"FAIL: {unranked_count} users unranked after Season 1.")
        return
    else:
        print("PASS: All users ranked after Season 1.")
        
    # 4. Apply Soft Reset
    print("\nApplying Soft Reset...")
    sim.apply_soft_reset(compression_factor=0.5, target_mean=1000)
    
    # 5. Verify Reset State
    # Should be Unranked (-1)
    unranked_count = np.sum(sim.user_tier_index == -1)
    if unranked_count != num_users:
        print(f"FAIL: {num_users - unranked_count} users still ranked after reset.")
        print(f"Tier distribution: {np.unique(sim.user_tier_index, return_counts=True)}")
    else:
        print("PASS: All users reset to Unranked (-1).")
        
    # Should have 0 matches played
    max_matches = np.max(sim.matches_played)
    if max_matches > 0:
        print(f"FAIL: Matches played not reset (Max: {max_matches})")
    else:
        print("PASS: Matches played reset to 0.")
        
    # 6. Run Matches (Season 2)
    print("\nRunning Season 2 (Placement)...")
    sim.run_day() # 10 matches -> Placement done again
    
    # Verify everyone is ranked again
    unranked_count = np.sum(sim.user_tier_index == -1)
    if unranked_count > 0:
        print(f"FAIL: {unranked_count} users unranked after Season 2 placement.")
    else:
        print("PASS: All users ranked after Season 2.")

if __name__ == "__main__":
    verify_reset_logic()
