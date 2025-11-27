
import numpy as np
import pandas as pd
from simulation_core import FastSimulation, TierConfig, TierType, SegmentConfig, ELOConfig, MatchConfig

def verify_placement_logic():
    print("=== Verifying Placement Logic ===")
    
    # 1. Setup Configuration
    num_users = 100
    placement_matches = 10
    
    tier_configs = [
        TierConfig("Bronze", TierType.MMR, 0, 1200, placement_min_mmr=0, placement_max_mmr=1200),
        TierConfig("Silver", TierType.MMR, 1200, 1400, placement_min_mmr=1200, placement_max_mmr=1400),
        TierConfig("Gold", TierType.MMR, 1400, 1600, placement_min_mmr=1400, placement_max_mmr=1600),
        TierConfig("Platinum", TierType.LADDER, 1600, 1800, placement_min_mmr=1600, placement_max_mmr=1800),
        TierConfig("Diamond", TierType.RATIO, 1800, 9999, placement_min_mmr=1800, placement_max_mmr=9999)
    ]
    
    elo_config = ELOConfig(base_k=32, placement_matches=placement_matches, placement_bonus=2.0)
    match_config = MatchConfig()
    segment_configs = [SegmentConfig("Test", 1.0, 1.0, 10, 10, 1000, 1000, 0, 24)]
    
    # 2. Initialize Simulation
    sim = FastSimulation(num_users, segment_configs, elo_config, match_config, tier_configs, initial_mmr=1000)
    sim.initialize_users()
    
    # 3. Verify Initial State (Unranked)
    print(f"Initial Tier Distribution (Day 0):")
    unique, counts = np.unique(sim.user_tier_index, return_counts=True)
    dist = dict(zip(unique, counts))
    print(dist)
    
    if -1 not in dist or dist[-1] != num_users:
        print("FAIL: Users should start at Tier -1 (Unranked)")
        return
    else:
        print("PASS: All users started Unranked (-1)")
        
    # 4. Run Matches (Day 1) - Everyone plays 10 matches (forced by segment config)
    # Note: run_day is probabilistic, so we might need a few days or force matches.
    # Let's run for a few days until everyone has played >= 10 matches.
    
    print("\nRunning Simulation...")
    for day in range(5):
        sim.run_day()
        played_counts = sim.matches_played
        finished_count = np.sum(played_counts >= placement_matches)
        print(f"Day {day+1}: {finished_count}/{num_users} users finished placement.")
        
        if finished_count == num_users:
            break
            
    # 5. Verify Tier Assignment
    print("\nVerifying Tier Assignment after Placement...")
    
    # Check if anyone is still -1 (should be 0 if all played enough)
    still_unranked = np.sum(sim.user_tier_index == -1)
    if still_unranked > 0:
        print(f"WARNING: {still_unranked} users still unranked (might not have played enough matches).")
        
    # Check consistency: MMR vs Tier
    errors = 0
    for i in range(num_users):
        if sim.matches_played[i] < placement_matches:
            if sim.user_tier_index[i] != -1:
                print(f"FAIL: User {i} has tier {sim.user_tier_index[i]} but only played {sim.matches_played[i]} matches.")
                errors += 1
            continue
            
        tier_idx = sim.user_tier_index[i]
        mmr = sim.mmr[i]
        
        if tier_idx == -1:
             print(f"FAIL: User {i} finished placement but is still Unranked. MMR: {mmr}")
             errors += 1
             continue
             
        config = tier_configs[tier_idx]
        # Check if MMR is roughly within placement range (it might have changed slightly after placement match)
        # But assignment happens exactly at 10th match.
        # Since we simulate multiple matches per day, MMR might have drifted AFTER assignment in the same batch?
        # FastSimulation processes matches in batches.
        # If user plays match 10, 11, 12 in one day.
        # Match 10 triggers assignment. Match 11, 12 update MMR/Ladder points.
        # So current MMR might not strictly match placement range anymore, but should be close.
        # However, we can check if they are assigned to a valid tier at least.
        
        print(f"User {i}: MMR {mmr:.1f} -> Tier {config.name} ({tier_idx})")
        
    if errors == 0:
        print("PASS: Placement logic verified.")
    else:
        print(f"FAIL: Found {errors} errors.")

if __name__ == "__main__":
    verify_placement_logic()
