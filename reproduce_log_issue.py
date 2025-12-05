import numpy as np
import pandas as pd
from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig, TierConfig, TierType

def test_log_consistency():
    print("--- Testing Log Consistency ---")
    
    # 1. Setup Simulation
    num_users = 100
    seg_config = SegmentConfig(
        name="TestSeg", 
        ratio=1.0, 
        daily_play_prob=1.0, # Play every day
        matches_per_day_min=5, # Multiple matches
        matches_per_day_max=5,
        true_skill_min=1000, 
        true_skill_max=1200
    )
    
    elo_config = ELOConfig(base_k=32)
    match_config = MatchConfig()
    tier_config = TierConfig(name="Bronze", type=TierType.MMR)
    
    sim = FastSimulation(
        num_users=num_users,
        segment_configs=[seg_config],
        elo_config=elo_config,
        match_config=match_config,
        tier_configs=[tier_config],
        initial_mmr=1000
    )
    
    # Force watch user 0
    target_idx = 0
    sim.watched_indices[target_idx] = "TestSeg"
    sim.match_logs[target_idx] = []
    
    # 2. Run Simulation
    print(f"Running simulation for 5 days...")
    for day in range(1, 6):
        sim.run_day(day)
        
    # 3. Analyze Logs
    logs = sim.match_logs.get(target_idx, [])
    print(f"Total Logs for User {target_idx}: {len(logs)}")
    
    if not logs:
        print("!! NO LOGS FOUND !!")
        return

    # Check Consistency
    prev_new_mmr = 1000.0 # Initial
    
    for i, log in enumerate(logs):
        # 1. Row Math Check
        # User UI Logic: Pre = New - Change
        # Logged Current: Post-match (New)
        # Logged Change: Delta
        
        # Verify if log.current_mmr is indeed Post-match
        # If it is post match, then Pre = Current - Change.
        
        pre_mmr_calc = log.current_mmr - log.mmr_change
        
        print(f"[{i}] Day {log.day}: Change {log.mmr_change:+.4f} | New {log.current_mmr:.4f} | Calc Pre {pre_mmr_calc:.4f}")
        
        # 2. Sequential Check
        # The 'New' MMR from previous match should equal 'Pre' MMR of this match
        diff = abs(pre_mmr_calc - prev_new_mmr)
        if diff > 0.0001:
            print(f"   >>> SEQUENCE ERROR! Prev New {prev_new_mmr:.4f} != This Pre {pre_mmr_calc:.4f} (Diff {diff:.4f})")
            
        prev_new_mmr = log.current_mmr

    # Final MMR Check
    print(f"Final Sim MMR: {sim.mmr[target_idx]:.4f}")
    if abs(sim.mmr[target_idx] - prev_new_mmr) > 0.0001:
         print(f"   >>> FINAL MMR MISMATCH! Sim {sim.mmr[target_idx]:.4f} != Last Log {prev_new_mmr:.4f}")

if __name__ == "__main__":
    test_log_consistency()
