
import numpy as np
from simulation_core import FastSimulation, TierConfig, TierType, ELOConfig, MatchConfig

def test_loss_point_correction():
    print("Testing Loss Point Correction...")
    
    # Setup Tier Config with correction
    # Tier 0: Normal (1.0)
    # Tier 1: Corrected (0.5)
    tiers = [
        TierConfig(name="Normal Tier", type=TierType.MMR, min_mmr=0, max_mmr=1000, loss_point_correction=1.0),
        TierConfig(name="Corrected Tier", type=TierType.MMR, min_mmr=1000, max_mmr=2000, loss_point_correction=0.5)
    ]
    
    # Mock Configs
    elo_config = ELOConfig()
    match_config = MatchConfig()
    segment_configs = [] # Empty list is fine for this test
    
    sim = FastSimulation(num_users=2, tier_configs=tiers, initial_mmr=1500, 
                         segment_configs=segment_configs, elo_config=elo_config, match_config=match_config)
    
    # Manually set user tiers
    # User 0 in Tier 0 (Normal)
    # User 1 in Tier 1 (Corrected)
    sim.user_tier_index[0] = 0
    sim.mmr[0] = 500
    sim.user_tier_index[1] = 1
    sim.mmr[1] = 1500
    
    # Mock update inputs
    # Both lose 20 MMR
    idx_a = np.array([0])
    idx_b = np.array([1])
    win_a = np.array([])
    draw = np.array([])
    loss_a = np.array([0]) # User 0 loses against User 1 (Wait, idx_b is opponent?)
    # Let's just use the internal logic of _process_tier_updates directly or mock the call
    # _process_tier_updates(self, idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b)
    
    # User 0 (Tier 0) loses 20 MMR
    # User 1 (Tier 1) loses 20 MMR (Hypothetically, let's say they played someone else or we force it)
    # To test both losing, we can pretend they played different matches or just pass arrays
    
    # Let's say User 0 played User 1 and both lost (impossible in 1 match, but function accepts arrays)
    # Actually, let's do: User 0 vs User 1. User 0 Wins. User 1 Loses.
    # User 0: +20 MMR. Tier 0 (Correction 1.0). Points should be +20.
    # User 1: -20 MMR. Tier 1 (Correction 0.5). Points should be -10.
    
    mmr_change_a = np.array([20.0])
    mmr_change_b = np.array([-20.0])
    
    # Reset points
    sim.user_ladder_points[:] = 0
    
    # Run update
    # idx_a=[0], idx_b=[1], win_a=[0] (index in idx_a), loss_a=[]
    sim._process_tier_updates(idx_a, idx_b, np.array([0], dtype=int), np.array([], dtype=int), np.array([], dtype=int), mmr_change_a, mmr_change_b)
    
    print(f"User 0 (Tier 0, Win) Points: {sim.user_ladder_points[0]}")
    print(f"User 1 (Tier 1, Loss) Points: {sim.user_ladder_points[1]}")
    
    assert sim.user_ladder_points[0] == 10, f"Expected 10 (20 * 0.5 convergence), got {sim.user_ladder_points[0]}" # Default convergence is 0.5
    assert sim.user_ladder_points[1] == -5, f"Expected -5 (-20 * 0.5 convergence * 0.5 correction), got {sim.user_ladder_points[1]}"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_loss_point_correction()
