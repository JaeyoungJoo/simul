
import numpy as np
from simulation_core import FastSimulation, TierConfig, TierType, ELOConfig, MatchConfig, SegmentConfig

def test_overhaul_logic():
    print("Testing Rank System Overhaul Logic...")

    # Setup Configs
    elo_config = ELOConfig(base_k=32, placement_matches=0, calibration_enabled=False)
    match_config = MatchConfig()
    
    # Tier 1: Ladder (Lives=3, Prom=100, Low=50, High=150)
    # Tier 2: MMR (Lives=2, Prom=100, Low=50, High=150) -> Note: MMR type now uses points!
    tier_configs = [
        TierConfig(name="Bronze", type=TierType.LADDER, min_mmr=0, max_mmr=1000, 
                   points_win=10, points_draw=5, promotion_points=100, 
                   promotion_points_low=50, promotion_points_high=150, demotion_lives=0),
        TierConfig(name="Silver", type=TierType.LADDER, min_mmr=1000, max_mmr=2000, 
                   points_win=10, points_draw=5, promotion_points=100, 
                   promotion_points_low=50, promotion_points_high=150, demotion_lives=3),
        TierConfig(name="Gold", type=TierType.MMR, min_mmr=2000, max_mmr=3000, 
                   promotion_points=100, promotion_points_low=50, promotion_points_high=150, demotion_lives=2)
    ]
    
    segment_configs = [SegmentConfig(name="General", ratio=1.0, daily_play_prob=1.0, matches_per_day_min=1, matches_per_day_max=1, true_skill_min=1000, true_skill_max=1000)]
    
    sim = FastSimulation(num_users=10, segment_configs=segment_configs, elo_config=elo_config, match_config=match_config, tier_configs=tier_configs, initial_mmr=1500, point_convergence_rate=0.5)
    
    # Initialize Users
    # User 0: Silver (Ladder), MMR 1500 (Normal Range). Lives=3.
    sim.user_tier_index[0] = 1
    sim.mmr[0] = 1500
    sim.user_demotion_lives[0] = 3
    
    # User 1: Silver (Ladder), MMR 900 (Low Range). Lives=3.
    sim.user_tier_index[1] = 1
    sim.mmr[1] = 900
    sim.user_demotion_lives[1] = 3
    
    # User 2: Gold (MMR), MMR 2500. Lives=2.
    sim.user_tier_index[2] = 2
    sim.mmr[2] = 2500
    sim.user_demotion_lives[2] = 2
    
    # --- Test 1: Ladder Promotion Targets ---
    print("\n--- Test 1: Ladder Promotion Targets ---")
    # User 0 (Normal): Needs 100 points. Win gives 10.
    # User 1 (Low): Needs 50 points. Win gives 10.
    
    # Simulate a win for User 0 and User 1
    # We'll manually trigger _process_tier_updates to control inputs
    idx_a = np.array([0, 1])
    idx_b = np.array([3, 4]) # Dummy opponents
    win_a = np.array([0, 1]) # Both win
    draw = np.array([], dtype=int)
    loss_a = np.array([], dtype=int)
    mmr_change_a = np.array([20.0, 20.0]) # Dummy MMR change
    mmr_change_b = np.array([-20.0, -20.0])
    
    sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b)
    
    print(f"User 0 Points: {sim.user_ladder_points[0]} (Expected 10)")
    print(f"User 1 Points: {sim.user_ladder_points[1]} (Expected 10)")
    
    # Boost points to check promotion
    sim.user_ladder_points[0] = 90
    sim.user_ladder_points[1] = 40
    
    # Win again
    sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b)
    
    print(f"User 0 Points: {sim.user_ladder_points[0]} (Expected 100 -> Promote?)") 
    # Wait, promotion resets points to 0.
    print(f"User 0 Tier: {sim.user_tier_index[0]} (Expected 2 - Gold)")
    print(f"User 1 Tier: {sim.user_tier_index[1]} (Expected 2 - Gold)")
    
    if sim.user_tier_index[0] == 2 and sim.user_tier_index[1] == 2:
        print("SUCCESS: Ladder Promotion Logic (Normal/Low Targets) works.")
    else:
        print("FAILURE: Ladder Promotion Logic failed.")

    # --- Test 2: Demotion Lives ---
    print("\n--- Test 2: Demotion Lives ---")
    # Reset User 0 to Silver, Lives=1
    sim.user_tier_index[0] = 1
    sim.user_demotion_lives[0] = 1
    
    # Loss
    win_a = np.array([], dtype=int)
    loss_a = np.array([0]) # User 0 loses
    idx_a = np.array([0])
    idx_b = np.array([3])
    
    sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a, mmr_change_a[:1], mmr_change_b[:1])
    
    print(f"User 0 Lives: {sim.user_demotion_lives[0]} (Expected 0)")
    print(f"User 0 Tier: {sim.user_tier_index[0]} (Expected 0 - Bronze)")
    
    if sim.user_tier_index[0] == 0:
        print("SUCCESS: Demotion Lives Logic works.")
    else:
        print("FAILURE: Demotion Lives Logic failed.")
        
    # --- Test 3: MMR Points Convergence ---
    print("\n--- Test 3: MMR Points Convergence ---")
    # User 2 is Gold (MMR Type). Convergence = 0.5.
    # MMR Change = 20. Points Change should be 10.
    
    sim.user_ladder_points[2] = 0
    idx_a = np.array([2])
    idx_b = np.array([5])
    win_a = np.array([0])
    loss_a = np.array([], dtype=int)
    mmr_change_a = np.array([20.0])
    
    sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b[:1])
    
    print(f"User 2 Points: {sim.user_ladder_points[2]} (Expected 10)")
    
    if sim.user_ladder_points[2] == 10:
        print("SUCCESS: MMR Points Convergence works.")
    else:
        print(f"FAILURE: MMR Points Convergence failed. Got {sim.user_ladder_points[2]}")

if __name__ == "__main__":
    test_overhaul_logic()
