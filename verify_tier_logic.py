import numpy as np
import pandas as pd
from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig, TierConfig, TierType, User

def test_ladder_multipliers():
    print("Testing Ladder Multipliers...")
    
    # Setup
    tier_configs = [
        TierConfig(
            name="Bronze", type=TierType.LADDER, min_mmr=0, max_mmr=1000,
            points_win=10, points_draw=5, promotion_points=100,
            promotion_mmr_2=800, promotion_mmr_3=600, promotion_mmr_4=400, promotion_mmr_5=200
        )
    ]
    
    sim = FastSimulation(
        num_users=5,
        segment_configs=[SegmentConfig("Test", 1.0, 1.0, 1, 1, 500, 500)],
        elo_config=ELOConfig(),
        match_config=MatchConfig(),
        tier_configs=tier_configs,
        initial_mmr=500
    )
    
    # Manually set MMRs to test multipliers
    sim.mmr[0] = 100 # Should get 5x ( < 200)
    sim.mmr[1] = 300 # Should get 4x ( < 400)
    sim.mmr[2] = 500 # Should get 3x ( < 600)
    sim.mmr[3] = 700 # Should get 2x ( < 800)
    sim.mmr[4] = 900 # Should get 1x ( >= 800)
    
    # Simulate Wins for all
    # We need to call _process_tier_updates directly or simulate a match where they win.
    # Let's mock the inputs for _process_tier_updates
    
    indices = np.array([0, 1, 2, 3, 4])
    results = np.array([1, 1, 1, 1, 1]) # All win
    mmr_changes = np.zeros(5) # Irrelevant for Ladder points
    
    # We need to construct the arguments for _process_tier_updates
    # It takes (idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b)
    # This is for a batch of matches.
    # Let's just manually invoke the logic inside _process_tier_updates or simpler:
    # Create a dummy match where 0 plays 0 (impossible but for logic test ok)
    # Or just use internal logic snippet.
    
    # Let's use the actual method.
    # We need pairs.
    # Let's pair them with dummy losers.
    # But we can just check the logic by inspecting the code or running a day.
    # Running a day is complex due to matchmaking.
    
    # Let's just replicate the logic here to verify it matches expectation, 
    # OR better, use a helper method if I exposed one? No.
    
    # Let's try to run a manual update step.
    # We can use `_process_tier_updates` if we format data correctly.
    # It expects arrays of indices for A and B.
    
    # Let's say 0 beats 5 (dummy), 1 beats 6, etc.
    # We need more users.
    
    sim = FastSimulation(
        num_users=10,
        segment_configs=[SegmentConfig("Test", 1.0, 1.0, 1, 1, 500, 500)],
        elo_config=ELOConfig(),
        match_config=MatchConfig(),
        tier_configs=tier_configs,
        initial_mmr=500
    )
    
    sim.mmr[0] = 100
    sim.mmr[1] = 300
    sim.mmr[2] = 500
    sim.mmr[3] = 700
    sim.mmr[4] = 900
    
    # Initialize Tier Index (Crucial!)
    sim.user_tier_index[:] = 0 # Bronze
    
    # 0 vs 5 (0 wins)
    # 1 vs 6 (1 wins)
    # 2 vs 7 (2 wins)
    # 3 vs 8 (3 wins)
    # 4 vs 9 (4 wins)
    
    idx_a = np.array([0, 1, 2, 3, 4])
    idx_b = np.array([5, 6, 7, 8, 9])
    win_a = np.array([0, 1, 2, 3, 4]) # Indices into idx_a that won? No.
    # _process_tier_updates(self, idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b)
    # win_a is boolean mask? No, "res_a[win_a] = 1". It's indices of matches?
    # Let's check simulation_core.py line 50: res_a[win_a] = 1.
    # win_a is indices of matches where A won.
    
    match_indices_win = np.array([0, 1, 2, 3, 4]) # All matches A won
    match_indices_draw = np.array([], dtype=int)
    match_indices_loss = np.array([], dtype=int)
    
    mmr_change_a = np.zeros(5)
    mmr_change_b = np.zeros(5)
    
    sim._process_tier_updates(idx_a, idx_b, match_indices_win, match_indices_draw, match_indices_loss, mmr_change_a, mmr_change_b)
    
    # Check Points
    # Base win = 10
    print(f"User 0 (MMR 100, <200): Points {sim.user_ladder_points[0]} (Expected 50)")
    print(f"User 1 (MMR 300, <400): Points {sim.user_ladder_points[1]} (Expected 40)")
    print(f"User 2 (MMR 500, <600): Points {sim.user_ladder_points[2]} (Expected 30)")
    print(f"User 3 (MMR 700, <800): Points {sim.user_ladder_points[3]} (Expected 20)")
    print(f"User 4 (MMR 900, >=800): Points {sim.user_ladder_points[4]} (Expected 10)")
    
    assert sim.user_ladder_points[0] == 50
    assert sim.user_ladder_points[1] == 40
    assert sim.user_ladder_points[2] == 30
    assert sim.user_ladder_points[3] == 20
    assert sim.user_ladder_points[4] == 10
    print("Ladder Multipliers Test Passed!")

def test_ladder_loss_points():
    print("\nTesting Ladder Points Loss & Demotion...")
    tier_configs = [
        TierConfig(name="Bronze", type=TierType.MMR, min_mmr=0, max_mmr=1000),
        TierConfig(
            name="Silver", type=TierType.LADDER, min_mmr=1000, max_mmr=2000,
            demotion_mmr=1100, demotion_lives=2, points_win=10, points_loss=5
        )
    ]
    
    sim = FastSimulation(
        num_users=4,
        segment_configs=[SegmentConfig("Test", 1.0, 1.0, 1, 1, 1000, 1000)],
        elo_config=ELOConfig(),
        match_config=MatchConfig(),
        tier_configs=tier_configs,
        initial_mmr=1050
    )
    
    # Initialize Tier Index
    sim.user_tier_index[:] = 1 # Silver
    sim.user_demotion_lives[:] = 2
    
    # User 0: 10 Points. User 1: 0 Points.
    sim.user_ladder_points[0] = 10
    sim.user_ladder_points[1] = 0
    
    idx_a = np.array([0, 1])
    idx_b = np.array([2, 3]) # Dummy opponents
    loss_a = np.array([0, 1]) # Both Lose
    
    # Match 1: Loss
    # User 0: 10 -> 5. Lives 2.
    # User 1: 0 -> 0. Lives 2 -> 1 (At Risk).
    
    sim._process_tier_updates(idx_a, idx_b, np.array([], dtype=int), np.array([], dtype=int), loss_a, np.zeros(2), np.zeros(2))
    
    print(f"User 0 Points: {sim.user_ladder_points[0]} (Expected 5)")
    print(f"User 1 Points: {sim.user_ladder_points[1]} (Expected 0)")
    print(f"User 1 Lives: {sim.user_demotion_lives[1]} (Expected 1)")
    
    assert sim.user_ladder_points[0] == 5
    assert sim.user_ladder_points[1] == 0
    assert sim.user_demotion_lives[1] == 1
    
    # Match 2: Loss
    # User 0: 5 -> 0. Lives 2.
    # User 1: 0 -> 0. Lives 1 -> 0 -> Demote to Bronze (0).
    
    sim._process_tier_updates(idx_a, idx_b, np.array([], dtype=int), np.array([], dtype=int), loss_a, np.zeros(2), np.zeros(2))
    
    print(f"User 0 Points: {sim.user_ladder_points[0]} (Expected 0)")
    print(f"User 1 Tier: {sim.user_tier_index[1]} (Expected 0 - Bronze)")
    
    assert sim.user_ladder_points[0] == 0
    assert sim.user_tier_index[1] == 0
    
    print("Ladder Points Loss Test Passed!")

def test_elo_tier():
    print("\nTesting ELO Tier...")
    tier_configs = [
        TierConfig(name="Bronze", type=TierType.ELO, min_mmr=0, max_mmr=1000),
        TierConfig(name="Silver", type=TierType.ELO, min_mmr=1000, max_mmr=2000)
    ]
    
    sim = FastSimulation(
        num_users=2,
        segment_configs=[SegmentConfig("Test", 1.0, 1.0, 1, 1, 500, 500)],
        elo_config=ELOConfig(),
        match_config=MatchConfig(),
        tier_configs=tier_configs,
        initial_mmr=500
    )
    
    sim.user_tier_index[0] = 0 # Bronze
    sim.mmr[0] = 1100 # Promotable
    
    sim.user_tier_index[1] = 1 # Silver
    sim.mmr[1] = 900 # Demotable
    
    # Trigger update
    # ELO tier updates happen on match processing
    idx_a = np.array([0, 1])
    idx_b = np.array([0, 1])
    # Result doesn't matter for ELO promotion trigger, just the check
    sim._process_tier_updates(idx_a, idx_b, np.array([0]), np.array([]), np.array([1]), np.zeros(2), np.zeros(2))
    
    print(f"User 0 Tier: {sim.user_tier_index[0]} (Expected 1)")
    print(f"User 1 Tier: {sim.user_tier_index[1]} (Expected 0)")
    
    assert sim.user_tier_index[0] == 1
    assert sim.user_tier_index[1] == 0
    print("ELO Tier Test Passed!")

if __name__ == "__main__":
    test_ladder_multipliers()
    test_ladder_loss_points()
    test_elo_tier()
