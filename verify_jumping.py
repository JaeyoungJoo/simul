
import numpy as np
from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig, TierType

def test_jumping():
    # 1. Setup Simulation
    # Create a user with High MMR (1200) but place them in Low Tier (Prospect 3, Max 400)
    # Prospect 3: Points Win 10, Prom High 10.
    
    # We need to mimic the TierConfig from csv exactly or close enough
    # From csv: Prospect 3 (idx 0), Ladder, Max 400. Win 10. Prom High 10.
    
    # Actually, we can just load the real config if we want, but mocking is safer/faster.
    from app import load_config
    # Mocking TierConfig for simplicity based on user description
    from dataclasses import dataclass
    
    @dataclass
    class MockTier:
        name: str
        type: TierType
        min_mmr: int
        max_mmr: int
        points_win: int = 10
        points_draw: int = 5
        points_loss: int = 5
        promotion_points: int = 100
        promotion_points_low: int = 100
        promotion_points_high: int = 10 # <--- Key Factor
        demotion_lives: int = 3
        demotion_mmr: int = 0
        placement_min_mmr: int = 0
        placement_max_mmr: int = 0
        capacity: int = 0
        promotion_mmr_2: int = 0
        promotion_mmr_3: int = 0
        promotion_mmr_4: int = 0
        promotion_mmr_5: int = 0

    tiers = [
        MockTier("Tier 1", TierType.LADDER, 0, 400, 10, 5, 5, 100, 100, 10), # Prom High = 10
        MockTier("Tier 2", TierType.LADDER, 401, 800, 10, 5, 5, 100, 100, 20),
        MockTier("Tier 3", TierType.LADDER, 801, 1200, 10, 5, 5, 100, 100, 30)
    ]
    
    config = ELOConfig()
    match_config = MatchConfig()
    seg = SegmentConfig("Test", 1.0, 1.0, 1.0, 1.0, 1200, 1200) # MMR 1200
    
    sim = FastSimulation(2, [seg], config, match_config, tiers, initial_mmr=1200)
    
    # Force Tier 0 (Prospect 3 equivalent) logic bypass
    # We need to manually set them because init might put them at -1?
    # Let's say we set them to Tier 0 manually to simulate "Placed Low but High MMR"
    sim.user_tier_index[:] = 0
    sim.user_ladder_points[:] = 0
    
    print(f"Start: Tier {sim.user_tier_index[0]}, MMR {sim.mmr[0]}, Points {sim.user_ladder_points[0]}")
    
    # Run 1 Match (Win)
    # We can use _simulate_batch_matches... no that involves random pairing.
    # Let's trace logic manually or try to force a win?
    # We can just check the logic unit: _update_single_batch
    
    # Simulate A wins against B
    indices = np.array([0, 1]) # Both High MMR (1200) in Tier 0 (Max 400)
    results = np.array([1, -1]) # 0 Wins, 1 Loses
    current_mmrs = sim.mmr[indices]
    
    print("\n--- Processing Batch (Win for User 0) ---")
    sim._update_single_batch(indices, results, current_mmrs)
    
    print(f"After: Tier {sim.user_tier_index[0]}, MMR {sim.mmr[0]}, Points {sim.user_ladder_points[0]}")
    
    # Check if promoted
    if sim.user_tier_index[0] == 1:
        print("RESULT: PROMOTED IMMEDIATELY! (Jumping confirmed)")
        print("Reason: 1200 MMR > 400 Max MMR -> Uses 'promotion_points_high' (10)")
        print("Win Points (10) >= Target (10) -> Promote.")
    else:
        print("RESULT: Not promoted.")

if __name__ == "__main__":
    test_jumping()
