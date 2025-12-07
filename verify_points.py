
import numpy as np
from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig, TierType, TierConfig

def test_points():
    print("--- Verifying Ladder Points ---")
    
    # Setup Tiers
    # Tier 0: Prospect 3 (Ladder)
    # ...
    # Tier 4: Semipro 3 (Ladder)
    tiers = []
    for i in range(5):
        tiers.append(TierConfig(
            name=f"Tier {i}",
            type=TierType.LADDER,
            min_mmr=i*100,
            max_mmr=(i+1)*100,
            points_win=10,
            points_draw=5,
            promotion_points=100
        ))
        
    elo_config = ELOConfig(placement_matches=0) # No placement to confuse things
    match_config = MatchConfig()
    seg = SegmentConfig("Test", 1.0, 1.0, 1.0, 1.0, 450.0, 450.0) # Tier 4 MMR = 400-500
    
    sim = FastSimulation(2, [seg], elo_config, match_config, tiers, initial_mmr=450.0)
    sim.initialize_users()
    
    # Force Tier 4
    sim.user_tier_index[:] = 4
    sim.user_ladder_points[:] = 0
    
    print(f"Initial: Tier {sim.user_tier_index[0]}, Points {sim.user_ladder_points[0]}")
    
    # Run 1 Match (Win for User 0)
    # Simulate batch manually to avoid random pairing issues
    indices = np.array([0, 1])
    results = np.array([1, -1]) # User 0 Wins
    current_mmrs = sim.mmr[indices]
    
    sim._update_single_batch(indices, results, current_mmrs)
    
    pts = sim.user_ladder_points[0]
    print(f"After Win: Tier {sim.user_tier_index[0]}, Points {pts}")
    
    if pts == 10:
        print("SUCCESS: Points added correctly.")
    else:
        print(f"FAILURE: Expected 10, got {pts}")

if __name__ == "__main__":
    test_points()
