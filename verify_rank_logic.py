import unittest
import numpy as np
from simulation_core import FastSimulation, TierConfig, TierType, SegmentConfig, ELOConfig, MatchConfig

class TestRankSystem(unittest.TestCase):
    def setUp(self):
        self.segment_configs = [SegmentConfig("Test", 1.0, 1.0, 1, 1, 1000, 1000)]
        self.elo_config = ELOConfig()
        self.match_config = MatchConfig()
        
    def test_ladder_logic(self):
        # Config: Bronze (MMR), Silver (Ladder)
        # Silver: Win=10, Prom=20, DemotionLives=2
        tiers = [
            TierConfig("Bronze", TierType.MMR, 0, 1000),
            TierConfig("Silver", TierType.LADDER, 1000, 2000, demotion_lives=2, points_win=10, promotion_points=20)
        ]
        sim = FastSimulation(2, self.segment_configs, self.elo_config, self.match_config, tiers, initial_mmr=1500)
        
        # Force users to Silver (Index 1)
        sim.user_tier_index[:] = 1
        sim.user_ladder_points[:] = 0
        sim.user_demotion_lives[:] = 0
        
        # User 0 Wins (vs User 1) -> +10 Points
        # User 1 Loses -> Lives +1
        
        # Mock match processing
        idx_a = np.array([0])
        idx_b = np.array([1])
        # Use Boolean Masks as per FastSimulation logic
        win_a = np.array([True]) 
        loss_a = np.array([False])
        draw = np.array([False])
        
        sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a)
        
        self.assertEqual(sim.user_ladder_points[0], 10)
        self.assertEqual(sim.user_tier_index[0], 1) # Still Silver
        
        self.assertEqual(sim.user_demotion_lives[1], 1) # 1 Loss
        self.assertEqual(sim.user_tier_index[1], 1) # Still Silver (Lives=1, Threshold=2)
        
        # Round 2: User 0 Wins again -> +10 Points (Total 20) -> Promote
        # User 1 Loses again -> Lives +1 (Total 2) -> Demote
        
        # Add Gold tier so promotion is possible
        sim.tier_configs.append(TierConfig("Gold", TierType.MMR, 2000, 3000))
        
        sim._process_tier_updates(idx_a, idx_b, win_a, draw, loss_a)
        
        print(f"DEBUG: User 0 Tier: {sim.user_tier_index[0]}, Points: {sim.user_ladder_points[0]}")
        print(f"DEBUG: User 1 Tier: {sim.user_tier_index[1]}, Lives: {sim.user_demotion_lives[1]}")
        
        # User 0: 20 Points -> Promoted to Tier 2 (Gold)
        self.assertEqual(sim.user_tier_index[0], 2) # Promoted
        self.assertEqual(sim.user_ladder_points[0], 0) # Reset
        
        # User 1: Lives was 1. +1 = 2. Threshold=2. 2 >= 2 -> Demote.
        self.assertEqual(sim.user_tier_index[1], 0) # Demoted to Bronze
        self.assertEqual(sim.user_demotion_lives[1], 0) # Reset

    def test_ratio_logic(self):
        # Config: Bronze, Silver (Ratio Top 1)
        tiers = [
            TierConfig("Bronze", TierType.MMR, 0, 1000),
            TierConfig("Silver", TierType.RATIO, 1000, 9999, capacity=1)
        ]
        sim = FastSimulation(3, self.segment_configs, self.elo_config, self.match_config, tiers, initial_mmr=1000)
        
        # Set MMRs
        sim.mmr[0] = 1200
        sim.mmr[1] = 1100
        sim.mmr[2] = 1000
        
        sim._initialize_tiers()
        sim._update_daily_tiers()
        
        # Capacity = 1.
        # Top MMR is User 0 (1200).
        # User 0 -> Silver (1).
        # User 1, 2 -> Bronze (0).
        
        self.assertEqual(sim.user_tier_index[0], 1)
        self.assertEqual(sim.user_tier_index[1], 0)
        self.assertEqual(sim.user_tier_index[2], 0)

if __name__ == '__main__':
    unittest.main()
