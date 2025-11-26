from simulation_core import ELOSystem, ELOConfig, Matchmaker, MatchConfig, User
import numpy as np

def verify_upset_rate():
    print("--- Upset Rate Verification ---")
    
    elo_config = ELOConfig()
    match_config = MatchConfig(draw_prob=0.1) # Explicitly set draw prob
    
    elo = ELOSystem(elo_config)
    matchmaker = Matchmaker(elo, match_config)
    
    # User A: 2000 Skill
    user_a = User(id=1, true_skill=2000.0, current_mmr=2000.0, segment_name="Pro",
                  daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                  active_hour_start=0, active_hour_end=24)
                  
    # User B: 1500 Skill (Gap 500)
    user_b = User(id=2, true_skill=1500.0, current_mmr=1500.0, segment_name="Noob",
                  daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                  active_hour_start=0, active_hour_end=24)
                  
    expected_score = elo.expected_score(user_a.true_skill, user_b.true_skill)
    print(f"Skill Gap: {user_a.true_skill - user_b.true_skill}")
    print(f"Expected Score (P(A wins) + 0.5*P(Draw)): {expected_score:.4f}")
    
    # Theoretical Calculation
    # P(Draw) = 0.1
    # Remaining = 0.9
    # P(A Win Reg) = 0.9 * expected_score
    # P(B Win Reg) = 0.9 * (1 - expected_score)
    
    prob_a_win_reg = 0.9 * expected_score
    prob_b_win_reg = 0.9 * (1 - expected_score)
    
    print(f"Theoretical Regular Win Rate A: {prob_a_win_reg:.4%}")
    print(f"Theoretical Regular Win Rate B (Upset): {prob_b_win_reg:.4%}")
    
    # Simulate
    n_matches = 50000
    upsets = 0
    total_losses = 0
    
    for _ in range(n_matches):
        user_a.match_history = []
        user_b.match_history = []
        
        matchmaker.simulate_match(user_a, user_b, day=1, hour=12)
        
        res = user_a.match_history[-1]
        if res.result == "Loss":
            total_losses += 1
            if res.result_type == "Regular":
                upsets += 1
                
    print(f"\nSimulated Results ({n_matches} matches):")
    print(f"Total Losses: {total_losses} ({total_losses/n_matches:.2%})")
    print(f"Regular Time Losses (Upsets): {upsets} ({upsets/n_matches:.2%})")
    
if __name__ == "__main__":
    verify_upset_rate()
