from simulation_core import ELOSystem, ELOConfig, Matchmaker, MatchConfig, User
import numpy as np

def verify_win_rate():
    print("--- Win Rate Verification (Large Skill Gap) ---")
    
    elo_config = ELOConfig()
    match_config = MatchConfig() # Default: Draw 0.1, ET 0.2, PK 0.5
    
    elo = ELOSystem(elo_config)
    matchmaker = Matchmaker(elo, match_config)
    
    # User A: 1400 Skill (Gap 400)
    user_a = User(id=1, true_skill=1400.0, current_mmr=1400.0, segment_name="Pro",
                  daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                  active_hour_start=0, active_hour_end=24)
                  
    # User B: 1000 Skill (Low)
    user_b = User(id=2, true_skill=1000.0, current_mmr=1000.0, segment_name="Noob",
                  daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                  active_hour_start=0, active_hour_end=24)
                  
    # Expected Win Rate (Pure ELO)
    expected_score = elo.expected_score(user_a.true_skill, user_b.true_skill)
    print(f"Skill Gap: {user_a.true_skill - user_b.true_skill}")
    print(f"Theoretical Win Prob (Pure ELO): {expected_score:.4f}")
    
    # Simulate 10,000 matches
    wins = 0
    losses = 0
    draws = 0
    
    results = {"Regular": 0, "Extra": 0, "PK": 0}
    
    for _ in range(10000):
        # Reset logs for clean state (optional, but good practice)
        user_a.match_history = []
        user_b.match_history = []
        
        matchmaker.simulate_match(user_a, user_b, day=1, hour=12)
        
        last_match = user_a.match_history[-1]
        if last_match.result == "Win":
            wins += 1
            results[last_match.result_type] += 1
        elif last_match.result == "Loss":
            losses += 1
        else:
            draws += 1
            
    total = wins + losses + draws
    print(f"\nSimulated Results ({total} matches):")
    print(f"Wins: {wins} ({wins/total:.2%})")
    print(f"Losses: {losses} ({losses/total:.2%})")
    print(f"Draws: {draws} ({draws/total:.2%})")
    
    print("\nWin Types:")
    for k, v in results.items():
        if wins > 0:
            print(f"  {k}: {v} ({v/wins:.2%} of wins)")
            
    # Check if PKs are causing losses
    # In this scenario, User A should almost never lose in Regular/Extra time.
    # Losses should come primarily from PKs (50/50 chance) or extreme bad luck.
    
if __name__ == "__main__":
    verify_win_rate()
