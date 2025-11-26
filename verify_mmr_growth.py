from simulation_core import ELOSystem, ELOConfig, Matchmaker, MatchConfig, User
import numpy as np
import matplotlib.pyplot as plt

def verify_mmr_growth():
    print("--- MMR Growth Verification ---")
    
    elo_config = ELOConfig(base_k=32, uncertainty_factor=0.9)
    match_config = MatchConfig(draw_prob=0.1) # Fixed 10% draw probability
    
    elo = ELOSystem(elo_config)
    matchmaker = Matchmaker(elo, match_config)
    
    # Hero: True Skill 2000, Starts at 1000
    hero = User(id=1, true_skill=2000.0, current_mmr=1000.0, segment_name="Hero",
                daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                active_hour_start=0, active_hour_end=24)
                
    # Opponent Pool: Average Skill 1500 (Fixed for simplicity to see if Hero can beat them)
    # In reality, opponents vary, but let's test if Hero can climb above 1500 to 2000.
    opponent_skill = 1500.0
    
    mmr_history = [hero.current_mmr]
    
    n_matches = 1000
    
    total_score_actual = 0.0
    total_score_expected = 0.0
    
    for i in range(n_matches):
        # Create a fresh opponent each time (or reset one)
        opponent = User(id=2, true_skill=opponent_skill, current_mmr=opponent_skill, segment_name="Avg",
                        daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                        active_hour_start=0, active_hour_end=24)
        
        # Calculate Expected Score BEFORE match
        exp_score = elo.expected_score(hero.current_mmr, opponent.current_mmr)
        total_score_expected += exp_score
        
        # Simulate
        matchmaker.simulate_match(hero, opponent, day=1, hour=12)
        
        # Get Actual Score
        last_match = hero.match_history[-1]
        actual_score = 0.0
        if last_match.result == "Win":
            actual_score = 1.0
        elif last_match.result == "Draw":
            actual_score = 0.5
        
        total_score_actual += actual_score
        mmr_history.append(hero.current_mmr)
        
    print(f"Final MMR after {n_matches} matches: {hero.current_mmr:.2f}")
    print(f"Target True Skill: {hero.true_skill}")
    print(f"Opponent Skill: {opponent_skill}")
    
    print(f"\nTotal Expected Score (Sum of P_win): {total_score_expected:.2f}")
    print(f"Total Actual Score (Wins + 0.5*Draws): {total_score_actual:.2f}")
    print(f"Difference: {total_score_actual - total_score_expected:.2f}")
    
    if hero.current_mmr < 1900:
        print("\n[ISSUE DETECTED] MMR failed to converge to True Skill (2000).")
        print("Reason Analysis:")
        print("If Actual Score < Expected Score, the user is 'underperforming' relative to ELO math.")
        print("This often happens if 'Draw Probability' is fixed but ELO expects near 100% win rate.")
        
if __name__ == "__main__":
    verify_mmr_growth()
