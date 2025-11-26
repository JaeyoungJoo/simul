from simulation_core import ELOSystem, ELOConfig, Matchmaker, MatchConfig, User
import numpy as np
import random

def verify_stagnation():
    print("--- MMR Stagnation Verification (Smurf Queue) ---")
    
    # Config
    elo_config = ELOConfig(base_k=32, uncertainty_factor=0.9)
    match_config = MatchConfig(draw_prob=0.1)
    
    elo = ELOSystem(elo_config)
    matchmaker = Matchmaker(elo, match_config)
    
    # Population: 1000 Users
    # 500 High Skill (2000), 500 Low Skill (1000)
    # All start at 1000 MMR
    users = []
    for i in range(500):
        users.append(User(id=i, true_skill=2000.0, current_mmr=1000.0, segment_name="High",
                          daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                          active_hour_start=0, active_hour_end=24))
    for i in range(500, 1000):
        users.append(User(id=i, true_skill=1000.0, current_mmr=1000.0, segment_name="Low",
                          daily_play_prob=1.0, matches_per_day_mean=1, matches_per_day_std=0,
                          active_hour_start=0, active_hour_end=24))
                          
    print(f"Initial State: High Skill (2000) users at 1000 MMR")
    
    # Simulate 50 Days (approx 50 matches per user)
    for day in range(50):
        # Strict Matchmaking: Sort by MMR and pair neighbors
        # This simulates the current logic in simulation_core.py
        random.shuffle(users) # Shuffle first to randomize order of same-MMR users
        users.sort(key=lambda u: u.current_mmr)
        
        for i in range(0, len(users)-1, 2):
            matchmaker.simulate_match(users[i], users[i+1], day, 12)
            
    # Analyze
    high_mmrs = [u.current_mmr for u in users if u.true_skill == 2000.0]
    low_mmrs = [u.current_mmr for u in users if u.true_skill == 1000.0]
    
    avg_high = np.mean(high_mmrs)
    avg_low = np.mean(low_mmrs)
    
    print(f"\nAfter 50 Matches:")
    print(f"High Skill (2000) Avg MMR: {avg_high:.2f}")
    print(f"Low Skill (1000) Avg MMR: {avg_low:.2f}")
    
    if avg_high < 1500:
        print("\n[ISSUE DETECTED] High skill users are stuck!")
        print("Reason: They are likely playing against each other too often.")
    else:
        print("\n[SUCCESS] Separation occurred.")

if __name__ == "__main__":
    verify_stagnation()
