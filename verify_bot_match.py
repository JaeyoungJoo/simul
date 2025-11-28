import numpy as np
from simulation_core import FastSimulation, TierConfig, TierType, SegmentConfig, ELOConfig, MatchConfig

def test_bot_match_logic():
    print("Testing Bot Match Logic...")
    
    # 1. Setup Configuration
    tier_configs = [
        TierConfig(
            name="TestTier",
            type=TierType.MMR,
            min_mmr=0,
            max_mmr=2000,
            bot_match_enabled=True,
            bot_trigger_goal_diff=3, # Trigger if lost by 3+ goals
            bot_trigger_loss_streak=2 # Trigger if lost 2 games in a row
        )
    ]
    
    segment_configs = [
        SegmentConfig("TestSeg", 1.0, 1.0, 1, 1, 1000, 1000)
    ]
    
    elo_config = ELOConfig(base_k=32)
    match_config = MatchConfig(bot_win_rate=1.0) # User ALWAYS wins against bot
    
    # 2. Initialize Simulation
    sim = FastSimulation(
        num_users=10, # Small number
        segment_configs=segment_configs,
        elo_config=elo_config,
        match_config=match_config,
        tier_configs=tier_configs,
        initial_mmr=1000
    )
    sim.initialize_users()
    sim.user_tier_index[:] = 0 # Assign all to TestTier (Index 0)
    sim.watched_indices = set(range(10)) # Watch all users
    sim.match_logs = {i: [] for i in range(10)} # Ensure dict exists
    
    # 3. Test Trigger: Goal Difference
    print("\n[Test 1] Goal Difference Trigger")
    # Force a match where User 0 loses to User 1 by 5 goals
    # We can't easily force specific match outcomes in run_day without mocking.
    # So we will manually call _process_matches.
    
    idx_a = np.array([0])
    idx_b = np.array([1])
    
    # Manually set MMR to ensure valid calculation
    sim.mmr[0] = 1000
    sim.mmr[1] = 1000
    
    # Simulate match: A loses, B wins. Goal diff 5.
    # _process_matches(idx_a, idx_b) calls internal logic.
    # We need to inject the result.
    # FastSimulation._process_matches calculates result based on random.
    # To test triggers, we can call _check_bot_triggers directly if we want, 
    # OR we can hack the random state, but that's hard.
    # Let's verify _check_bot_triggers logic directly.
    
    # Manually trigger for User 0
    print("Manually checking trigger for User 0 (Goal Diff 5)...")
    sim._check_bot_triggers(np.array([0]), np.array([5]))
    
    if sim.pending_bot_match[0]:
        print("PASS: User 0 has pending bot match.")
    else:
        print("FAIL: User 0 should have pending bot match.")
        
    # 4. Test Bot Match Execution
    print("\n[Test 2] Bot Match Execution")
    # Run a "day". User 0 should play bot. User 1 should play regular (or wait).
    # We need more users for regular match.
    
    # Reset pending for test
    sim.pending_bot_match[:] = False
    sim.pending_bot_match[0] = True
    
    # Run one day
    sim.run_day()
    
    # Check logs for User 0
    logs = sim.match_logs[0]
    if len(logs) > 0:
        last_log = logs[-1]
        print(f"User 0 Last Match Opponent: {last_log.opponent_id}")
        if last_log.opponent_id == -999:
            print("PASS: User 0 played against Bot.")
        else:
            print(f"FAIL: User 0 played against {last_log.opponent_id}, expected -999.")
            
        if last_log.result == "Win":
             print("PASS: User 0 won against Bot (as expected with 1.0 win rate).")
        else:
             print("FAIL: User 0 lost against Bot.")
             
        if not sim.pending_bot_match[0]:
             print("PASS: Pending flag cleared after win.")
        else:
             print("FAIL: Pending flag NOT cleared after win.")
    else:
        print("FAIL: No match logs for User 0.")

    # 5. Test Trigger: Loss Streak
    print("\n[Test 3] Loss Streak Trigger")
    # Set User 2 streak to -2
    sim.streak[2] = -2
    # Trigger check with small goal diff (should trigger due to streak)
    sim._check_bot_triggers(np.array([2]), np.array([1]))
    
    if sim.pending_bot_match[2]:
        print("PASS: User 2 has pending bot match (Streak -2).")
    else:
        print("FAIL: User 2 should have pending bot match.")

    # 6. Test Retry Logic (Loss to Bot)
    print("\n[Test 4] Bot Match Retry Logic")
    sim.match_config.bot_win_rate = 0.0 # User ALWAYS loses to bot
    sim.pending_bot_match[3] = True
    
    sim.run_day()
    
    logs = sim.match_logs[3]
    if len(logs) > 0:
        last_log = logs[-1]
        if last_log.opponent_id == -999 and last_log.result == "Loss":
            print("PASS: User 3 lost to Bot.")
            if sim.pending_bot_match[3]:
                print("PASS: Pending flag retained (Retry).")
            else:
                print("FAIL: Pending flag cleared despite loss.")
        else:
            print("FAIL: Match result unexpected.")

if __name__ == "__main__":
    test_bot_match_logic()
