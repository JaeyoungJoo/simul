from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import numpy as np

def run_verification():
    print("Starting verification (Log Fields)...")
    
    # 1. Configs
    segments = [
        SegmentConfig("Test", 1.0, 1.0, 1.0, 0.0, 1500.0, 0.0, 0, 24)
    ]
    elo_config = ELOConfig()
    match_config = MatchConfig()
    
    # 2. Initialize
    sim = FastSimulation(num_users=100, segment_configs=segments, elo_config=elo_config, match_config=match_config)
    sim.initialize_users()
    
    # 3. Run Day
    sim.run_day()
    
    # 4. Check Logs
    has_logs = False
    for idx, logs in sim.match_logs.items():
        if logs:
            has_logs = True
            log = logs[0]
            print(f"Log Found: Opponent MMR={log.opponent_mmr}, Opponent Skill={log.opponent_true_skill}")
            if hasattr(log, 'opponent_true_skill') and log.opponent_true_skill > 0:
                print("SUCCESS: opponent_true_skill field exists and is populated.")
            else:
                print("FAILURE: opponent_true_skill missing or invalid.")
            break
            
    if not has_logs:
        print("WARNING: No logs found (try increasing user count or match prob).")

if __name__ == "__main__":
    run_verification()
