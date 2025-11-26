from simulation_core import Simulation, FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import time

def run_verification():
    print("Starting verification (Advanced ELO)...")
    
    # 1. Configs
    segments = [
        SegmentConfig("Test_Comp", 1.0, 1.0, 5.0, 0.0, 1500.0, 0.0, 10, 12)
    ]
    elo_config = ELOConfig(
        base_k=32,
        placement_matches=5,
        placement_bonus=2.0,
        streak_bonus=0.1,
        goal_diff_weight=0.05
    )
    match_config = MatchConfig(
        draw_prob=0.1,
        prob_extra_time=0.5,
        prob_pk=0.5,
        max_goal_diff=5
    )
    
    # 2. Initialize Fast Simulation
    try:
        sim = FastSimulation(num_users=1000, segment_configs=segments, elo_config=elo_config, match_config=match_config)
        sim.initialize_users()
        print(f"Initialized {sim.num_users} users.")
    except Exception as e:
        print(f"FAILED to initialize simulation: {e}")
        return

    # 3. Run
    try:
        sim.run_day()
        stats = sim.get_stats()
        print(f"Day 1: Avg MMR {stats['avg_mmr']:.2f}")
    except Exception as e:
        print(f"FAILED during simulation loop: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check Logs for Details
    has_logs = False
    for idx, name in sim.watched_indices.items():
        logs = sim.match_logs.get(idx, [])
        if logs:
            has_logs = True
            log = logs[0]
            print(f"Sample Log: {log.result} ({log.result_type}) Diff:{log.goal_diff} Change:{log.mmr_change:.1f}")
            
    if not has_logs:
        print("WARNING: No logs found.")
    else:
        print("Log verification PASSED.")
            
    print("Verification PASSED.")

if __name__ == "__main__":
    run_verification()
