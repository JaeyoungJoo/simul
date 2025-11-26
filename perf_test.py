from simulation_core import FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import time
import numpy as np

def run_perf_test(num_users):
    print(f"\n--- Testing with {num_users:,} users ---")
    
    segments = [
        SegmentConfig("Comp", 0.2, 0.9, 10.0, 2.0, 1500.0, 200.0, 18, 24),
        SegmentConfig("Casual", 0.8, 0.5, 3.0, 1.0, 1000.0, 400.0, 12, 20)
    ]
    elo_config = ELOConfig()
    match_config = MatchConfig()
    
    start_init = time.time()
    sim = FastSimulation(num_users, segments, elo_config, match_config)
    sim.initialize_users()
    init_time = time.time() - start_init
    print(f"Initialization: {init_time:.4f}s")
    
    start_run = time.time()
    sim.run_day()
    run_time = time.time() - start_run
    print(f"Run Day 1: {run_time:.4f}s")
    
    return run_time

if __name__ == "__main__":
    # Warmup
    run_perf_test(10000)
    
    # 100k
    t_100k = run_perf_test(100000)
    
    # 1M
    # Only run if 100k was fast enough (< 2s) to avoid hanging if it's too slow
    if t_100k < 2.0:
        run_perf_test(1000000)
    else:
        print("Skipping 1M test as 100k took > 2s")
