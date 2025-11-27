import random
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

CORE_VERSION = "1.3 (Rank System Overhaul)"
print(f"Simulation Core Loaded: Version {CORE_VERSION}")

class TierType(Enum):
    MMR = "MMR"
    LADDER = "Ladder"
    RATIO = "Ratio"

@dataclass
class TierConfig:
    name: str
    type: TierType
    # MMR specific
    min_mmr: int = 0
    max_mmr: int = 9999
    demotion_mmr: int = 0 # Not used directly, logic uses min_mmr
    demotion_lives: int = 0 # 0 = No demotion
    
    # Ladder specific
    points_win: int = 0
    points_draw: int = 0
    promotion_points: int = 100
    
    # Ratio specific
    capacity: int = 0 # Absolute number of users (e.g., 100)

    # Placement specific
    placement_min_mmr: int = 0
    placement_max_mmr: int = 0


@dataclass
class MatchLog:
    day: int
    hour: int
    opponent_id: int
    opponent_mmr: float
    opponent_true_skill: float
    result: str # 'Win', 'Loss', 'Draw'
    result_type: str # 'Regular', 'Extra', 'PK'
    goal_diff: int
    mmr_change: float
    current_mmr: float
    current_tier_index: int = 0
    current_ladder_points: int = 0

@dataclass
class SegmentConfig:
    name: str
    ratio: float # 0.0 to 1.0
    daily_play_prob: float # Probability to play on a given day
    matches_per_day_min: float
    matches_per_day_max: float
    true_skill_min: float
    true_skill_max: float
    active_hour_start: int = 0
    active_hour_end: int = 23

@dataclass
class ELOConfig:
    base_k: int = 32
    placement_matches: int = 10
    placement_bonus: float = 4.0
    streak_rules: List[Dict] = field(default_factory=list) # [{'min_streak': 3, 'bonus': 5.0}, ...]
    goal_diff_rules: List[Dict] = field(default_factory=list) # [{'min_diff': 2, 'bonus': 2.0}, ...]
    win_type_decay: Dict[str, float] = field(default_factory=lambda: {'Regular': 1.0, 'Extra': 0.8, 'PK': 0.6})
    uncertainty_factor: float = 0.9 # Correction for randomness (draws/upsets). 1.0 = No correction.
    calibration_k_bonus: float = 1.0 # Multiplier during calibration
    calibration_enabled: bool = False
    calibration_match_count: int = 10

@dataclass
class MatchConfig:
    draw_prob: float = 0.1 # Probability of draw in regular time
    prob_extra_time: float = 0.2 # If draw, prob to go to Extra Time
    prob_pk: float = 0.5 # If draw in Extra Time, prob to go to PK
    max_goal_diff: int = 5
    matchmaking_jitter: float = 50.0 # Standard deviation of noise added to MMR for sorting

@dataclass
class User:
    id: int
    true_skill: float
    current_mmr: float
    segment_name: str
    
    # Play patterns
    daily_play_prob: float
    matches_per_day_min: float
    matches_per_day_max: float
    active_hour_start: int
    active_hour_end: int
    
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    current_streak: int = 0 # + for wins, - for losses
    
    match_history: List[MatchLog] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

class ELOSystem:
    def __init__(self, config: ELOConfig):
        self.config = config

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def calculate_k_factor(self, matches_played: int, streak: int, is_calibration: bool = False) -> float:
        k = self.config.base_k
        
        # Placement Bonus
        if matches_played < self.config.placement_matches:
            k *= self.config.placement_bonus
            
        # Calibration Bonus (Moved to calculate_new_ratings for directional logic)
            
        # Streak Bonus (Tiered)
        streak_abs = abs(streak)
        bonus = 0.0
        # Apply the highest applicable bonus
        for rule in sorted(self.config.streak_rules, key=lambda x: x['min_streak']):
            if streak_abs >= rule['min_streak']:
                bonus = rule['bonus']
        
        k += bonus
             
        return k

    def calculate_new_ratings(self, user_a: User, user_b: User, actual_score_a: float, 
                              goal_diff: int, win_type: str, 
                              is_calibration_a: bool = False, is_calibration_b: bool = False) -> Tuple[float, float]:
        
        # Asymmetric Pricing: If calibrating, use Opponent's True Skill for expectation
        rating_b_for_a = user_b.true_skill if is_calibration_a else user_b.current_mmr
        rating_a_for_b = user_a.true_skill if is_calibration_b else user_a.current_mmr
        
        expected_a = self.expected_score(user_a.current_mmr, rating_b_for_a)
        expected_b = self.expected_score(user_b.current_mmr, rating_a_for_b)
        
        # Apply Uncertainty Correction
        if self.config.uncertainty_factor != 1.0:
            expected_a = 0.5 + (expected_a - 0.5) * self.config.uncertainty_factor
            expected_b = 0.5 + (expected_b - 0.5) * self.config.uncertainty_factor
        
        # Calculate K-Factors
        k_a = self.calculate_k_factor(user_a.matches_played, user_a.current_streak, is_calibration_a)
        k_b = self.calculate_k_factor(user_b.matches_played, user_b.current_streak, is_calibration_b)
        
        # Performance Multiplier
        # Goal Diff Bonus (Tiered)
        # Additive to K-Factor? Or Multiplier?
        # User request: "Add K-factor bonus" -> Additive
        
        goal_diff_bonus = 0.0
        diff_abs = abs(goal_diff)
        for rule in sorted(self.config.goal_diff_rules, key=lambda x: x['min_diff']):
            if diff_abs >= rule['min_diff']:
                goal_diff_bonus = rule['bonus']
        
        k_a += goal_diff_bonus
        k_b += goal_diff_bonus
        
        # --- Smart Calibration (Directional Bonus) ---
        # Only apply bonus if the result moves MMR towards True Skill
        
        # User A
        if is_calibration_a:
            delta_a = actual_score_a - expected_a
            # Direction of change: sign(delta_a)
            # Direction to target: sign(user_a.true_skill - user_a.current_mmr)
            
            # We use a small epsilon for float comparison or just simple logic
            # If delta_a > 0 (Win/Good Draw) and TrueSkill > MMR -> Good
            # If delta_a < 0 (Loss/Bad Draw) and TrueSkill < MMR -> Good
            
            direction_match_a = (delta_a > 0 and user_a.true_skill > user_a.current_mmr) or \
                                (delta_a < 0 and user_a.true_skill < user_a.current_mmr)
            
            if direction_match_a:
                k_a *= self.config.calibration_k_bonus

        # User B
        if is_calibration_b:
            actual_score_b = 1 - actual_score_a
            delta_b = actual_score_b - expected_b
            direction_match_b = (delta_b > 0 and user_b.true_skill > user_b.current_mmr) or \
                                (delta_b < 0 and user_b.true_skill < user_b.current_mmr)
            
        self.elo_system = elo_system
        self.match_config = match_config

    def simulate_match(self, user_a: User, user_b: User, day: int, hour: int):
        # 1. Determine Winner based on True Skill
        prob_a_win = self.elo_system.expected_score(user_a.true_skill, user_b.true_skill)
        rand = random.random()
        
        outcome_a = "" # Win/Loss/Draw
        result_type = "Regular"
        actual_score_a = 0.5
        
        # Determine Base Outcome (Regular Time)
        if rand < self.match_config.draw_prob:
            # Draw in Regular Time
            # Check Extra Time
            if random.random() < self.match_config.prob_extra_time:
                # Goes to Extra Time
                # Recalculate win prob for ET (simplified, use same prob)
                rand_et = random.random()
                if rand_et < 0.5: # Simplified 50/50 in ET for now or reuse prob
                     # Actually let's use original prob re-normalized
                     if random.random() < prob_a_win:
                         outcome_a = "Win"
                         result_type = "Extra"
                         actual_score_a = 1.0
                     else:
                         outcome_a = "Loss"
                         result_type = "Extra"
                         actual_score_a = 0.0
                else:
                    # Draw in ET, check PK
                    if random.random() < self.match_config.prob_pk:
                        # Goes to PK
                        if random.random() < 0.5: # PK is mostly luck
                            outcome_a = "Win"
                            result_type = "PK"
                            actual_score_a = 1.0
                        else:
                            outcome_a = "Loss"
                            result_type = "PK"
                            actual_score_a = 0.0
                    else:
                         outcome_a = "Draw"
                         result_type = "Regular" # Or 'Extra' but draw
                         actual_score_a = 0.5
            else:
                outcome_a = "Draw"
                result_type = "Regular"
                actual_score_a = 0.5
        else:
            # Decisive in Regular Time
            remaining = 1.0 - self.match_config.draw_prob
            if rand < self.match_config.draw_prob + (prob_a_win * remaining):
                outcome_a = "Win"
                result_type = "Regular"
                actual_score_a = 1.0
            else:
                outcome_a = "Loss"
                result_type = "Regular"
                actual_score_a = 0.0

        # 2. Determine Goal Difference
        # Correlate with skill diff
        skill_diff = (user_a.true_skill - user_b.true_skill) / 100.0
        base_diff = abs(int(np.random.normal(skill_diff, 1.0)))
        goal_diff = min(max(1, base_diff), self.match_config.max_goal_diff)
        
        if outcome_a == "Draw":
            goal_diff = 0
            
        # 3. Update Ratings
        is_cal_a = self.match_config.calibration_enabled and user_a.matches_played < self.match_config.calibration_match_count
        is_cal_b = self.match_config.calibration_enabled and user_b.matches_played < self.match_config.calibration_match_count
        
        new_mmr_a, new_mmr_b = self.elo_system.calculate_new_ratings(
            user_a, user_b, actual_score_a, goal_diff, result_type, is_cal_a, is_cal_b
        )
        
        change_a = new_mmr_a - user_a.current_mmr
        change_b = new_mmr_b - user_b.current_mmr
        
        user_a.current_mmr = new_mmr_a
        user_b.current_mmr = new_mmr_b
        
        # 4. Update Stats & Streak
        user_a.matches_played += 1
        user_b.matches_played += 1
        
        outcome_b = "Draw" if outcome_a == "Draw" else ("Loss" if outcome_a == "Win" else "Win")
        
        if outcome_a == "Win":
            user_a.wins += 1
            user_b.losses += 1
            user_a.current_streak = user_a.current_streak + 1 if user_a.current_streak > 0 else 1
            user_b.current_streak = user_b.current_streak - 1 if user_b.current_streak < 0 else -1
        elif outcome_a == "Loss":
            user_a.losses += 1
            user_b.wins += 1
            user_a.current_streak = user_a.current_streak - 1 if user_a.current_streak < 0 else -1
            user_b.current_streak = user_b.current_streak + 1 if user_b.current_streak > 0 else 1
        else:
            user_a.draws += 1
            user_b.draws += 1
            user_a.current_streak = 0
            user_b.current_streak = 0
            
        # 5. Log
        user_a.match_history.append(MatchLog(
            day=day, hour=hour, opponent_id=user_b.id, opponent_mmr=user_b.current_mmr,
            opponent_true_skill=user_b.true_skill,
            result=outcome_a, result_type=result_type, goal_diff=goal_diff,
            mmr_change=change_a, current_mmr=user_a.current_mmr
        ))
        user_b.match_history.append(MatchLog(
            day=day, hour=hour, opponent_id=user_a.id, opponent_mmr=user_a.current_mmr,
            opponent_true_skill=user_a.true_skill,
            result=outcome_b, result_type=result_type, goal_diff=goal_diff,
            mmr_change=change_b, current_mmr=user_b.current_mmr
        ))

class Simulation:
    def __init__(self, num_users: int, segment_configs: List[SegmentConfig], 
                 elo_config: ELOConfig, match_config: MatchConfig, initial_mmr: float = 1000.0):
        self.num_users = num_users
        self.elo_config = elo_config
        self.match_config = match_config
        self.initial_mmr = initial_mmr
        self.elo_system = ELOSystem(elo_config)
        self.matchmaker = Matchmaker(self.elo_system, match_config)
        self.segment_configs = segment_configs
        self.users: List[User] = []
        self.day = 0
        
    def initialize_users(self):
        current_id = 0
        total_ratio = sum(s.ratio for s in self.segment_configs)
        
        for config in self.segment_configs:
            count = int(self.num_users * (config.ratio / total_ratio))
            true_skills = np.random.uniform(config.true_skill_min, config.true_skill_max, count)
            
            for i in range(count):
                self.users.append(User(
                    id=current_id,
                    true_skill=true_skills[i],
                    current_mmr=self.initial_mmr,
                    segment_name=config.name,
                    daily_play_prob=config.daily_play_prob,
                    matches_per_day_min=config.matches_per_day_min,
                    matches_per_day_max=config.matches_per_day_max,
                    active_hour_start=config.active_hour_start,
                    active_hour_end=config.active_hour_end
                ))
                current_id += 1
                
        while len(self.users) < self.num_users:
            config = self.segment_configs[0]
            self.users.append(User(
                id=current_id,
                true_skill=random.uniform(config.true_skill_min, config.true_skill_max),
                current_mmr=self.initial_mmr,
                segment_name=config.name,
                daily_play_prob=config.daily_play_prob,
                matches_per_day_min=config.matches_per_day_min,
                matches_per_day_max=config.matches_per_day_max,
                active_hour_start=config.active_hour_start,
                active_hour_end=config.active_hour_end
            ))
            current_id += 1
                
        for hour in range(24):
            queue = hourly_queue[hour]
            if len(queue) < 2:
                continue
            if len(queue) < 2:
                continue
            random.shuffle(queue)
            # Jittered Sort: Sort by (MMR + Noise)
            jitter = self.match_config.matchmaking_jitter
            queue.sort(key=lambda u: u.current_mmr + random.gauss(0, jitter))
            
            
            for i in range(0, len(queue) - 1, 2):
                self.matchmaker.simulate_match(queue[i], queue[i+1], self.day, hour)

    def get_stats(self):
        mmrs = [u.current_mmr for u in self.users]
        return {
            "day": self.day,
            "avg_mmr": np.mean(mmrs),
            "min_mmr": np.min(mmrs),
            "max_mmr": np.max(mmrs),
            "std_mmr": np.std(mmrs)
        }

    def apply_soft_reset(self, compression_factor: float, target_mean: float):
        """
        Applies soft reset to all users:
        NewMMR = Target + (OldMMR - Target) * Factor
        Resets seasonal stats (matches, w/l/d, streak) to trigger placement matches again.
        """
        for user in self.users:
            user.current_mmr = target_mean + (user.current_mmr - target_mean) * compression_factor
            
            # Reset Seasonal Stats
            user.matches_played = 0
            user.wins = 0
            user.losses = 0
            user.draws = 0
            user.current_streak = 0
            # Note: We keep match_history for now, or we could clear it. 
            # If we want to track full history, we keep it.

    def apply_tiered_reset(self, rules: List[Dict]):
        """
        Applies soft reset based on tiered rules.
        rules: List of dicts {'min': float, 'max': float, 'target': float, 'compression': float}
        """
        for user in self.users:
            mmr = user.current_mmr
            
            # Find matching rule
            applied_rule = None
            for rule in rules:
                if rule['min'] <= mmr < rule['max']:
                    applied_rule = rule
                    break
            
            # If no rule matches, do nothing or apply default? 
            # Let's assume rules cover all ranges, or we skip.
            if applied_rule:
                target = applied_rule['target']
                comp = applied_rule['compression']
                user.current_mmr = target + (mmr - target) * comp
                
            # Reset Seasonal Stats
            user.matches_played = 0
            user.wins = 0
            user.losses = 0
            user.draws = 0
            user.current_streak = 0

class FastSimulation:
    def __init__(self, num_users: int, segment_configs: List[SegmentConfig],
                 elo_config: ELOConfig, match_config: MatchConfig, 
                 tier_configs: List[TierConfig] = None,
                 initial_mmr: float = 1000.0):
        self.num_users = num_users
        self.elo_config = elo_config
        self.match_config = match_config
        self.initial_mmr = initial_mmr
        self.segment_configs = segment_configs
        self.tier_configs = tier_configs if tier_configs else []
        self.day = 0
        
        self.ids = np.arange(num_users)
        self.mmr = np.full(num_users, self.initial_mmr)
        self.true_skill = np.zeros(num_users)
        self.wins = np.zeros(num_users, dtype=int)
        self.losses = np.zeros(num_users, dtype=int)
        self.draws = np.zeros(num_users, dtype=int)
        self.matches_played = np.zeros(num_users, dtype=int)
        self.streak = np.zeros(num_users, dtype=int)
        
        # Tier Tracking
        # 0 = Lowest Tier
        # -1 = Unranked (Placement Matches)
        if self.elo_config.placement_matches > 0:
            self.user_tier_index = np.full(num_users, -1, dtype=int)
        else:
            self.user_tier_index = np.zeros(num_users, dtype=int)
            
        self.user_ladder_points = np.zeros(num_users, dtype=int)
        self.user_demotion_lives = np.zeros(num_users, dtype=int) # Current consecutive losses or lives used
        
        # Stats Tracking
        self.promotion_counts = {} # {tier_idx: count}
        self.demotion_counts = {} # {tier_idx: count}
        
        # Initialize Tiers based on Initial MMR (Only if no placement matches)
        if self.tier_configs and self.elo_config.placement_matches == 0:
            self._initialize_tiers()

        self.segment_indices = np.zeros(num_users, dtype=int)
        self.seg_daily_prob = []
        self.seg_matches_min = []
        self.seg_matches_max = []
        self.seg_names = []
        
        self.watched_indices = {}
        self.match_logs = {}

    def initialize_users(self):
        # Reset segment tracking lists
        self.seg_daily_prob = []
        self.seg_matches_min = []
        self.seg_matches_max = []
        self.seg_names = []

        current_idx = 0
        total_ratio = sum(s.ratio if not isinstance(s, dict) else s['ratio'] for s in self.segment_configs)
        
        if total_ratio <= 0:
            total_ratio = 1.0
        
        for i, config in enumerate(self.segment_configs):
            if isinstance(config, dict):
                ratio = config['ratio']
                true_skill_min = config['true_skill_min']
                true_skill_max = config['true_skill_max']
                daily_play_prob = config['daily_play_prob']
                matches_per_day_min = config['matches_per_day_min']
                matches_per_day_max = config['matches_per_day_max']
                name = config['name']
            else:
                ratio = config.ratio
                true_skill_min = config.true_skill_min
                true_skill_max = config.true_skill_max
                daily_play_prob = config.daily_play_prob
                matches_per_day_min = config.matches_per_day_min
                matches_per_day_max = config.matches_per_day_max
                name = config.name

            count = int(self.num_users * (ratio / total_ratio))
            if current_idx + count > self.num_users:
                count = self.num_users - current_idx
            
            skills = np.random.uniform(true_skill_min, true_skill_max, count)
            self.true_skill[current_idx:current_idx+count] = skills
            self.segment_indices[current_idx:current_idx+count] = i
            
            self.seg_daily_prob.append(daily_play_prob)
            self.seg_matches_min.append(matches_per_day_min)
            self.seg_matches_max.append(matches_per_day_max)
            self.seg_names.append(name)
            
            if count > 0:
                segment_skills = self.true_skill[current_idx:current_idx+count]
                target_skill = (true_skill_min + true_skill_max) / 2
                closest_offset = np.abs(segment_skills - target_skill).argmin()
                sample_idx = current_idx + closest_offset
                
                self.watched_indices[sample_idx] = name
                self.match_logs[sample_idx] = []
                
            current_idx += count
            
        if current_idx < self.num_users:
            config = self.segment_configs[0]
            if isinstance(config, dict):
                 ts_mean = (config['true_skill_min'] + config['true_skill_max']) / 2
            else:
                 ts_mean = (config.true_skill_min + config.true_skill_max) / 2
                 
            self.true_skill[current_idx:] = ts_mean
            self.segment_indices[current_idx:] = 0
            
        self.seg_daily_prob = np.array(self.seg_daily_prob)
        self.seg_matches_min = np.array(self.seg_matches_min)
        self.seg_matches_max = np.array(self.seg_matches_max)

    def _initialize_tiers(self):
        for i in range(self.num_users):
            mmr = self.mmr[i]
            assigned_idx = 0
            for idx, config in enumerate(self.tier_configs):
                if config.type == TierType.MMR:
                    if config.min_mmr <= mmr < config.max_mmr:
                        assigned_idx = idx
            self.user_tier_index[i] = assigned_idx

    def run_day(self):
        self.day += 1
        
        rand_probs = np.random.rand(self.num_users)
        user_thresholds = self.seg_daily_prob[self.segment_indices]
        active_mask = rand_probs < user_thresholds
        active_indices = self.ids[active_mask]
        
        if len(active_indices) < 2:
            return

        mins = self.seg_matches_min[self.segment_indices[active_indices]]
        maxs = self.seg_matches_max[self.segment_indices[active_indices]]
        
        raw_counts = np.random.uniform(mins, maxs)
        base_counts = np.floor(raw_counts).astype(int)
        extra_probs = raw_counts - base_counts
        extra_matches = (np.random.rand(len(active_indices)) < extra_probs).astype(int)
        
        counts = base_counts + extra_matches
        counts = np.maximum(1, counts)
        
        matches_remaining = np.zeros(self.num_users, dtype=int)
        matches_remaining[active_indices] = counts
        
        while True:
            candidates_mask = matches_remaining > 0
            if np.sum(candidates_mask) < 2:
                break
                
            candidate_indices = self.ids[candidates_mask]
            candidate_mmrs = self.mmr[candidate_indices]
            
            if self.elo_config.calibration_enabled:
                candidate_matches = self.matches_played[candidate_indices]
                candidate_true_skill = self.true_skill[candidate_indices]
                cal_mask = candidate_matches < self.elo_config.calibration_match_count
                sort_values = np.where(cal_mask, candidate_true_skill, candidate_mmrs)
            else:
                sort_values = candidate_mmrs

            jitter = self.match_config.matchmaking_jitter
            noise = np.random.normal(0, jitter, len(sort_values))
            
            sorted_order = np.argsort(sort_values + noise)
            sorted_candidates = candidate_indices[sorted_order]
            
            n_candidates = len(sorted_candidates)
            if n_candidates % 2 != 0:
                sorted_candidates = sorted_candidates[:-1]
                
            idx_a = sorted_candidates[0::2]
            idx_b = sorted_candidates[1::2]
            
            self._process_matches(idx_a, idx_b)
            
            matches_remaining[idx_a] -= 1
            matches_remaining[idx_b] -= 1
            
        if self.tier_configs:
            self._update_daily_tiers()

    def _process_matches(self, idx_a, idx_b):
        ra = self.mmr[idx_a]
        rb = self.mmr[idx_b]
        ts_a = self.true_skill[idx_a]
        ts_b = self.true_skill[idx_b]
        
        prob_a_win = 1 / (1 + 10 ** ((ts_b - ts_a) / 400))
        rands = np.random.rand(len(idx_a))
        
        draw_prob = self.match_config.draw_prob
        is_draw_reg = rands < draw_prob
        
        rands_et = np.random.rand(len(idx_a))
        goes_to_et = is_draw_reg & (rands_et < self.match_config.prob_extra_time)
        goes_to_pk = goes_to_et & (np.random.rand(len(idx_a)) < self.match_config.prob_pk)
        
        rem_prob = 1.0 - draw_prob
        adj_win = prob_a_win * rem_prob
        
        win_reg = (rands >= draw_prob) & (rands < (draw_prob + adj_win))
        loss_reg = rands >= (draw_prob + adj_win)
        
        et_win_a = goes_to_et & ~goes_to_pk & (np.random.rand(len(idx_a)) < prob_a_win)
        et_loss_a = goes_to_et & ~goes_to_pk & ~et_win_a
        
        pk_win_a = goes_to_pk & (np.random.rand(len(idx_a)) < 0.5)
        pk_loss_a = goes_to_pk & ~pk_win_a
        
        final_win_a = win_reg | et_win_a | pk_win_a
        final_loss_a = loss_reg | et_loss_a | pk_loss_a
        final_draw = is_draw_reg & ~goes_to_et
        
        res_type = np.full(len(idx_a), 'Regular', dtype=object)
        res_type[goes_to_et] = 'Extra'
        res_type[goes_to_pk] = 'PK'
        
        scores_a = np.zeros(len(idx_a))
        scores_a[final_win_a] = 1.0
        scores_a[final_draw] = 0.5
        scores_a[final_loss_a] = 0.0
        
        skill_diff = (ts_a - ts_b) / 100.0
        base_diff = np.abs(np.random.normal(skill_diff, 1.0))
        goal_diff = np.clip(base_diff, 1, self.match_config.max_goal_diff)
        goal_diff[final_draw] = 0
        
        cal_mask_a = self.matches_played[idx_a] < self.elo_config.calibration_match_count
        rating_b_for_a = np.where(self.elo_config.calibration_enabled & cal_mask_a, 
                                  self.true_skill[idx_b], self.mmr[idx_b])
        expected_a = 1 / (1 + 10 ** ((rating_b_for_a - self.mmr[idx_a]) / 400))
        
        cal_mask_b = self.matches_played[idx_b] < self.elo_config.calibration_match_count
        rating_a_for_b = np.where(self.elo_config.calibration_enabled & cal_mask_b, 
                                  self.true_skill[idx_a], self.mmr[idx_a])
        expected_b = 1 / (1 + 10 ** ((rating_a_for_b - self.mmr[idx_b]) / 400))

        if self.elo_config.uncertainty_factor != 1.0:
            expected_a = 0.5 + (expected_a - 0.5) * self.elo_config.uncertainty_factor
            expected_b = 0.5 + (expected_b - 0.5) * self.elo_config.uncertainty_factor

        def get_k(matches, streak):
            k = np.full(len(matches), self.elo_config.base_k, dtype=float)
            place_mask = matches < self.elo_config.placement_matches
            k[place_mask] *= self.elo_config.placement_bonus
            
            streak_abs = np.abs(streak)
            streak_bonus = np.zeros(len(matches))
            
            sorted_rules = sorted(self.elo_config.streak_rules, key=lambda x: x['min_streak'])
            for rule in sorted_rules:
                mask = streak_abs >= rule['min_streak']
                streak_bonus[mask] = rule['bonus']
                
            k += streak_bonus
            return k
            
        k_a = get_k(self.matches_played[idx_a], self.streak[idx_a])
        k_b = get_k(self.matches_played[idx_b], self.streak[idx_b])
        
        goal_diff_bonus = np.zeros(len(idx_a))
        sorted_gd_rules = sorted(self.elo_config.goal_diff_rules, key=lambda x: x['min_diff'])
        
        for rule in sorted_gd_rules:
            mask = goal_diff >= rule['min_diff']
            goal_diff_bonus[mask] = rule['bonus']
            
        k_a += goal_diff_bonus
        k_b += goal_diff_bonus
        
        type_mult = np.ones(len(idx_a))
        type_mult[res_type == 'Extra'] = self.elo_config.win_type_decay['Extra']
        type_mult[res_type == 'PK'] = self.elo_config.win_type_decay['PK']
        
        final_k_a = k_a * type_mult
        final_k_b = k_b * type_mult
        
        if self.elo_config.calibration_enabled:
            delta_a = scores_a - expected_a
            delta_b = (1 - scores_a) - expected_b
            
            target_dir_a = np.sign(self.true_skill[idx_a] - self.mmr[idx_a])
            move_dir_a = np.sign(delta_a)
            apply_a = cal_mask_a & (target_dir_a != 0) & (move_dir_a == target_dir_a)
            final_k_a[apply_a] *= self.elo_config.calibration_k_bonus
            
            target_dir_b = np.sign(self.true_skill[idx_b] - self.mmr[idx_b])
            move_dir_b = np.sign(delta_b)
            apply_b = cal_mask_b & (target_dir_b != 0) & (move_dir_b == target_dir_b)
            final_k_b[apply_b] *= self.elo_config.calibration_k_bonus
        
        new_ra = ra + final_k_a * (scores_a - expected_a)
        new_rb = rb + final_k_b * ((1 - scores_a) - expected_b)
        
        self.mmr[idx_a] = new_ra
        self.mmr[idx_b] = new_rb
        
        self.matches_played[idx_a] += 1
        self.matches_played[idx_b] += 1
        
        self.wins[idx_a[final_win_a]] += 1
        self.losses[idx_b[final_win_a]] += 1
        
        self.losses[idx_a[final_loss_a]] += 1
        self.wins[idx_b[final_loss_a]] += 1
        
        self.draws[idx_a[final_draw]] += 1
        self.draws[idx_b[final_draw]] += 1
        
        mask_win = final_win_a
        self.streak[idx_a[mask_win]] = np.where(self.streak[idx_a[mask_win]] > 0, self.streak[idx_a[mask_win]] + 1, 1)
        self.streak[idx_b[mask_win]] = np.where(self.streak[idx_b[mask_win]] < 0, self.streak[idx_b[mask_win]] - 1, -1)
        
        mask_loss = final_loss_a
        self.streak[idx_a[mask_loss]] = np.where(self.streak[idx_a[mask_loss]] < 0, self.streak[idx_a[mask_loss]] - 1, -1)
        self.streak[idx_b[mask_loss]] = np.where(self.streak[idx_b[mask_loss]] > 0, self.streak[idx_b[mask_loss]] + 1, 1)
        
        mask_draw = final_draw
        self.streak[idx_a[mask_draw]] = 0
        self.streak[idx_b[mask_draw]] = 0
        
        for w_id in self.watched_indices:
            loc_a = np.where(idx_a == w_id)[0]
            if len(loc_a) > 0:
                i = loc_a[0]
                res = "Win" if final_win_a[i] else ("Draw" if final_draw[i] else "Loss")
                change = new_ra[i] - ra[i]
                self.match_logs[w_id].append(MatchLog(
                    day=self.day, hour=12, opponent_id=int(idx_b[i]), opponent_mmr=rb[i],
                    opponent_true_skill=ts_b[i],
                    result=res, result_type=res_type[i], goal_diff=int(goal_diff[i]),
                    mmr_change=change, current_mmr=new_ra[i],
                    current_tier_index=self.user_tier_index[idx_a[i]],
                    current_ladder_points=self.user_ladder_points[idx_a[i]]
                ))
            loc_b = np.where(idx_b == w_id)[0]
            if len(loc_b) > 0:
                i = loc_b[0]
                res = "Win" if final_loss_a[i] else ("Draw" if final_draw[i] else "Loss")
                change = new_rb[i] - rb[i]
                self.match_logs[w_id].append(MatchLog(
                    day=self.day, hour=12, opponent_id=int(idx_a[i]), opponent_mmr=ra[i],
                    opponent_true_skill=ts_a[i],
                    result=res, result_type=res_type[i], goal_diff=int(goal_diff[i]),
                    mmr_change=change, current_mmr=new_rb[i],
                    current_tier_index=self.user_tier_index[idx_b[i]],
                    current_ladder_points=self.user_ladder_points[idx_b[i]]
                ))

        if self.tier_configs:
            self._process_tier_updates(idx_a, idx_b, final_win_a, final_draw, final_loss_a)

    def _process_tier_updates(self, idx_a, idx_b, win_a, draw, loss_a):
        all_idx = np.concatenate([idx_a, idx_b])
        res_a = np.zeros(len(idx_a), dtype=int)
        res_a[win_a] = 1
        res_a[loss_a] = -1
        
        res_b = np.zeros(len(idx_b), dtype=int)
        res_b[win_a] = -1
        res_b[loss_a] = 1
        
        all_res = np.concatenate([res_a, res_b])
        
        # Placement Logic
        if self.elo_config.placement_matches > 0:
            just_finished_mask = (self.matches_played[all_idx] == self.elo_config.placement_matches)
            if just_finished_mask.any():
                finished_indices = all_idx[just_finished_mask]
                self._assign_placement_tier(finished_indices)

        current_tiers = self.user_tier_index[all_idx]
        unique_tiers = np.unique(current_tiers)
        
        for t_idx in unique_tiers:
            if t_idx == -1: continue # Skip unranked
            if t_idx >= len(self.tier_configs): continue
            
            config = self.tier_configs[t_idx]
            mask = current_tiers == t_idx
            indices = all_idx[mask]
            results = all_res[mask]
            
            if config.type == TierType.LADDER:
                points_change = np.zeros(len(indices), dtype=int)
                points_change[results == 1] = config.points_win
                points_change[results == 0] = config.points_draw
                
                self.user_ladder_points[indices] += points_change
                
                prom_mask = self.user_ladder_points[indices] >= config.promotion_points
                if t_idx < len(self.tier_configs) - 1:
                    prom_indices = indices[prom_mask]
                    if len(prom_indices) > 0:
                        self.user_tier_index[prom_indices] += 1
                        self.user_ladder_points[prom_indices] = 0
                        self.user_demotion_lives[prom_indices] = 0
                        self.promotion_counts[t_idx + 1] = self.promotion_counts.get(t_idx + 1, 0) + len(prom_indices)
                
                reset_mask = results >= 0
                self.user_demotion_lives[indices[reset_mask]] = 0
                
                loss_mask = results == -1
                if config.demotion_lives > 0:
                    self.user_demotion_lives[indices[loss_mask]] += 1
                    demote_mask = self.user_demotion_lives[indices] >= config.demotion_lives
                    
                    if t_idx > 0:
                        dem_indices = indices[demote_mask]
                        if len(dem_indices) > 0:
                            self.user_tier_index[dem_indices] -= 1
                            self.user_ladder_points[dem_indices] = 0
                            self.user_demotion_lives[dem_indices] = 0
                            self.demotion_counts[t_idx] = self.demotion_counts.get(t_idx, 0) + len(dem_indices)
            
            elif config.type == TierType.MMR:
                current_mmrs = self.mmr[indices]
                prom_mask = current_mmrs >= config.max_mmr
                if t_idx < len(self.tier_configs) - 1:
                    prom_indices = indices[prom_mask]
                    if len(prom_indices) > 0:
                        self.user_tier_index[prom_indices] += 1
                        self.user_demotion_lives[prom_indices] = 0
                        self.promotion_counts[t_idx + 1] = self.promotion_counts.get(t_idx + 1, 0) + len(prom_indices)
                
                at_min_mask = current_mmrs <= config.min_mmr
                loss_mask = results == -1
                risk_mask = at_min_mask & loss_mask
                
                if config.demotion_lives > 0:
                    safe_mask = results >= 0
                    self.user_demotion_lives[indices[safe_mask]] = 0
                    self.user_demotion_lives[indices[risk_mask]] += 1
                    demote_ready = self.user_demotion_lives[indices] > config.demotion_lives
                    
                    if t_idx > 0:
                        dem_indices = indices[demote_ready]
                        if len(dem_indices) > 0:
                            self.user_tier_index[dem_indices] -= 1
                            self.user_demotion_lives[dem_indices] = 0
                            self.demotion_counts[t_idx] = self.demotion_counts.get(t_idx, 0) + len(dem_indices)

    def _assign_placement_tier(self, user_indices):
        current_mmr = self.mmr[user_indices]
        for t_idx, config in enumerate(self.tier_configs):
            if config.placement_max_mmr > 0:
                in_range_mask = (current_mmr >= config.placement_min_mmr) & (current_mmr < config.placement_max_mmr)
                if in_range_mask.any():
                    target_users = user_indices[in_range_mask]
                    self.user_tier_index[target_users] = t_idx
                    self.user_ladder_points[target_users] = 0
                    self.user_demotion_lives[target_users] = 0

    def _update_daily_tiers(self):
        for t_idx in range(len(self.tier_configs) - 1, 0, -1):
            config = self.tier_configs[t_idx]
            prev_config = self.tier_configs[t_idx - 1]
            
            if config.type == TierType.RATIO:
                # Candidates: Users currently in This Tier OR Previous Tier
                # Actually, strictly "Previous Tier" users are candidates.
                # But users already in This Tier are also candidates to stay.
                
                target_mask = (self.user_tier_index == t_idx) | (self.user_tier_index == t_idx - 1)
                candidate_indices = self.ids[target_mask]
                
                if len(candidate_indices) == 0:
                    continue
                    
                # Sort by MMR (Descending)
                # Add tie-breaker? (e.g. ID)
                candidate_mmrs = self.mmr[candidate_indices]
                # Negative for descending sort
                sorted_args = np.argsort(-candidate_mmrs)
                sorted_candidates = candidate_indices[sorted_args]
                
                # Cutoff
                capacity = int(config.capacity)
                if capacity > len(sorted_candidates):
                    capacity = len(sorted_candidates)
                    
                promoted = sorted_candidates[:capacity]
                demoted = sorted_candidates[capacity:]
                
                # Apply
                self.user_tier_index[promoted] = t_idx
                self.user_tier_index[demoted] = t_idx - 1
                
                # Track (Approximate, since we don't know who moved exactly without diff)
                # But we can assume new entrants are promotions?
                # Actually, Ratio is tricky. Let's skip tracking for Ratio for now or do it properly later.
                # Or just count total in tier.



    def get_stats(self):
        return {
            "day": self.day,
            "avg_mmr": np.mean(self.mmr),
            "min_mmr": np.min(self.mmr),
            "max_mmr": np.max(self.mmr),
            "std_mmr": np.std(self.mmr)
        }

    def apply_soft_reset(self, compression_factor: float, target_mean: float):
        """
        Applies soft reset to all users (Vectorized).
        """
        self.mmr = target_mean + (self.mmr - target_mean) * compression_factor
        
        # Reset Seasonal Stats
        self.matches_played.fill(0)
        self.wins.fill(0)
        self.losses.fill(0)
        self.draws.fill(0)
        self.streak.fill(0)
        
        # Clear logs for the new season to keep inspector clean? 
        # Or maybe we should keep them. For FastMode, let's clear them to save memory/confusion.
        for k in self.match_logs:
            self.match_logs[k] = []

    def apply_tiered_reset(self, rules: List[Dict]):
        """
        Applies soft reset based on tiered rules (Vectorized).
        """
        old_mmr = self.mmr.copy()
        
        for rule in rules:
            min_val = rule['min']
            max_val = rule['max']
            target = rule['target']
            comp = rule['compression']
            
            # Mask for users in this range
            mask = (old_mmr >= min_val) & (old_mmr < max_val)
            
            if np.any(mask):
                self.mmr[mask] = target + (old_mmr[mask] - target) * comp
                
        # Reset Seasonal Stats
        self.matches_played.fill(0)
        self.wins.fill(0)
        self.losses.fill(0)
        self.draws.fill(0)
        self.streak.fill(0)
        
        for k in self.match_logs:
            self.match_logs[k] = []
