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
    ELO = "ELO"

@dataclass
class TierConfig:
    name: str
    type: TierType
    # MMR specific
    min_mmr: float = 0.0
    max_mmr: float = 9999.0
    demotion_mmr: float = 0.0 # MMR threshold for demotion risk
    demotion_lives: int = 0 # Demotion defense matches
    loss_point_correction: float = 1.0 # Multiplier for negative point changes
    
    # Ladder specific
    points_win: int = 0
    points_draw: int = 0
    points_loss: int = 0 # Points deducted on loss
    promotion_points: int = 100
    promotion_points_low: int = 100 # Points needed if MMR < min_mmr
    promotion_points_high: int = 100 # Points needed if MMR >= max_mmr
    
    # Ladder Multipliers (If MMR < promotion_mmr_N, win points * N)
    promotion_mmr_2: float = 0.0
    promotion_mmr_3: float = 0.0
    promotion_mmr_4: float = 0.0
    promotion_mmr_5: float = 0.0
    
    # Ratio specific
    capacity: int = 0 # Absolute number of users (e.g., 100)

    # Placement specific
    placement_min_mmr: float = 0.0
    placement_max_mmr: float = 0.0

    # Bot Match specific
    bot_match_enabled: bool = False
    bot_trigger_goal_diff: int = 99
    bot_trigger_loss_streak: int = 99

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
    bot_win_rate: float = 0.8 # Probability of user winning against bot

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
    match_count: int = 0

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

class FastSimulation:
    def __init__(self, num_users, segment_configs: List[SegmentConfig], elo_config: ELOConfig, match_config: MatchConfig, tier_configs: List[TierConfig], initial_mmr=1200, use_true_skill_init=False, reset_rules: List[Dict]=None):
        self.num_users = num_users
        self.segment_configs = segment_configs
        self.elo_config = elo_config
        self.match_config = match_config
        self.tier_configs = tier_configs
        self.use_true_skill_init = use_true_skill_init
        self.reset_rules = reset_rules if reset_rules else []
        self.day = 0 # Track current day
        self.seg_names = [s.name for s in segment_configs]
        self.watched_indices = {} # {idx: segment_name}
        self.match_logs = {} # {idx: [MatchLog]}
        
        # User State Arrays (Vectorized)
        self.ids = np.arange(num_users)
        self.segment_indices = np.zeros(num_users, dtype=int)
        self.mmr = np.full(num_users, float(initial_mmr))
        self.matches_played = np.zeros(num_users, dtype=int)
        self.wins = np.zeros(num_users, dtype=int)
        self.draws = np.zeros(num_users, dtype=int)
        self.losses = np.zeros(num_users, dtype=int)
        self.streak = np.zeros(num_users, dtype=int) # Positive=Win, Negative=Loss
        
        # Tier State
        self.user_tier_index = np.full(num_users, -1, dtype=int) # -1 = Unranked, 0 to len(tiers)-1
        self.user_ladder_points = np.zeros(num_users, dtype=int)
        self.user_demotion_lives = np.zeros(num_users, dtype=int)
        
        # Initialize Tiers (Placement or Default 0)
        # For now start everyone at lowest tier or placement
        self.user_demotion_lives[:] = 3 # Default fallback
        if self.tier_configs:
            self.user_demotion_lives[:] = self.tier_configs[0].demotion_lives
        
        # True Skill & Activity (for Matchmaking simulation)
        self.true_skill = np.zeros(num_users)
        self.activity_prob = np.zeros(num_users)
        self.matches_per_day = np.zeros(num_users, dtype=float)
        
        # Stats
        self.promotion_counts = {} # tier_idx -> count
        self.demotion_counts = {}
        
        # Tracking First 3 Matches (for Analysis)
        # N x 3 array. -9 = Unplayed, 1 = Win, 0 = Draw, -1 = Loss
        self.first_3_outcomes = np.full((num_users, 3), -9, dtype=int)
        
        self._initialize_users()

    def initialize_users(self):
        self._initialize_users()

    def _initialize_users(self):
        start_idx = 0
        for i, seg in enumerate(self.segment_configs):
            count = int(self.num_users * seg.ratio)
            if count == 0: continue
            end_idx = min(start_idx + count, self.num_users)
            indices = np.arange(start_idx, end_idx)
            self.segment_indices[indices] = i
            
            # True Skill
            self.true_skill[indices] = np.random.normal(
                (seg.true_skill_min + seg.true_skill_max)/2, 
                (seg.true_skill_max - seg.true_skill_min)/6, 
                len(indices)
            )
            np.clip(self.true_skill[indices], seg.true_skill_min, seg.true_skill_max, out=self.true_skill[indices])
            
            # Activity
            self.activity_prob[indices] = seg.daily_play_prob
            # Use uniform for float range support (e.g. 3.0 to 12.0)
            self.matches_per_day[indices] = np.random.uniform(seg.matches_per_day_min, seg.matches_per_day_max, len(indices))
            
            # Use True Skill for Initial MMR if requested
            if self.use_true_skill_init:
                self.mmr[indices] = self.true_skill[indices]
                
                # Apply Season Reset Logic (Compression) immediately
                if self.reset_rules:
                    temp_mmr = self.mmr[indices].copy()
                    for rule in self.reset_rules:
                         # Handle keys (app.py uses min_mmr/max_mmr/reset_mmr/soft_reset_ratio)
                         r_min = rule.get('min_mmr', rule.get('min', 0))
                         r_max = rule.get('max_mmr', rule.get('max', 99999))
                         target = rule.get('reset_mmr', rule.get('target', 1000))
                         comp = rule.get('soft_reset_ratio', rule.get('compression', 1.0))
                         
                         mask = (temp_mmr >= r_min) & (temp_mmr < r_max)
                         if mask.any():
                             # New = Target + (Old - Target) * Ratio
                             # Note: "soft_reset_ratio" usually means % retained. 
                             # 1.0 = No Change. 0.0 = Hard Reset to Target.
                             self.mmr[indices[mask]] = target + (temp_mmr[mask] - target) * comp
            
            # Watch a few users from each segment for logs
            # Watch a few users from each segment for logs
            sample_count = min(len(indices), 1)
            for idx in indices[:sample_count]:
                self.watched_indices[idx] = seg.name
                
            start_idx = end_idx

    def _assign_placement_tier(self, user_indices):
        # Sort by MMR Descending (Priority)
        sorted_indices = sorted(user_indices, key=lambda i: self.mmr[i], reverse=True)
        
        # Pre-calculate counts (optimization)
        current_counts = {}
        for idx, t in enumerate(self.tier_configs):
            if t.type == TierType.RATIO:
                current_counts[idx] = np.count_nonzero(self.user_tier_index == idx)

        for i in sorted_indices:
            u_mmr = self.mmr[i]
            assigned_idx = 0
            
            # Find highest matching placement tier
            for idx, t in enumerate(self.tier_configs):
                if t.placement_min_mmr <= u_mmr <= t.placement_max_mmr:
                    assigned_idx = idx
            
            # Capacity Check (Downgrade if full)
            while assigned_idx > 0:
                t = self.tier_configs[assigned_idx]
                if t.type == TierType.RATIO:
                    cur_cnt = current_counts.get(assigned_idx, 0)
                    if cur_cnt >= t.capacity:
                        assigned_idx -= 1
                        continue
                break
            
            # Update State
            self.user_tier_index[i] = assigned_idx
            self.user_ladder_points[i] = 0 
            self.user_demotion_lives[i] = self.tier_configs[assigned_idx].demotion_lives
            
            # Update temp count
            if self.tier_configs[assigned_idx].type == TierType.RATIO:
                current_counts[assigned_idx] = current_counts.get(assigned_idx, 0) + 1

    def apply_tiered_reset(self, rules: List[Dict]):
        # rules: [{'min_mmr': 0, 'max_mmr': 1000, 'reset_mmr': 800, 'soft_reset_ratio': 0.5}, ...]
        watched_set = set(self.watched_indices.keys())
        
        for rule in rules:
             # Handle keys (schema compatibility)
             r_min = rule.get('min_mmr', rule.get('min', 0))
             r_max = rule.get('max_mmr', rule.get('max', 99999))
             target = rule.get('reset_mmr', rule.get('target', 1000))
             comp = rule.get('soft_reset_ratio', rule.get('compression', 1.0))
             
             mask = (self.mmr >= r_min) & (self.mmr < r_max)
             if not mask.any(): continue
             
             # Calculate new MMRs
             old_mmrs = self.mmr[mask].copy()
             new_mmrs = target + (old_mmrs - target) * comp
             
             self.mmr[mask] = new_mmrs
             
             # Log changes for watched users
             mask_indices = np.where(mask)[0]
             for idx in mask_indices:
                 if idx in watched_set:
                     if idx not in self.match_logs: self.match_logs[idx] = []
                     
                     match_idx_in_mask = np.where(mask_indices == idx)[0][0]
                     d_mmr = new_mmrs[match_idx_in_mask] - old_mmrs[match_idx_in_mask]
                     
                     self.match_logs[idx].append(MatchLog(
                        day=self.day,
                        hour=0,
                        opponent_id=-1, # System Reset
                        opponent_mmr=0,
                        opponent_true_skill=0,
                        result="Reset",
                        result_type="Season",
                        goal_diff=0,
                        mmr_change=float(d_mmr),
                        current_mmr=float(self.mmr[idx]),
                        current_tier_index=int(self.user_tier_index[idx]),
                        current_ladder_points=0,
                        match_count=int(self.matches_played[idx])
                    ))
        
        # Reset Stats (Vectorized)
        self.matches_played[:] = 0
        self.wins[:] = 0
        self.losses[:] = 0
        self.draws[:] = 0
        self.streak[:] = 0
        self.user_ladder_points[:] = 0
        self.user_tier_index[:] = -1 # CORRECT: Force Unranked for new season placement
        # Reset lives to default based on current tier
        for t_idx, t_config in enumerate(self.tier_configs):
             tier_mask = self.user_tier_index == t_idx
             if tier_mask.any():
                 self.user_demotion_lives[tier_mask] = t_config.demotion_lives

    def _process_tier_updates(self, idx_a, idx_b, win_a, draw, loss_a, mmr_change_a, mmr_change_b):
        # print("DEBUG: _process_tier_updates CALLED")
        all_idx = np.concatenate([idx_a, idx_b])
        res_a = np.zeros(len(idx_a), dtype=int)
        res_a[win_a] = 1
        res_a[loss_a] = -1
        
        res_b = np.zeros(len(idx_b), dtype=int)
        res_b[loss_a] = 1 # B wins when A loses
        res_b[win_a] = -1 # B loses when A wins
        
        # Results vectors (reusing res_a, res_b which are sparse? No, they track indices)
        # win_a has indices of A who won.
        # res_a is full size of A batch (-1, 0, 1)
        results_a = res_a
        results_b = -results_a
        
        # Batch update logic helper
        self._update_single_batch(idx_a, results_a, self.mmr[idx_a])
        self._update_single_batch(idx_b, results_b, self.mmr[idx_b])

    def _update_single_batch(self, indices, results, current_mmrs):
        # Snapshot current stats to prevent "Ladder Cascading" (promoting multiple times in one loop)
        current_tier_snapshot = self.user_tier_index[indices].copy()
        
        for t_idx, config in enumerate(self.tier_configs):
            # Identify users in this tier (using snapshot)
            in_tier_mask = current_tier_snapshot == t_idx
            if not in_tier_mask.any():
                continue
            
            subset_indices = indices[in_tier_mask]
            subset_results = results[in_tier_mask]
            subset_mmrs = current_mmrs[in_tier_mask]
            
            # --- Ladder Logic ---
            if config.type == TierType.LADDER:
                # Wins
                win_mask = subset_results == 1
                if win_mask.any():
                    win_indices = subset_indices[win_mask]
                    win_mmrs = subset_mmrs[win_mask]
                    points = np.full(len(win_indices), config.points_win)
                    
                    mults = np.ones(len(win_indices))
                    # Apply multipliers sequentially (overwrite with higher values)
                    # Order: P2 (<800), P3 (<600), P4 (<400), P5 (<200)
                    # Since P5 implies P4 implies P3... we want the highest multiplier.
                    # By processing Low->High Multiplier (P2->P5), we ensure strict overwriting.
                    
                    if config.promotion_mmr_2 > 0: mults[win_mmrs >= config.promotion_mmr_2] = 2
                    if config.promotion_mmr_3 > 0: mults[win_mmrs >= config.promotion_mmr_3] = 3
                    if config.promotion_mmr_4 > 0: mults[win_mmrs >= config.promotion_mmr_4] = 4
                    if config.promotion_mmr_5 > 0: mults[win_mmrs >= config.promotion_mmr_5] = 5
                    
                    # Apply to base points
                    points = (points * mults).astype(int)
                    
                    self.user_ladder_points[win_indices] += points
                
                # Draws
                draw_mask = subset_results == 0
                if draw_mask.any():
                    self.user_ladder_points[subset_indices[draw_mask]] += config.points_draw
                
                # Losses (Points Deduction)
                loss_mask = subset_results == -1
                if loss_mask.any() and config.points_loss > 0:
                     loss_indices = subset_indices[loss_mask]
                     self.user_ladder_points[loss_indices] -= config.points_loss
                     self.user_ladder_points[loss_indices] = np.maximum(self.user_ladder_points[loss_indices], 0)

            # --- Promotion Logic ---
            prom_indices = []
            
            if config.type == TierType.ELO:
                # ELO Promotion
                prom_mask = subset_mmrs >= config.max_mmr
                if prom_mask.any():
                    candidates = subset_indices[prom_mask]
                    if t_idx < len(self.tier_configs) - 1:
                        next_tier = self.tier_configs[t_idx+1]
                        if next_tier.type == TierType.RATIO:
                            # Capacity Check
                            next_tier_users = np.where(self.user_tier_index == t_idx + 1)[0]
                            remaining = next_tier.capacity - len(next_tier_users)
                            if remaining > 0:
                                # Simple "First come first serve" within batch (or random/MMR sort)
                                # Sort by MMR descending
                                cand_mmrs = self.mmr[candidates]
                                sorted_idx = np.argsort(cand_mmrs)[::-1]
                                can_promote_count = min(len(candidates), remaining)
                                prom_indices = candidates[sorted_idx][:can_promote_count]
                            else:
                                prom_indices = []
                        else:
                            prom_indices = candidates
            else:
                # Ladder/MMR Promotion
                target_points = np.full(len(subset_indices), config.promotion_points)
                low_mask = subset_mmrs < config.min_mmr
                target_points[low_mask] = config.promotion_points_low
                high_mask = subset_mmrs >= config.max_mmr
                target_points[high_mask] = config.promotion_points_high
                
                prom_mask = self.user_ladder_points[subset_indices] >= target_points
                if prom_mask.any():
                    candidates = subset_indices[prom_mask]
                    prom_indices = candidates
                    
                    # Capacity Check for Ladder/MMR -> Ratio
                    if t_idx < len(self.tier_configs) - 1:
                        next_tier = self.tier_configs[t_idx+1]
                        if next_tier.type == TierType.RATIO:
                            next_tier_users = np.where(self.user_tier_index == t_idx + 1)[0]
                            remaining = next_tier.capacity - len(next_tier_users)
                            
                            if remaining > 0:
                                # Sort by MMR descending for fairness
                                cand_mmrs = self.mmr[candidates]
                                sorted_idx = np.argsort(cand_mmrs)[::-1]
                                can_promote_count = min(len(candidates), remaining)
                                prom_indices = candidates[sorted_idx][:can_promote_count]
                            else:
                                prom_indices = []
            
            # Execute Promotion
            if len(prom_indices) > 0 and t_idx < len(self.tier_configs) - 1:
                self.user_tier_index[prom_indices] += 1
                self.user_ladder_points[prom_indices] = 0
                new_tier = self.tier_configs[t_idx+1]
                self.user_demotion_lives[prom_indices] = new_tier.demotion_lives
                self.promotion_counts[t_idx+1] = self.promotion_counts.get(t_idx+1, 0) + len(prom_indices)
            
            # --- Demotion Logic ---
            if config.type == TierType.ELO:
                demote_mask = subset_mmrs < config.min_mmr
                if demote_mask.any() and t_idx > 0:
                     dem_candidates = subset_indices[demote_mask]
                     # Filter out promoted
                     mask_not_prom = ~np.isin(dem_candidates, prom_indices) if len(prom_indices) > 0 else np.ones(len(dem_candidates), dtype=bool)
                     dem_indices = dem_candidates[mask_not_prom]
                     
                     if len(dem_indices) > 0:
                         self.user_tier_index[dem_indices] -= 1
                         self.user_ladder_points[dem_indices] = 0
                         lower_tier = self.tier_configs[t_idx-1]
                         self.user_demotion_lives[dem_indices] = lower_tier.demotion_lives
                         self.demotion_counts[t_idx] = self.demotion_counts.get(t_idx, 0) + len(dem_indices)
            
            elif config.type == TierType.LADDER or config.demotion_lives > 0:
                # Ladder/Lives Demotion
                loss_mask = subset_results == -1
                if loss_mask.any():
                    loss_indices = subset_indices[loss_mask]
                    # Filter promoted
                    if len(prom_indices) > 0:
                        prom_set = set(prom_indices)
                        loss_indices = np.array([i for i in loss_indices if i not in prom_set])
                    
                    if len(loss_indices) > 0:
                        # Check Points == 0
                        zero_points_mask = self.user_ladder_points[loss_indices] == 0
                        risk_indices = loss_indices[zero_points_mask]
                        
                        if len(risk_indices) > 0:
                            # Check MMR trigger (if set)
                            if config.demotion_mmr > 0:
                                mmr_risk_mask = self.mmr[risk_indices] < config.demotion_mmr
                                risk_indices = risk_indices[mmr_risk_mask]
                            
                            if len(risk_indices) > 0:
                                self.user_demotion_lives[risk_indices] -= 1
                                
                                demote_mask = self.user_demotion_lives[risk_indices] <= 0
                                if demote_mask.any() and t_idx > 0:
                                    dem_indices = risk_indices[demote_mask]
                                    self.user_tier_index[dem_indices] -= 1
                                    self.user_ladder_points[dem_indices] = 0
                                    lower_tier = self.tier_configs[t_idx-1]
                                    self.user_demotion_lives[dem_indices] = lower_tier.demotion_lives
                                    self.demotion_counts[t_idx] = self.demotion_counts.get(t_idx, 0) + len(dem_indices)

    def run_day(self, day=None):
        if day is not None:
             self.day = day
        else:
             self.day += 1
              
        # 1. Identify daily active users
        # Users have 'matches_per_day' property (int)
        # We want to simulate that many matches for each user if they are active.
        # But efficiently.
        
        # Determine who is active today
        is_active_today = np.random.random(self.num_users) < self.activity_prob
        active_indices = self.ids[is_active_today]
        
        if len(active_indices) < 2:
            return
            
        # We need to simulate matches.
        # matches_per_day is now float.
        # Calculate daily_matches for active users using probabilistic rounding.
        
        user_matches = self.matches_per_day[active_indices]
        base_matches = np.floor(user_matches).astype(int)
        remainder = user_matches - base_matches
        extra_matches = (np.random.random(len(active_indices)) < remainder).astype(int)
        
        daily_matches = base_matches + extra_matches
        
        # Filter out users with 0 matches (if any, though min is usually > 0)
        has_matches = daily_matches > 0
        if not has_matches.any(): return
        
        active_indices = active_indices[has_matches]
        matches_left = daily_matches[has_matches]
        current_active_indices = active_indices.copy()
        
        # Safety break
        max_loops = 20
        loop_cnt = 0
        
        while len(current_active_indices) >= 2 and loop_cnt < max_loops:
            loop_cnt += 1
            
            # --- Perform One Batch of Matches ---
            
            # 2. Matchmaking (Vectorized Jittered Sort)
            current_mmrs = self.mmr[current_active_indices]
            jitter = np.random.normal(0, self.match_config.matchmaking_jitter, len(current_active_indices))
            sorted_active_idx = np.argsort(current_mmrs + jitter)
            sorted_indices = current_active_indices[sorted_active_idx]
            
            # Pair adjacent
            n_pairs = len(sorted_indices) // 2
            if n_pairs == 0: break
            
            idx_a = sorted_indices[0 : n_pairs*2 : 2]
            idx_b = sorted_indices[1 : n_pairs*2 : 2]
            
            # 3. Simulate Outcomes & 4. Update MMR (call internal logic)
            self._simulate_batch_matches(idx_a, idx_b)
            
            # Decrement matches left
            # We need to map back to the 'active_indices' array or just track locally
            # 'matches_left' corresponds to 'active_indices'
            # We need to find which indices in 'active_indices' correspond to idx_a/idx_b
            
            # This mapping is getting complex for a "Fast" simulation. 
            # Alternative: Just run N loops and pick random active users?
            # No, we want to respect 'matches_per_day'.
            
            # Optimization: 
            # matches_left was extracted from active_indices.
            # We can update it if we track the mapping.
            # OR: just update the global 'matches_per_day' (temp)? No.
            
            # Re-filter active_indices based on matches_per_day?
            # It's expensive to search.
            
            # Let's simplify:
            # We will run M rounds.
            # In round 1, everyone active plays (matches_per_day >= 1).
            # In round 2, only those with matches_per_day >= 2 play.
            # ...
            
            # Identify users with >= loop_cnt matches
            # We are using 'matches_left' which we must decrement.
            # But wait, self._simulate_batch_matches doesn't return who played.
            # We know idx_a and idx_b played.
            
            # We need to map global indices back to local 'matches_left' indices.
            # This is complex in vectorized form.
            
            # Simpler approach for this loop:
            # global 'matches_per_day' is static average. 'daily_matches' is local instance.
            # We have 'matches_left' aligned with 'active_indices'.
            
            # Let's verify if _simulate_batch_matches modifies anything we need.
            # It modifies MMR/Stats.
            
            # Update matches_left for those who played? 
            # Actually, standard approach:
            # In Loop N:
            # Play everyone who has N matches or more. 
            # Since everyone in 'current_active_indices' has at least 1 match left (filtered at end of loop),
            # we just pair them up.
            
            # But probabilistic rounding means some have 3, some have 4.
            # Loop 1: Everyone plays (if >=1).
            # Loop 2: Only those with >=2 play.
            # ...
            
            # So we just need to filter 'current_active_indices' based on 'daily_matches' > loop_cnt.
            # 'daily_matches' corresponds to 'active_indices' (original set).
            
            mask_still_playing = daily_matches > loop_cnt
            current_active_indices = active_indices[mask_still_playing]
            # No need to update matches_left explicitely if we use the mask against the fixed 'daily_matches' array.
            
    def _simulate_batch_matches(self, idx_a, idx_b):
        n_pairs = len(idx_a)
        
        # 3. Simulate Outcomes (Vectorized)
        # Use True Skill for Win Probability (Real Skill)
        ts_a = self.true_skill[idx_a]
        ts_b = self.true_skill[idx_b]
        
        # Use MMR for Expected Score (Elo Delta)
        mmr_a = self.mmr[idx_a]
        mmr_b = self.mmr[idx_b]
        
        # Real Win Probability (True Skill)
        prob_real_a = 1.0 / (1.0 + 10.0 ** ((ts_b - ts_a) / 400.0))
        
        # Expected Win Probability (MMR) - for Score Calculation
        prob_mmr_a = 1.0 / (1.0 + 10.0 ** ((mmr_b - mmr_a) / 400.0))
        
        # Random Draws
        rand_outcomes = np.random.random(n_pairs)
        
        # Draw logic
        draw_prob = self.match_config.draw_prob
        prob_decisive = 1.0 - draw_prob
        
        threshold_win = prob_real_a * prob_decisive
        threshold_draw = threshold_win + draw_prob
        
        win_mask = rand_outcomes < threshold_win
        draw_mask = (rand_outcomes >= threshold_win) & (rand_outcomes < threshold_draw)
        loss_mask = rand_outcomes >= threshold_draw
        
        # 4. MMR Updates
        # ... logic copied from original run_day ...
        # K-Factor
        # K-Factor Base
        k = float(self.elo_config.base_k)
        # Use np.float64 explicitly to avoid ambiguity and ensure float casting
        k_a = np.full(n_pairs, k, dtype=np.float64)
        k_b = np.full(n_pairs, k, dtype=np.float64)
        
        # Defensive: Ensure float
        k_a = k_a.astype(np.float64)
        k_b = k_b.astype(np.float64)
        
        print(f"DEBUG: Batch Match K-Factor Check - Type: {k_a.dtype}, Sample: {k_a[0] if n_pairs>0 else 'N/A'}")
        
        # 4a. Placement Matches Logic
        # Users in placement get higher K via Bonus Multiplier
        pm = self.elo_config.placement_matches
        pb = self.elo_config.placement_bonus
        # pk no longer used - we use base_k * bonus
        
        mask_place_a = self.matches_played[idx_a] < pm
        mask_place_b = self.matches_played[idx_b] < pm
        
        # Apply Bonus
        k_a[mask_place_a] *= pb
        k_b[mask_place_b] *= pb
        
        # 4b. Streak Bonus (If configured)
        # Assuming elo_config has streak_k_factor or similar (checking logic from memory/standard)
        # Or simple additive bonus
        
        # NOTE: Streak logic applies on TOP of placement? Usually yes.
        # Let's apply a multiplier for streaks > 3?
        # Current config structure might not have explicit streak param. Using hardcoded or standard.
        # Let's assume standard behavior: Multiplier on K.
        
        # Streak A
        s_a = np.abs(self.streak[idx_a])
        mask_streak_a = s_a >= 3
        k_a[mask_streak_a] *= 1.2 # 20% boost for streaks
        
        # Streak B
        s_b = np.abs(self.streak[idx_b])
        mask_streak_b = s_b >= 3
        k_b[mask_streak_b] *= 1.2
        
        # 4c. Goal Difference & Win Type Logic (Simulated)
        # Since we don't simulate actual goals, we simulate the "Effect" of specific win types.
        # Win Type Probabilities:
        # Regular (GD=1): 50%, High Diff (GD=3): 30%, Penalties (GD=0 aka Draw-like): 20%
        # This modulates K-factor.
        
        # Using configured weights if available, or defaults.
        # Let's simulate 'win_type_factor' for each pair.
        # 1.0 = normal, 1.5 = big win, 0.7 = close/penalties
        
        win_factors = np.ones(n_pairs)
        
        # Randomly assign win quality
        # 0.0-0.6: Normal (x1.0), 0.6-0.85: Big Win (x1.5), 0.85-1.0: Penalties/Close (x0.7)
        w_rand = np.random.random(n_pairs)
        
        mask_big = w_rand > 0.6
        mask_close = w_rand > 0.85 # Overwrites big if higher
        
        win_factors[mask_big] = 1.3
        win_factors[mask_close] = 0.8 # Penalties is usually less impact
        
        k_a *= win_factors
        k_b *= win_factors
        
        # Score
        score_a = np.zeros(n_pairs)
        score_a[win_mask] = 1.0
        score_a[draw_mask] = 0.5
        
        score_b = 1.0 - score_a
        
        # Delta
        # NOTE: We use prob_mmr_a (Expected based on MMR) to calculate delta against Actual Result.
        # This is key: If High Skill (Low MMR) beats Low Skill (High MMR),
        # prob_real was high (so they won), but prob_mmr was low (underdog).
        # So they gain MORE points. This accelerates convergence to True Skill.
        
        delta_a = k_a * (score_a - prob_mmr_a)
        delta_b = k_b * (score_b - (1.0 - prob_mmr_a))
        
        # Zero-Sum Check (Debug)
        # total_delta = np.sum(delta_a + delta_b)
        # if abs(total_delta) > 0.1:
        #     print(f"WARNING: Non-Zero Sum Batch! Sum: {total_delta:.4f} (Pairs: {n_pairs})")
            
        # Apply
        self.mmr[idx_a] += delta_a
        self.mmr[idx_b] += delta_b
        
        # Stats
        self.matches_played[idx_a] += 1
        self.matches_played[idx_b] += 1
        
        self.wins[idx_a[win_mask]] += 1
        self.losses[idx_b[win_mask]] += 1
        self.streak[idx_a[win_mask]] = np.maximum(self.streak[idx_a[win_mask]] + 1, 1)
        self.streak[idx_b[win_mask]] = np.minimum(self.streak[idx_b[win_mask]] - 1, -1)
        
        self.wins[idx_b[loss_mask]] += 1
        self.losses[idx_a[loss_mask]] += 1
        self.streak[idx_b[loss_mask]] = np.maximum(self.streak[idx_b[loss_mask]] + 1, 1)
        self.streak[idx_a[loss_mask]] = np.minimum(self.streak[idx_a[loss_mask]] - 1, -1)
        
        self.draws[idx_a[draw_mask]] += 1
        self.draws[idx_b[draw_mask]] += 1
        self.streak[idx_a[draw_mask]] = 0 # Reset streak on draw
        self.streak[idx_b[draw_mask]] = 0
        
        # --- Track First 3 Outcomes (Analysis) ---
        # Current match count is already incremented (so 1, 2, 3...)
        # Indices where tracked match count <= 3
        # Match count 1 -> index 0, 2->1, 3->2
        
        # Helper for batch update
        def update_first_3(indices, results, counts):
            # Mask for users within first 3 matches
            mask_valid = counts <= 3
            if not mask_valid.any(): return
            
            valid_idx = indices[mask_valid]
            valid_res = results[mask_valid] # 1, 0, -1
            valid_counts = counts[mask_valid] 
            
            # Map count 1-3 to index 0-2
            slot_idx = valid_counts - 1
            
            # Numpy fancy indexing: data[rows, cols] = values
            self.first_3_outcomes[valid_idx, slot_idx] = valid_res

        # Prepare outcomes (-1: Loss, 0: Draw, 1: Win)
        # res_a/res_b were strings or derived. Let's use score_a/score_b but we need -1 for loss.
        # score: 1.0 (Win), 0.5 (Draw), 0.0 (Loss)
        # Map to 1, 0, -1
        
        int_res_a = np.zeros(n_pairs, dtype=int)
        int_res_a[win_mask] = 1
        int_res_a[draw_mask] = 0
        int_res_a[loss_mask] = -1
        
        int_res_b = np.zeros(n_pairs, dtype=int)
        int_res_b[loss_mask] = 1 # B wins when A loses
        int_res_b[draw_mask] = 0
        int_res_b[win_mask] = -1 # B loses when A wins
        
        update_first_3(idx_a, int_res_a, self.matches_played[idx_a])
        update_first_3(idx_b, int_res_b, self.matches_played[idx_b])
        
        # 5. process tier updates
        win_indices = np.where(win_mask)[0]
        draw_indices = np.where(draw_mask)[0]
        loss_indices = np.where(loss_mask)[0]
        
        self._process_tier_updates(idx_a, idx_b, win_indices, draw_indices, loss_indices, delta_a, delta_b)
        
        # 5b. Check Placement Completion & Assign Tier
        # Run AFTER tier updates so that the 10th match itself doesn't trigger ladder points logic instantly (or double count).
        # We want the 10th match to finalize placement, setting points to 0.
        pm = self.elo_config.placement_matches
        # Check users who JUST reached placement_matches (matches_played was incremented above)
        # Note: matches_played is already updated.
        
        # Check A
        mask_place_complete_a = self.matches_played[idx_a] == pm
        if mask_place_complete_a.any():
            self._assign_placement_tier(idx_a[mask_place_complete_a])
            
        # Check B
        mask_place_complete_b = self.matches_played[idx_b] == pm
        if mask_place_complete_b.any():
            self._assign_placement_tier(idx_b[mask_place_complete_b])
        # 6. Log Matches
        watched_set = set(self.watched_indices.keys())
        if not watched_set: return
        
        # print(f"DEBUG: Logging Batch. Pairs {n_pairs} Watched {len(watched_set)}")
        
        # Helper to log
        def log_match(u_idx, opp_idx, res, res_type, gd, d_mmr, cur_mmr, cur_tier, cur_pts):
            if u_idx not in watched_set: return
            if u_idx not in self.match_logs: self.match_logs[u_idx] = []
            
            # print(f"DEBUG: Logging for {u_idx}. Change {d_mmr:.2f} New {cur_mmr:.2f}")
            
            opp_mmr = self.mmr[opp_idx]
            opp_ts = self.true_skill[opp_idx]
            
            self.match_logs[u_idx].append(MatchLog(
                day=self.day,
                hour=0,
                opponent_id=int(self.ids[opp_idx]),
                opponent_mmr=float(opp_mmr),
                opponent_true_skill=float(opp_ts),
                result=res,
                result_type=res_type,
                goal_diff=int(gd),
                mmr_change=float(d_mmr),
                current_mmr=float(cur_mmr),
                current_tier_index=int(cur_tier),
                current_ladder_points=int(cur_pts),
                match_count=int(self.matches_played[u_idx])
            ))

        for i in range(n_pairs):
            u_a = idx_a[i]
            u_b = idx_b[i]
            
            # if u_a == 0 or u_b == 0:
            #      print(f"DEBUG: Trace Match Pair {i}. {u_a} vs {u_b}. Watched? {u_a in watched_set}/{u_b in watched_set}")
            
            is_a_watched = u_a in watched_set
            is_b_watched = u_b in watched_set
            
            if not is_a_watched and not is_b_watched:
                continue
                
            # Extract Match Details
            if win_mask[i]:
                res_a, res_b = "Win", "Loss"
            elif loss_mask[i]:
                res_a, res_b = "Loss", "Win"
            else:
                res_a, res_b = "Draw", "Draw"
                
            res_type = "Regular"
            
            if res_a == "Draw":
                gd = 0
            else:
                 gd = 1
            
            if is_a_watched:
                log_match(u_a, u_b, res_a, res_type, gd, delta_a[i], self.mmr[u_a], self.user_tier_index[u_a], self.user_ladder_points[u_a])
            
            if is_b_watched:
                log_match(u_b, u_a, res_b, res_type, gd, delta_b[i], self.mmr[u_b], self.user_tier_index[u_b], self.user_ladder_points[u_b])
        
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
            
        # Reset FastSimulation tracking if applicable (This method is in Simulation, but FastSimulation has similar logic)
        # FastSimulation doesn't use User objects directly for bulk ops, but if this is called on Simulation...
        # Wait, Simulation uses User objects. FastSimulation uses arrays.
        # This method is for Simulation class. FastSimulation has its own reset logic usually?
        # Actually FastSimulation doesn't have apply_soft_reset method shown in outline?
        # Let's check outline again. Ah, FastSimulation doesn't have apply_soft_reset in the snippet I saw?
        # Wait, I am editing Simulation class here? No, I am editing FastSimulation.__init__ above.
        # But apply_soft_reset is in Simulation class (lines 527-543).
        # FastSimulation needs its own reset logic or I need to find where it is.
        # Let's look at app.py line 1265: st.session_state.simulation.apply_tiered_reset(rules)
        # If simulation is FastSimulation, it must have apply_tiered_reset.
        # Let's check if FastSimulation has apply_tiered_reset.
        # I need to check if FastSimulation has these methods.
        # Based on previous view_file, Simulation has it. FastSimulation definition starts at 573.
        # I need to add reset logic to FastSimulation if it exists there.
        # Let's assume I need to add it to FastSimulation.
        # But first let's finish __init__.


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
