import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from simulation_core import Simulation, FastSimulation, SegmentConfig, ELOConfig, MatchConfig

st.set_page_config(page_title="FC Online Rank Simulation", layout="wide")

import json
import os

# --- Configuration Persistence (Google Sheets) ---
from streamlit_gsheets import GSheetsConnection

def load_config():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Read the first worksheet. We assume it contains the JSON config in cell A1 or similar structure.
        # Actually, let's store it as a simple 2-column dataframe: [Key, Value] to be flexible, 
        # or just one cell with JSON string if we want to keep it simple.
        # Let's try reading as a DataFrame.
        df = conn.read()
        
        # Check if empty
        if df.empty:
            return {}
            
        # Strategy: We will store the entire JSON config string in the first cell (A1) of the sheet.
        # This avoids schema issues with complex nested JSONs in columns.
        # The dataframe read might interpret it as a header.
        # Let's assume we store: Header "ConfigJSON", Row 1: "{...}"
        
        if "ConfigJSON" in df.columns and len(df) > 0:
            json_str = df.iloc[0]["ConfigJSON"]
            return json.loads(json_str)
            
        return {}
    except Exception as e:
        # Fallback to local if secrets not found or connection fails (e.g. first run)
        # st.warning(f"Google Sheets Load Failed: {e}. Using local defaults.")
        if os.path.exists(CONFIG_FILE):
             try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
             except:
                 pass
        return {}

def save_config():
    config = {
        # Global
        "num_users": st.session_state.get("num_users", 1000),
        "num_days": st.session_state.get("num_days", 365),
        "initial_mmr": st.session_state.get("initial_mmr", 1000.0),
        # Match Config
        "draw_prob": st.session_state.get("draw_prob", 0.1),
        "prob_et": st.session_state.get("prob_et", 0.2),
        "prob_pk": st.session_state.get("prob_pk", 0.5),
        "max_goal_diff": st.session_state.get("max_goal_diff", 5),
        "matchmaking_jitter": st.session_state.get("matchmaking_jitter", 50.0),
        # ELO Config
        "base_k": st.session_state.get("base_k", 32),
        "placement_matches": st.session_state.get("placement_matches", 10),
        "placement_bonus": st.session_state.get("placement_bonus", 4.0),
        "streak_rules": st.session_state.get("streak_rules", pd.DataFrame([
            {"min_streak": 3, "bonus": 5.0},
            {"min_streak": 5, "bonus": 10.0}
        ])).to_dict('records') if 'streak_rules' in st.session_state else [],
        "goal_diff_rules": st.session_state.get("goal_diff_rules", pd.DataFrame([
            {"min_diff": 2, "bonus": 2.0},
            {"min_diff": 4, "bonus": 5.0}
        ])).to_dict('records') if 'goal_diff_rules' in st.session_state else [],
        "decay_et": st.session_state.get("decay_et", 0.8),
        "decay_pk": st.session_state.get("decay_pk", 0.6),
        "calibration_k_bonus": st.session_state.get("calibration_k_bonus", 2.0),
        "calibration_enabled": st.session_state.get("calibration_enabled", False),
        "calibration_match_count": st.session_state.get("calibration_match_count", 10),
        # Segments (Serialize)
        "segments": [
            {
                "name": s.name, "ratio": s.ratio, "daily_play_prob": s.daily_play_prob,
                "matches_per_day_min": s.matches_per_day_min, "matches_per_day_max": s.matches_per_day_max,
                "true_skill_min": s.true_skill_min, "true_skill_max": s.true_skill_max,
                "active_hour_start": s.active_hour_start, "active_hour_end": s.active_hour_end
            } for s in st.session_state.get("segments", [])
        ],
        # Reset Rules (Serialize DataFrame as records)
        "reset_rules": st.session_state.get("reset_rules", pd.DataFrame()).to_dict('records') if 'reset_rules' in st.session_state else [],
        # User Comments
        "user_comments": st.session_state.get("user_comments", "")
    }
    
    # Save to Google Sheets
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        json_str = json.dumps(config)
        # Create a DataFrame with one cell
        df_to_save = pd.DataFrame([{"ConfigJSON": json_str}])
        conn.update(data=df_to_save)
        # st.toast("Config saved to Google Sheets!")
    except Exception as e:
        # st.error(f"Failed to save to Google Sheets: {e}")
        # Fallback to local
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except:
            pass

# Load Config at Startup
if 'config_loaded' not in st.session_state:
    with st.spinner("Loading config from database..."):
        loaded_config = load_config()
        
    if loaded_config:
        # Apply to session state for widgets
        for k, v in loaded_config.items():
            if k == 'segments':
                try:
                    st.session_state.segments = [SegmentConfig(**s) for s in v]
                except TypeError:
                    st.warning("Config schema mismatch for segments. Using defaults.")
                    st.session_state.segments = [] # Will be re-initialized below
            elif k == 'reset_rules':
                st.session_state.reset_rules = pd.DataFrame(v)
            elif k == 'streak_rules':
                st.session_state.streak_rules = pd.DataFrame(v)
            elif k == 'goal_diff_rules':
                st.session_state.goal_diff_rules = pd.DataFrame(v)
            else:
                st.session_state[k] = v
        # Load comments specifically if not in loop
        if "user_comments" in loaded_config:
            st.session_state.user_comments = loaded_config["user_comments"]
            
    st.session_state.config_loaded = True

st.title("Rank Simulation")

# --- Sidebar: Global Settings ---
st.sidebar.header("Global Parameters")
num_users = st.sidebar.number_input("Number of Users", min_value=100, max_value=1000000, value=1000, step=100, help="시뮬레이션에 참여할 총 유저 수입니다.", key="num_users")
num_days = st.sidebar.number_input("Simulation Days", min_value=1, max_value=3650, value=30, step=1, help="시뮬레이션을 진행할 총 일수입니다.", key="num_days")
initial_mmr = st.sidebar.number_input("Initial MMR", min_value=0.0, value=1000.0, step=100.0, help="모든 유저의 시작 MMR입니다. 빠른 수렴을 위해 True Skill 평균값에 가깝게 설정하세요.", key="initial_mmr")

# --- Sidebar: Advanced ELO & Match Config ---
with st.sidebar.expander("Advanced ELO Settings", expanded=True):
    st.subheader("Match Configuration")
    draw_prob = st.slider("Draw Probability (Regular Time)", 0.0, 0.5, 0.1, help="정규 시간 내에 무승부가 발생할 확률입니다.", key="draw_prob")
    prob_et = st.slider("Extra Time Probability (if Draw)", 0.0, 1.0, 0.2, help="무승부 시 연장전으로 이어질 확률입니다.", key="prob_et")
    prob_pk = st.slider("PK Probability (if Draw in ET)", 0.0, 1.0, 0.5, help="연장전에서도 승부가 나지 않아 승부차기로 갈 확률입니다.", key="prob_pk")
    max_goal_diff = st.slider("Max Goal Difference", 1, 10, 5, help="한 경기에서 발생할 수 있는 최대 골 득실차입니다.", key="max_goal_diff")
    matchmaking_jitter = st.slider("Matchmaking Jitter (Noise)", 0.0, 200.0, 50.0, help="매칭 시 MMR에 추가되는 노이즈의 표준편차입니다. 높을수록 매칭 범위가 넓어집니다.", key="matchmaking_jitter")
    
    st.subheader("ELO Factors")
    base_k = st.number_input("Base K-Factor", 10, 100, 32, help="기본 K-Factor입니다. 승패에 따른 점수 변동폭의 기준이 됩니다.", key="base_k")
    
    st.markdown("**Placement Matches**")

    placement_matches = st.number_input("Placement Match Count", 0, 20, 10, help="배치고사로 간주되는 초기 매치 수입니다.", key="placement_matches")
    placement_bonus = st.slider("Placement K-Multiplier", 1.0, 5.0, 4.0, help="배치고사 기간 동안 적용되는 K-Factor 보너스 배율입니다.", key="placement_bonus")
    
    st.markdown("**Streak Bonus Rules (Additive K)**")
    if 'streak_rules' not in st.session_state:
        st.session_state.streak_rules = pd.DataFrame([
            {"min_streak": 3, "bonus": 5.0},
            {"min_streak": 5, "bonus": 10.0}
        ])
    
    st.session_state.streak_rules = st.data_editor(
        st.session_state.streak_rules,
        column_config={
            "min_streak": st.column_config.NumberColumn("Min Streak", min_value=2, step=1, required=True, help="보너스를 받기 위한 최소 연승/연패 수"),
            "bonus": st.column_config.NumberColumn("K-Factor Bonus", step=1.0, required=True, help="해당 연승/연패 시 추가되는 K-Factor")
        },
        num_rows="dynamic",
        key="streak_rules_editor",
        use_container_width=True
    )

    st.markdown("**Goal Difference Rules (Additive K)**")
    if 'goal_diff_rules' not in st.session_state:
        st.session_state.goal_diff_rules = pd.DataFrame([
            {"min_diff": 2, "bonus": 2.0},
            {"min_diff": 4, "bonus": 5.0}
        ])

    st.session_state.goal_diff_rules = st.data_editor(
        st.session_state.goal_diff_rules,
        column_config={
            "min_diff": st.column_config.NumberColumn("Min Goal Diff", min_value=1, step=1, required=True, help="보너스를 받기 위한 최소 골 득실차"),
            "bonus": st.column_config.NumberColumn("K-Factor Bonus", step=1.0, required=True, help="해당 득실차 승리 시 추가되는 K-Factor")
        },
        num_rows="dynamic",
        key="goal_diff_rules_editor",
        use_container_width=True
    )
    
    st.markdown("**Win Type Decay**")
    decay_et = st.slider("Extra Time Win Multiplier", 0.1, 1.0, 0.8, help="연장전 승리 시 획득 점수 배율입니다. (정규 시간 승리 대비)", key="decay_et")
    decay_pk = st.slider("PK Win Multiplier", 0.1, 1.0, 0.6, help="승부차기 승리 시 획득 점수 배율입니다.", key="decay_pk")
    
    st.markdown("**MMR Compression Correction**")
    uncertainty_factor = st.slider("Uncertainty Factor", 0.5, 1.0, 0.9, help="실력 차이가 클 때 승률 기대치를 낮춰 무승부/이변 확률을 보정합니다. 낮을수록 최상위권 MMR이 높아집니다.", key="uncertainty_factor")

    st.subheader("Calibration Mode")
    calibration_enabled = st.checkbox("Enable Calibration Mode", value=False, help="활성화 시, 초기 매치에서 True Skill을 기반으로 매칭하여 수렴 속도를 높입니다.", key="calibration_enabled")
    if calibration_enabled:
        calibration_match_count = st.number_input("Calibration Matches", 1, 50, 10, help="보정 모드가 적용되는 초기 매치 수입니다.", key="calibration_match_count")
        calibration_k_bonus = st.slider("Calibration K-Multiplier", 1.0, 5.0, 2.0, help="배치고사 기간 동안 적용되는 K-Factor 배율입니다.", key="calibration_k_bonus")
    else:
        calibration_match_count = 10
        calibration_k_bonus = 1.0

# --- Sidebar: Segment Configuration ---
st.sidebar.header("User Segments Configuration")

# Validate Schema for existing session state (Hot-reload fix)
if 'segments' in st.session_state and st.session_state.segments:
    try:
        _ = st.session_state.segments[0].matches_per_day_min
    except AttributeError:
        st.warning("Segment schema updated. Resetting segments to defaults.")
        del st.session_state.segments

if 'segments' not in st.session_state or not st.session_state.segments:
    st.session_state.segments = [
        SegmentConfig("Competitive", 0.2, 0.9, 18, 24, 1300.0, 1700.0, 18, 24),
        SegmentConfig("Social", 0.5, 0.5, 5, 10, 900.0, 1500.0, 16, 22),
        SegmentConfig("Casual", 0.3, 0.2, 1, 5, 600.0, 1400.0, 12, 20)
    ]

def segments_to_df(segments):
    data = []
    for s in segments:
        data.append({
            "Name": s.name,
            "Ratio": s.ratio,
            "Daily Play Prob": s.daily_play_prob,
            "Matches/Day (Min)": s.matches_per_day_min,
            "Matches/Day (Max)": s.matches_per_day_max,
            "True Skill (Min)": s.true_skill_min,
            "True Skill (Max)": s.true_skill_max,
            "Start Hour": s.active_hour_start,
            "End Hour": s.active_hour_end
        })
    return pd.DataFrame(data)

st.sidebar.subheader("Edit Segments")
df_segments = segments_to_df(st.session_state.segments)

column_config = {
    "Name": st.column_config.TextColumn("Name", required=True),
    "Ratio": st.column_config.NumberColumn("Ratio", format="%.2f"),
    "Daily Play Prob": st.column_config.NumberColumn("Daily Play Prob", format="%.2f"),
    "Matches/Day (Min)": st.column_config.NumberColumn("Matches/Day (Min)", min_value=0.1, max_value=100.0, step=0.1, format="%.1f"),
    "Matches/Day (Max)": st.column_config.NumberColumn("Matches/Day (Max)", min_value=0.1, max_value=100.0, step=0.1, format="%.1f"),
    "True Skill (Min)": st.column_config.NumberColumn("True Skill (Min)", step=10.0),
    "True Skill (Max)": st.column_config.NumberColumn("True Skill (Max)", step=10.0),
    "Start Hour": st.column_config.NumberColumn("Start Hour", format="%d"),
    "End Hour": st.column_config.NumberColumn("End Hour", format="%d")
}

edited_df = st.sidebar.data_editor(
    df_segments, 
    num_rows="dynamic",
    column_config=column_config,
    use_container_width=True,
    hide_index=True
)

with st.sidebar.expander("Bulk Import (Excel/CSV)", expanded=False):
    st.caption("Paste data here (Tab-separated for Excel, Comma for CSV). Columns: Name, Ratio, Prob, Matches(Min), Matches(Max), Skill(Min), Skill(Max), Start, End")
    paste_data = st.text_area("Paste Data", height=100)
    if st.button("Import Pasted Data"):
        try:
            from io import StringIO
            # Try tab separator first (Excel default)
            if "\t" in paste_data:
                sep = "\t"
            else:
                sep = ","
            
            new_df = pd.read_csv(StringIO(paste_data), sep=sep, header=None)
            # Expected 9 columns
            if new_df.shape[1] == 9:
                new_df.columns = df_segments.columns
                # Update session state
                new_segments = []
                for _, row in new_df.iterrows():
                    new_segments.append(SegmentConfig(
                        name=str(row[0]),
                        ratio=float(row[1]),
                        daily_play_prob=float(row[2]),
                        matches_per_day_min=float(row[3]),
                        matches_per_day_max=float(row[4]),
                        true_skill_min=float(row[5]),
                        true_skill_max=float(row[6]),
                        active_hour_start=int(row[7]),
                        active_hour_end=int(row[8])
                    ))
                st.session_state.segments = new_segments
                st.rerun()
            else:
                st.error(f"Expected 9 columns, found {new_df.shape[1]}")
        except Exception as e:
            st.error(f"Import failed: {e}")

updated_segments = []
total_ratio = 0
for index, row in edited_df.iterrows():
    try:
        seg = SegmentConfig(
            name=row["Name"],
            ratio=float(row["Ratio"]),
            daily_play_prob=float(row["Daily Play Prob"]),
            matches_per_day_min=float(row["Matches/Day (Min)"]),
            matches_per_day_max=float(row["Matches/Day (Max)"]),
            true_skill_min=float(row["True Skill (Min)"]),
            true_skill_max=float(row["True Skill (Max)"]),
            active_hour_start=int(row["Start Hour"]),
            active_hour_end=int(row["End Hour"])
        )
        updated_segments.append(seg)
        total_ratio += seg.ratio
    except Exception as e:
        st.sidebar.error(f"Error parsing segment row {index}: {e}")

# Sync back to session state to persist edits
st.session_state.segments = updated_segments

if abs(total_ratio - 1.0) > 0.01:
    st.sidebar.warning(f"Total Ratio is {total_ratio:.2f}. It should sum to 1.0")

# --- Run Simulation ---
reset_sim = st.sidebar.checkbox("Force Full Reset (New Simulation)", value=False, help="Check this to discard current progress and start over from Season 1.")

if st.sidebar.button("Run Simulation"):
    # Build Configs
    elo_config = ELOConfig(
        base_k=base_k,
        placement_matches=placement_matches,
        placement_bonus=placement_bonus,
        streak_rules=st.session_state.streak_rules.to_dict('records'),
        goal_diff_rules=st.session_state.goal_diff_rules.to_dict('records'),
        win_type_decay={'Regular': 1.0, 'Extra': decay_et, 'PK': decay_pk},
        uncertainty_factor=uncertainty_factor,
        calibration_k_bonus=calibration_k_bonus
    )
    
    match_config = MatchConfig(
        draw_prob=draw_prob,
        prob_extra_time=prob_et,
        prob_pk=prob_pk,
        max_goal_diff=max_goal_diff,
        matchmaking_jitter=matchmaking_jitter,
        calibration_enabled=calibration_enabled,
        calibration_match_count=calibration_match_count
    )
    
    with st.spinner("Simulating..."):
        # Determine if we are continuing or starting new
        if 'sim_result' in st.session_state and not reset_sim:
            sim = st.session_state.sim_result
            # Update Configs in case they changed
            sim.elo_config = elo_config
            sim.match_config = match_config
            # Ensure segment configs are updated if possible? 
            # Changing segments mid-sim is tricky. For now, we assume segments are fixed or we re-apply.
            # Re-applying segments to existing users is hard. We'll skip that for "Continue".
            st.info(f"Continuing Season {st.session_state.get('season_count', 1)}...")
        else:
            if num_users > 10000:
                st.warning("Running in Fast Mode (Numpy). Detailed logs are sampled.")
                sim = FastSimulation(num_users, updated_segments, elo_config, match_config, initial_mmr)
            else:
                sim = Simulation(num_users, updated_segments, elo_config, match_config, initial_mmr)
                
            sim.initialize_users()
            st.session_state.season_count = 1
            st.session_state.stats_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # If continuing, we append to existing history or start new history for this run?
    # The stats_history in session_state might be cleared by Soft Reset.
    # Let's just append to whatever is in session_state.stats_history
    if 'stats_history' not in st.session_state:
        st.session_state.stats_history = []
        
    current_history = st.session_state.stats_history
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stats_history = []
    
    for day in range(num_days):
        sim.run_day()
        stats = sim.get_stats()
        current_history.append(stats)
        
        progress_bar.progress((day + 1) / num_days)
        status_text.text(f"Simulating Day {day + 1}/{num_days}")
        
    status_text.text("Simulation Complete!")
    st.session_state.sim_result = sim
    st.session_state.stats_history = current_history
    if 'season_count' not in st.session_state:
        st.session_state.season_count = 1
    st.session_state.season_count = 1

# --- Season Management (Soft Reset) ---
if 'sim_result' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("Season Management")
    
    current_season = st.session_state.get('season_count', 1)
    st.sidebar.info(f"Current Season: {current_season}")
    
    with st.sidebar.expander("Soft Reset Settings (Tiered)", expanded=False):
        st.caption("Define reset rules for different MMR ranges. Rules are applied in order.")
        
        if 'reset_rules' not in st.session_state:
            st.session_state.reset_rules = pd.DataFrame([
                {"Min MMR": 0, "Max MMR": 10000, "Target MMR": 1000, "Compression": 0.5}
            ])
            
        column_config_rules = {
            "Min MMR": st.column_config.NumberColumn("Min MMR", required=True),
            "Max MMR": st.column_config.NumberColumn("Max MMR", required=True),
            "Target MMR": st.column_config.NumberColumn("Target MMR", required=True),
            "Compression": st.column_config.NumberColumn("Compression (0-1)", min_value=0.0, max_value=1.0, format="%.2f")
        }
        
        edited_rules = st.sidebar.data_editor(
            st.session_state.reset_rules,
            num_rows="dynamic",
            column_config=column_config_rules,
            use_container_width=True,
            hide_index=True
        )
        st.session_state.reset_rules = edited_rules
        
    if st.sidebar.button("End Season & Soft Reset"):
        sim = st.session_state.sim_result
        
        # Convert DF to list of dicts
        rules = []
        for _, row in edited_rules.iterrows():
            rules.append({
                'min': float(row['Min MMR']),
                'max': float(row['Max MMR']),
                'target': float(row['Target MMR']),
                'compression': float(row['Compression'])
            })
            
        sim.apply_tiered_reset(rules)
        
        st.session_state.season_count += 1
        st.session_state.stats_history = [] 
        
        save_config() # Save after reset
        st.success(f"Season {current_season} Ended. Tiered Soft Reset Applied!")
        st.rerun()

# Auto-save config at the end of script execution if values changed
# We can just call save_config() here. It's lightweight.
save_config()

# --- Comments Section ---
st.sidebar.markdown("---")
st.sidebar.header("Memo / Comments")
user_comments = st.sidebar.text_area("Leave your notes here:", value=st.session_state.get("user_comments", ""), height=150, help="시뮬레이션 설정이나 결과에 대한 메모를 남길 수 있습니다. (자동 저장)")
if user_comments != st.session_state.get("user_comments", ""):
    st.session_state.user_comments = user_comments
    save_config()

# --- Results Display ---
if 'sim_result' in st.session_state:
    sim = st.session_state.sim_result
    stats_history = st.session_state.stats_history
    is_fast_mode = isinstance(sim, FastSimulation)
    
    st.header("Simulation Results")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Segment Analysis", "Match Inspector"])
    
    with tab1:
        st.subheader("MMR Trends")
        st.subheader("MMR Trends")
        if stats_history:
            df_stats = pd.DataFrame(stats_history)
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['avg_mmr'], name='Average MMR'))
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['min_mmr'], name='Min MMR'))
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['max_mmr'], name='Max MMR'))
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No match history for this season yet. Run simulation to generate data.")
        
        st.subheader("Final MMR Distribution")
        if is_fast_mode:
            final_mmrs = sim.mmr
        else:
            final_mmrs = [u.current_mmr for u in sim.users]
        fig_dist = px.histogram(final_mmrs, nbins=50, title="MMR Distribution", labels={'value': 'MMR'})
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.subheader("Segment Performance")
        if is_fast_mode:
            # Stratified Sampling to ensure all segments are represented
            sample_size = min(10000, sim.num_users)
            if sample_size < sim.num_users:
                indices = np.random.choice(sim.num_users, sample_size, replace=False)
            else:
                indices = np.arange(sim.num_users)
                
            segment_data = {
                "Segment": [sim.seg_names[sim.segment_indices[i]] for i in indices],
                "MMR": sim.mmr[indices],
                "True Skill": sim.true_skill[indices],
                "Matches": sim.matches_played[indices],
                "Win Rate": sim.wins[indices] / (sim.matches_played[indices] + 1e-9)
            }
            df_users = pd.DataFrame(segment_data)
        else:
            segment_data = []
            for u in sim.users:
                segment_data.append({
                    "ID": u.id,
                    "Segment": u.segment_name,
                    "MMR": u.current_mmr,
                    "True Skill": u.true_skill,
                    "Matches": u.matches_played,
                    "Win Rate": u.win_rate
                })
            df_users = pd.DataFrame(segment_data)
            
        # Segment Filter
        all_segments = sorted(df_users["Segment"].unique())
        selected_segments = st.multiselect("Filter Segments", all_segments, default=all_segments)
        
        if selected_segments:
            df_users = df_users[df_users["Segment"].isin(selected_segments)]
        
        col1, col2 = st.columns(2)
        with col1:
            fig_box = px.box(df_users, x="Segment", y="MMR", title="MMR Distribution by Segment")
            st.plotly_chart(fig_box, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(df_users, x="True Skill", y="MMR", color="Segment", title="True Skill vs MMR", opacity=0.5)
            fig_scatter.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=2500, y0=0, y1=2500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        if not is_fast_mode:
            st.dataframe(df_users)

    with tab3:
        match_logic_help = """
승/무/패 결정 로직 설명

시뮬레이션 내부(_process_matches)에서 승패는 다음과 같은 확률적 절차로 결정.

1. 승률 계산 (True Skill 기반)
두 유저의 True Skill(실제 실력) 차이를 기반으로 A가 이길 확률(prob_a_win)을 계산
공식: 1 / (1 + 10^((Skill_B - Skill_A) / 400))
실력이 같으면 50%, A가 400점 높으면 약 91%

2. 무승부 판정 (정규 시간)
먼저 0~1 사이의 난수(rand)를 생성.
설정된 **무승부 확률(draw_prob, 예: 10%)**보다 난수가 작으면 일단 **무승부(Draw)**로 간주.
rand < 0.1 -> 무승부 상황 진입

3. 승패 판정 (정규 시간)
무승부가 아니라면, 남은 확률(90%)을 prob_a_win 비율로 나눠 갖음.
Win Threshold = 0.1 + (0.9 * prob_a_win)
rand < Win Threshold -> A 승리
그 외 -> A 패배 (B 승리)

4. 연장전 및 승부차기 (옵션)
무승부 상황에서 추가 난수를 굴려 연장전(prob_extra_time) 여부를 결정.
연장전에서도 승부가 안 나면 **승부차기(prob_pk)**로 넘어감.
승부차기는 실력 차이 없이 50:50 운으로 결정.

요약: 기본적으로 실력 차이가 승률을 지배하지만, 무승부 변수와 **운(Randomness)**이 개입하여 이변을 만들어냄.
"""
        st.subheader("User Match Inspector", help=match_logic_help)
        
        if is_fast_mode:
            st.info("Fast Mode Active: Viewing Sampled Users per Segment")
            
            # Create a mapping for dropdown that handles duplicate names
            # Format: "SegmentName (ID: 123)"
            dropdown_options = {}
            for idx, name in sim.watched_indices.items():
                label = f"{name} (ID: {idx})"
                dropdown_options[label] = idx
                
            selected_label = st.selectbox("Select Segment Sample to View", list(dropdown_options.keys()))
            
            if selected_label:
                target_idx = dropdown_options[selected_label]
                # Extract name from label or look it up
                selected_segment = sim.watched_indices[target_idx]
                
                st.write(f"**Sample User ID: {target_idx} ({selected_segment})**")
                st.write(f"Current MMR: {sim.mmr[target_idx]:.2f} | True Skill: {sim.true_skill[target_idx]:.2f}")
                st.write(f"Record: {sim.wins[target_idx]}W - {sim.draws[target_idx]}D - {sim.losses[target_idx]}L")
                
                logs = sim.match_logs.get(target_idx, [])
                if logs:
                    log_data = []
                    for log in logs:
                        log_data.append({
                            "Day": log.day,
                            "Opponent ID": log.opponent_id,
                            "Opponent MMR": f"{log.opponent_mmr:.1f}",
                            "Opponent Skill": f"{log.opponent_true_skill:.1f}",
                            "Result": f"{log.result} ({log.result_type})",
                            "Goal Diff": log.goal_diff,
                            "Change": f"{log.mmr_change:+.1f}",
                            "New MMR": f"{log.current_mmr:.1f}"
                        })
                    st.dataframe(pd.DataFrame(log_data))
                else:
                    st.info("No matches played yet.")
        else:
            user_id_input = st.number_input("Enter User ID to View Logs", min_value=0, max_value=num_users-1, value=0, step=1)
            target_user = next((u for u in sim.users if u.id == user_id_input), None)
            
            if target_user:
                st.write(f"**User {target_user.id} ({target_user.segment_name})**")
                st.write(f"Current MMR: {target_user.current_mmr:.2f} | True Skill: {target_user.true_skill:.2f}")
                st.write(f"Record: {target_user.wins}W - {target_user.draws}D - {target_user.losses}L (Win Rate: {target_user.win_rate:.2%})")
                
                if target_user.match_history:
                    log_data = []
                    for log in target_user.match_history:
                        log_data.append({
                            "Day": log.day,
                            "Hour": log.hour,
                            "Opponent ID": log.opponent_id,
                            "Opponent MMR": f"{log.opponent_mmr:.1f}",
                            "Opponent Skill": f"{log.opponent_true_skill:.1f}",
                            "Result": f"{log.result} ({log.result_type})",
                            "Goal Diff": log.goal_diff,
                            "Change": f"{log.mmr_change:+.1f}",
                            "New MMR": f"{log.current_mmr:.1f}"
                        })
                    st.dataframe(pd.DataFrame(log_data))
                else:
                    st.info("No matches played yet.")
            else:
                st.error("User not found.")
