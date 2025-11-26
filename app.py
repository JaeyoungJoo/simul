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

CONFIG_FILE = "sim_config.json"

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
