import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from simulation_core import Simulation, FastSimulation, SegmentConfig, ELOConfig, MatchConfig
import json
import os
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="FC Online Rank Simulation", layout="wide")

# --- Configuration Persistence ---
CONFIG_FILE = "sim_config.json"

def load_config():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Config") # Explicitly read 'Config' worksheet if possible, or default
        
        if df.empty:
            return {}
            
        # Expecting ConfigJSON in the first cell/column
        if "ConfigJSON" in df.columns and len(df) > 0:
            json_str = df.iloc[0]["ConfigJSON"]
            return json.loads(json_str)
        # Fallback: try reading first cell if column name doesn't match
        if len(df) > 0 and len(df.columns) > 0:
             # This is a bit risky without strict schema, but let's try
             pass
             
        return {}
    except Exception as e:
        # Fallback to local
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
        df_to_save = pd.DataFrame([{"ConfigJSON": json_str}])
        conn.update(data=df_to_save) # Updates the default/first sheet
    except Exception as e:
        # Fallback to local
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except:
            pass

# --- Authentication ---
def check_password(username, password):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Try to read 'Users' worksheet
        df = pd.DataFrame()
        try:
            df = conn.read(worksheet="Users", ttl=0)
        except Exception:
            # Fallback: Try lowercase 'users'
            try:
                df = conn.read(worksheet="users", ttl=0)
            except Exception as e:
                st.error(f"Error: Could not find a worksheet named 'Users' or 'users'. Please check the tab name in your Google Sheet. (Details: {e})")
                return False
        
        if df.empty:
            st.error("Debug: Users sheet is empty.")
            return False
            
        # Normalize columns: strip whitespace and lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        if 'username' not in df.columns or 'password' not in df.columns:
            st.error(f"Debug: Missing 'username' or 'password' columns. Found: {df.columns.tolist()}")
            return False

        # Normalize data for comparison
        # Convert username column to string, strip whitespace
        df['username'] = df['username'].astype(str).str.strip()
        
        # Find user
        user_row = df[df['username'] == username.strip()]
        
        if not user_row.empty:
            # Handle password: convert to string, remove potential '.0' if it was read as float
            stored_password = str(user_row.iloc[0]['password']).strip()
            if stored_password.endswith('.0'):
                stored_password = stored_password[:-2]
                
            if stored_password == password.strip():
                return True
            else:
                # Debug: Password mismatch
                # st.warning(f"Debug: Password mismatch for {username}. Stored: '{stored_password}', Input: '{password}'")
                pass
        else:
             # Debug: User not found
             # st.warning(f"Debug: User '{username}' not found in sheet.")
             pass
             
        return False
    except Exception as e:
        st.error(f"Login Error: {e}")
        return False

def login_page():
    st.title("Login")
    st.info("Please log in to access the simulation.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_password(username, password):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    st.session_state["authenticated"] = False
    st.rerun()

# --- Main Execution Flow ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    # Sidebar Logout
    with st.sidebar:
        st.write(f"Logged in.")
        if st.button("Logout"):
            logout()
        st.divider()

    # --- Existing App Logic (Restored) ---
    
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
                        st.session_state.segments = [] 
                elif k == 'reset_rules':
                    st.session_state.reset_rules = pd.DataFrame(v)
                elif k == 'streak_rules':
                    st.session_state.streak_rules = pd.DataFrame(v)
                elif k == 'goal_diff_rules':
                    st.session_state.goal_diff_rules = pd.DataFrame(v)
                else:
                    st.session_state[k] = v
            if "user_comments" in loaded_config:
                st.session_state.user_comments = loaded_config["user_comments"]
                
        st.session_state.config_loaded = True

    # Initialize Session State (Defaults)
    if 'segments' not in st.session_state:
        st.session_state.segments = [
            SegmentConfig("Super Champions", 0.0003, 0.95, 10, 30, 4.0, 5.0, 18, 2),
            SegmentConfig("Champions", 0.0027, 0.85, 5, 15, 3.0, 4.0, 18, 2),
            SegmentConfig("World Class", 0.15, 0.60, 3, 10, 1.0, 3.0, 19, 1),
            SegmentConfig("Professional", 0.30, 0.40, 2, 5, -1.0, 1.0, 19, 1),
            SegmentConfig("Semi-Pro", 0.35, 0.20, 1, 3, -3.0, -1.0, 20, 0),
            SegmentConfig("Amateur", 0.197, 0.10, 0, 2, -5.0, -3.0, 20, 0)
        ]

    if 'simulation' not in st.session_state:
        st.session_state.simulation = None

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Simulation Settings")
        
        if st.button("⚠️ Reset Config (Emergency)", help="Click this if you see errors. It will reset all settings to default."):
            # Factory Reset: Overwrite remote config with empty JSON to force defaults on reload
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df_to_save = pd.DataFrame([{"ConfigJSON": "{}"}])
                conn.update(data=df_to_save)
            except Exception as e:
                st.error(f"Failed to reset remote config: {e}")
            
            st.session_state.clear()
            st.rerun()
        st.divider()
        
        with st.expander("Global Settings", expanded=True):
            st.session_state.num_users = st.number_input("Number of Users", min_value=100, max_value=1000000, value=st.session_state.get("num_users", 1000), step=100, help="Total number of users in the simulation.")
            st.session_state.num_days = st.number_input("Simulation Days", min_value=1, max_value=3650, value=st.session_state.get("num_days", 365), help="Duration of the simulation in days.")
            st.session_state.initial_mmr = st.number_input("Initial MMR", value=st.session_state.get("initial_mmr", 1000.0), help="Starting MMR for all users.")

        with st.expander("Match Configuration"):
            st.session_state.draw_prob = st.slider("Draw Probability (Regular Time)", 0.0, 0.5, st.session_state.get("draw_prob", 0.1), help="Probability of a match ending in a draw after regular time.")
            st.session_state.prob_et = st.slider("Extra Time Probability (given Draw)", 0.0, 1.0, st.session_state.get("prob_et", 0.2), help="Probability of going to extra time if the match is a draw.")
            st.session_state.prob_pk = st.slider("Penalty Shootout Probability (given ET)", 0.0, 1.0, st.session_state.get("prob_pk", 0.5), help="Probability of going to penalties if extra time is also a draw.")
            st.session_state.max_goal_diff = st.slider("Max Goal Difference", 1, 10, st.session_state.get("max_goal_diff", 5), help="Maximum possible goal difference in a match.")
            st.session_state.matchmaking_jitter = st.number_input("Matchmaking Jitter (MMR)", value=st.session_state.get("matchmaking_jitter", 50.0), help="Randomness added to matchmaking to simulate imperfect pairing.")

        with st.expander("ELO System Config"):
            st.session_state.base_k = st.number_input("Base K-Factor", value=st.session_state.get("base_k", 32), help="Standard K-factor for ELO calculations.")
            
            st.subheader("Placement Matches")
            st.session_state.placement_matches = st.number_input("Number of Placement Matches", value=st.session_state.get("placement_matches", 10), help="Number of initial matches with boosted K-factor.")
            st.session_state.placement_bonus = st.number_input("Placement K-Factor Bonus Multiplier", value=st.session_state.get("placement_bonus", 4.0), help="Multiplier for K-factor during placement matches.")
            
            st.subheader("Streak Multipliers")
            if 'streak_rules' not in st.session_state:
                st.session_state.streak_rules = pd.DataFrame([
                    {"min_streak": 3, "bonus": 5.0},
                    {"min_streak": 5, "bonus": 10.0}
                ])
            # Defensive check: Ensure it's a DataFrame (in case loaded as list from JSON)
            if not isinstance(st.session_state.streak_rules, pd.DataFrame):
                st.session_state.streak_rules = pd.DataFrame(st.session_state.streak_rules)
            
            # Ensure columns exist if empty (Fix for TypeError on empty list)
            if st.session_state.streak_rules.empty and len(st.session_state.streak_rules.columns) == 0:
                st.session_state.streak_rules = pd.DataFrame(columns=["min_streak", "bonus"])
                
            try:
                st.session_state.streak_rules = st.data_editor(st.session_state.streak_rules, num_rows="dynamic", use_container_width=True, key="streak_editor")
            except Exception as e:
                st.error(f"Error displaying streak rules: {e}")
                st.session_state.streak_rules = pd.DataFrame(columns=["min_streak", "bonus"])
            
            st.subheader("Goal Difference Multipliers")
            if 'goal_diff_rules' not in st.session_state:
                st.session_state.goal_diff_rules = pd.DataFrame([
                    {"min_diff": 2, "bonus": 2.0},
                    {"min_diff": 4, "bonus": 5.0}
                ])
            # Defensive check
            if not isinstance(st.session_state.goal_diff_rules, pd.DataFrame):
                st.session_state.goal_diff_rules = pd.DataFrame(st.session_state.goal_diff_rules)
            
            # Ensure columns exist if empty
            if st.session_state.goal_diff_rules.empty and len(st.session_state.goal_diff_rules.columns) == 0:
                st.session_state.goal_diff_rules = pd.DataFrame(columns=["min_diff", "bonus"])
                
            try:
                st.session_state.goal_diff_rules = st.data_editor(st.session_state.goal_diff_rules, num_rows="dynamic", use_container_width=True, key="gd_editor")
            except Exception as e:
                st.error(f"Error displaying goal diff rules: {e}")
                st.session_state.goal_diff_rules = pd.DataFrame(columns=["min_diff", "bonus"])
            
            st.subheader("Win Type Decay")
            st.session_state.decay_et = st.slider("Extra Time Win Decay", 0.0, 1.0, st.session_state.get("decay_et", 0.8), help="Multiplier for ELO gain when winning in extra time.")
            st.session_state.decay_pk = st.slider("Penalty Win Decay", 0.0, 1.0, st.session_state.get("decay_pk", 0.6), help="Multiplier for ELO gain when winning in penalties.")

            st.subheader("MMR Compression Correction (Calibration)")
            st.session_state.calibration_enabled = st.checkbox("Enable Calibration Mode", value=st.session_state.get("calibration_enabled", False), help="If enabled, applies a K-factor bonus when MMR is compressed.")
            if st.session_state.calibration_enabled:
                st.session_state.calibration_k_bonus = st.number_input("Calibration K-Bonus Multiplier", value=st.session_state.get("calibration_k_bonus", 2.0), help="Multiplier for K-factor during calibration.")
                st.session_state.calibration_match_count = st.number_input("Calibration Match Count", value=st.session_state.get("calibration_match_count", 10), help="Number of matches to apply calibration bonus.")

        with st.expander("User Segments"):
            st.write("Define user segments and their properties.")
            segment_data = []
            for s in st.session_state.segments:
                segment_data.append({
                    "name": s.name,
                    "ratio": s.ratio,
                    "daily_play_prob": s.daily_play_prob,
                    "matches_per_day_min": s.matches_per_day_min,
                    "matches_per_day_max": s.matches_per_day_max,
                    "true_skill_min": s.true_skill_min,
                    "true_skill_max": s.true_skill_max,
                    "active_hour_start": s.active_hour_start,
                    "active_hour_end": s.active_hour_end
                })
            
            try:
                edited_segments = st.data_editor(pd.DataFrame(segment_data), num_rows="dynamic", use_container_width=True, key="segment_editor")
            except Exception as e:
                st.error(f"Error displaying segments: {e}")
                edited_segments = pd.DataFrame() # Fallback
            
            new_segments = []
            total_ratio = 0
            if not edited_segments.empty:
                for index, row in edited_segments.iterrows():
                    try:
                        s = SegmentConfig(
                            row["name"], float(row["ratio"]), float(row["daily_play_prob"]),
                            int(row["matches_per_day_min"]), int(row["matches_per_day_max"]),
                            float(row["true_skill_min"]), float(row["true_skill_max"]),
                            int(row["active_hour_start"]), int(row["active_hour_end"])
                        )
                        new_segments.append(s)
                        total_ratio += s.ratio
                    except:
                        pass
                st.session_state.segments = new_segments
            st.write(f"Total Ratio: {total_ratio:.4f}")

        with st.expander("Reset Rules (Season End)"):
            if 'reset_rules' not in st.session_state:
                st.session_state.reset_rules = pd.DataFrame(columns=["tier_name", "min_mmr", "reset_mmr", "soft_reset_ratio"])
            # Defensive check
            if not isinstance(st.session_state.reset_rules, pd.DataFrame):
                st.session_state.reset_rules = pd.DataFrame(st.session_state.reset_rules)
            
            # Ensure columns exist if empty
            if st.session_state.reset_rules.empty and len(st.session_state.reset_rules.columns) == 0:
                 st.session_state.reset_rules = pd.DataFrame(columns=["tier_name", "min_mmr", "reset_mmr", "soft_reset_ratio"])
                 
            try:
                st.session_state.reset_rules = st.data_editor(st.session_state.reset_rules, num_rows="dynamic", use_container_width=True, key="reset_editor")
            except Exception as e:
                st.error(f"Error displaying reset rules: {e}")
                st.session_state.reset_rules = pd.DataFrame(columns=["tier_name", "min_mmr", "reset_mmr", "soft_reset_ratio"])

        if st.button("Save Configuration"):
            save_config()
            st.success("Configuration saved!")

    # --- Main Content ---
    st.title("Rank Simulation Dashboard")

    tab1, tab2, tab3 = st.tabs(["Run Simulation", "Analysis", "Match Inspector"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Control Panel")
            if st.button("Start Simulation", type="primary"):
                with st.spinner("Initializing Simulation..."):
                    elo_config = ELOConfig(
                        base_k=st.session_state.base_k,
                        placement_matches=st.session_state.placement_matches,
                        placement_bonus=st.session_state.placement_bonus,
                        streak_rules=st.session_state.streak_rules.to_dict('records'),
                        goal_diff_rules=st.session_state.goal_diff_rules.to_dict('records'),
                        decay_et=st.session_state.decay_et,
                        decay_pk=st.session_state.decay_pk,
                        calibration_k_bonus=st.session_state.calibration_k_bonus,
                        calibration_enabled=st.session_state.calibration_enabled,
                        calibration_match_count=st.session_state.calibration_match_count
                    )
                    match_config = MatchConfig(
                        draw_prob=st.session_state.draw_prob,
                        prob_et=st.session_state.prob_et,
                        prob_pk=st.session_state.prob_pk,
                        max_goal_diff=st.session_state.max_goal_diff,
                        matchmaking_jitter=st.session_state.matchmaking_jitter
                    )
                    st.session_state.simulation = FastSimulation(
                        num_users=st.session_state.num_users,
                        num_days=st.session_state.num_days,
                        segments=st.session_state.segments,
                        elo_config=elo_config,
                        match_config=match_config,
                        initial_mmr=st.session_state.initial_mmr
                    )
                    
                # Run Simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                sim = st.session_state.simulation
                
                stats_history = []
                for day in range(st.session_state.num_days):
                    sim.run_day(day)
                    # Collect daily stats for plotting
                    # FastSimulation might not store history per user, so we aggregate
                    mmrs = sim.mmr
                    stats_history.append({
                        "day": day + 1,
                        "avg_mmr": np.mean(mmrs),
                        "min_mmr": np.min(mmrs),
                        "max_mmr": np.max(mmrs)
                    })
                    
                    progress = (day + 1) / st.session_state.num_days
                    progress_bar.progress(progress)
                    status_text.text(f"Simulating Day {day+1}/{st.session_state.num_days}...")
                
                status_text.text("Simulation Complete!")
                st.session_state.stats_history = stats_history
                st.success("Simulation finished successfully.")

        with col2:
            st.subheader("Real-time Stats (Last Day)")
            if st.session_state.simulation:
                sim = st.session_state.simulation
                total_matches = np.sum(sim.matches_played)
                avg_mmr = np.mean(sim.mmr)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Matches", f"{total_matches:,}")
                m2.metric("Average MMR", f"{avg_mmr:.2f}")
                m3.metric("Active Users", f"{sim.num_users:,}")
                
                fig = px.histogram(x=sim.mmr, nbins=50, title="Final MMR Distribution", labels={'x': 'MMR', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Simulation Analysis")
        if st.session_state.simulation and 'stats_history' in st.session_state:
            stats_history = st.session_state.stats_history
            
            # MMR Trends
            df_stats = pd.DataFrame(stats_history)
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['avg_mmr'], name='Average MMR'))
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['min_mmr'], name='Min MMR'))
            fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['max_mmr'], name='Max MMR'))
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Tier Distribution
            st.markdown("### Tier Distribution")
            tiers = [
                ("Bronze", 0, 1200), ("Silver", 1200, 1400), ("Gold", 1400, 1600),
                ("Platinum", 1600, 1800), ("Diamond", 1800, 2000), ("Master", 2000, 2400), ("Challenger", 2400, 5000)
            ]
            sim = st.session_state.simulation
            tier_counts = {name: 0 for name, _, _ in tiers}
            for mmr in sim.mmr:
                for name, low, high in tiers:
                    if low <= mmr < high:
                        tier_counts[name] += 1
                        break
            
            tier_df = pd.DataFrame(list(tier_counts.items()), columns=["Tier", "Count"])
            fig_tier = px.bar(tier_df, x="Tier", y="Count", title="User Count by Tier")
            st.plotly_chart(fig_tier, use_container_width=True)
            
        else:
            st.info("Run the simulation first to see analysis.")

    with tab3:
        st.subheader("Match Inspector")
        if st.session_state.simulation:
            sim = st.session_state.simulation
            # Fast Mode Logic
            st.info("Fast Mode Active: Viewing Sampled Users per Segment")
            dropdown_options = {}
            for idx, name in sim.watched_indices.items():
                label = f"{name} (ID: {idx})"
                dropdown_options[label] = idx
                
            selected_label = st.selectbox("Select Segment Sample to View", list(dropdown_options.keys()))
            
            if selected_label:
                target_idx = dropdown_options[selected_label]
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
                            "Result": f"{log.result} ({log.result_type})",
                            "Goal Diff": log.goal_diff,
                            "Change": f"{log.mmr_change:+.1f}",
                            "New MMR": f"{log.current_mmr:.1f}"
                        })
                    st.dataframe(pd.DataFrame(log_data))
                else:
                    st.info("No matches played yet.")
        else:
            st.info("Run the simulation first to inspect matches.")

    # --- Comments Section ---
    st.divider()
    st.subheader("User Comments / Feedback")
    comments = st.text_area("Leave your feedback here:", value=st.session_state.get("user_comments", ""), height=100)
    if st.button("Save Comments"):
        st.session_state.user_comments = comments
        save_config()
        st.success("Comments saved!")
