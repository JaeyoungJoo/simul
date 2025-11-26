import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from simulation_core import Simulation, FastSimulation, SegmentConfig, ELOConfig, MatchConfig, CORE_VERSION
import json
import os
from streamlit_gsheets import GSheetsConnection
import extra_streamlit_components as stx
import time
import datetime

st.set_page_config(page_title="FC 온라인 랭크 시뮬레이션", layout="wide")

# --- Configuration Persistence ---
CONFIG_FILE = "sim_config.json"

def load_config():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Config", ttl=0) # Explicitly read 'Config' worksheet with no cache
        
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
        # Tier Config
        "tier_config": st.session_state.get("tier_config", []),
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
        conn.update(worksheet="Config", data=df_to_save) # Explicitly update 'Config' worksheet
    except Exception as e:
        # Fallback to local
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except:
            pass

# --- Authentication ---
# --- Authentication & Session Management ---
cookie_manager = stx.CookieManager()

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
                st.error(f"오류: 'Users' 또는 'users' 시트를 찾을 수 없습니다. 구글 시트의 탭 이름을 확인해주세요. (상세: {e})")
                return False
        
        if df.empty:
            st.error("디버그: Users 시트가 비어있습니다.")
            return False
            
        # Normalize columns: strip whitespace and lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        if 'username' not in df.columns or 'password' not in df.columns:
            st.error(f"디버그: 'username' 또는 'password' 컬럼이 없습니다. 발견된 컬럼: {df.columns.tolist()}")
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
                pass
        else:
             pass
             
        return False
    except Exception as e:
        st.error(f"로그인 오류: {e}")
        return False

def login_page():
    st.title("로그인")
    st.info("시뮬레이션을 이용하려면 로그인하세요.")
    
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submit = st.form_submit_button("로그인")
        
        if submit:
            if check_password(username, password):
                st.session_state["authenticated"] = True
                # Set cookies for persistence (expire in 1 day)
                expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
                cookie_manager.set("auth_user", username, expires_at=expires_at, key="set_auth_user")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 잘못되었습니다.")

def logout():
    st.session_state["authenticated"] = False
    # Only delete auth_user. last_activity becomes irrelevant without auth_user.
    cookie_manager.delete("auth_user", key="delete_auth_user")
    st.rerun()

# --- Main Execution Flow ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Check Cookies for Persistence
if not st.session_state["authenticated"]:
    auth_user = cookie_manager.get("auth_user")
    last_activity = cookie_manager.get("last_activity")
    
    if auth_user and last_activity:
        try:
            last_activity_time = float(last_activity)
            current_time = time.time()
            # 30 minutes = 1800 seconds
            if current_time - last_activity_time > 1800:
                # Timeout
                cookie_manager.delete("auth_user")
                cookie_manager.delete("last_activity")
                st.warning("30분 동안 활동이 없어 로그아웃 되었습니다.")
            else:
                # Restore Session
                st.session_state["authenticated"] = True
                # Update last activity
                expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
                cookie_manager.set("last_activity", str(current_time), expires_at=expires_at)
                st.rerun()
        except ValueError:
            pass

if not st.session_state["authenticated"]:
    login_page()
else:
    # Update last activity timestamp on every interaction
    current_time = time.time()
    expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
    # Only update if enough time passed to avoid excessive cookie writes? 
    # Streamlit reruns on interaction, so updating here keeps it alive.
    # To prevent infinite rerun loops, we don't call rerun() here, just set the cookie.
    # CookieManager.set might trigger a rerun if key changes, but we are updating value.
    # Let's check if we need to throttle. For now, simple update.
    # Actually, setting cookie might trigger rerun. Let's only update if > 1 minute passed since last check?
    # But we don't have easy access to 'last check' without reading cookie again.
    # Let's trust the manager or just update.
    # Optimization: Read cookie first, if diff < 60s, skip.
    try:
        last_activity_cookie = cookie_manager.get("last_activity")
        if last_activity_cookie:
            if current_time - float(last_activity_cookie) > 60:
                 cookie_manager.set("last_activity", str(current_time), expires_at=expires_at)
        else:
             cookie_manager.set("last_activity", str(current_time), expires_at=expires_at)
    except:
        pass

    # Sidebar Logout
    with st.sidebar:
        st.write(f"로그인됨.")
        if st.button("로그아웃"):
            logout()
        st.divider()

    # --- Existing App Logic (Restored) ---
    
    # Load Config at Startup
    if 'config_loaded' not in st.session_state:
        with st.spinner("데이터베이스에서 설정을 불러오는 중..."):
            loaded_config = load_config()
            
        if loaded_config:
            # Apply to session state for widgets
            for k, v in loaded_config.items():
                if k == 'segments':
                    try:
                        st.session_state.segments = [SegmentConfig(**s) for s in v]
                    except TypeError:
                        st.warning("세그먼트 설정 형식이 맞지 않아 기본값을 사용합니다.")
                        st.session_state.segments = [] 
                elif k == 'reset_rules':
                    st.session_state.reset_rules = pd.DataFrame(v)
                elif k == 'streak_rules':
                    st.session_state.streak_rules = pd.DataFrame(v)
                elif k == 'goal_diff_rules':
                    st.session_state.goal_diff_rules = pd.DataFrame(v)
                elif k == 'tier_config':
                    st.session_state.tier_config = v
                else:
                    st.session_state[k] = v
            if "user_comments" in loaded_config:
                st.session_state.user_comments = loaded_config["user_comments"]
                
        st.session_state.config_loaded = True

    # Initialize Session State (Defaults)
    if 'segments' not in st.session_state:
        st.session_state.segments = [
            SegmentConfig("Super Champions", 0.0003, 0.95, 10.0, 30.0, 4.0, 5.0, 18, 2),
            SegmentConfig("Champions", 0.0027, 0.85, 5.0, 15.0, 3.0, 4.0, 18, 2),
            SegmentConfig("World Class", 0.15, 0.60, 3.0, 10.0, 1.0, 3.0, 19, 1),
            SegmentConfig("Professional", 0.30, 0.40, 2.0, 5.0, -1.0, 1.0, 19, 1),
            SegmentConfig("Semi-Pro", 0.35, 0.20, 1.0, 3.0, -3.0, -1.0, 20, 0),
            SegmentConfig("Amateur", 0.197, 0.10, 0.0, 2.0, -5.0, -3.0, 20, 0)
        ]

    if 'simulation' not in st.session_state:
        st.session_state.simulation = None

    # --- Sidebar Configuration ---
    with st.sidebar:
        with st.expander("기본 설정 (Global Settings)", expanded=True):
            st.session_state.num_users = st.number_input("유저 수 (Number of Users)", min_value=100, max_value=1000000, value=st.session_state.get("num_users", 1000), step=100, help="시뮬레이션에 참여할 총 유저 수입니다.")
            st.session_state.num_days = st.number_input("시뮬레이션 기간 (일)", min_value=1, max_value=3650, value=st.session_state.get("num_days", 365), help="시뮬레이션을 진행할 총 기간(일)입니다.")
            st.session_state.initial_mmr = st.number_input("초기 MMR", value=st.session_state.get("initial_mmr", 1000.0), help="모든 유저의 시작 MMR 점수입니다.")

        with st.expander("매치 설정 (Match Configuration)"):
            st.session_state.draw_prob = st.slider("무승부 확률 (정규 시간)", 0.0, 0.5, st.session_state.get("draw_prob", 0.1), help="정규 시간 내에 무승부가 발생할 확률입니다.")
            st.session_state.prob_et = st.slider("연장전 확률 (무승부 시)", 0.0, 1.0, st.session_state.get("prob_et", 0.2), help="무승부 시 연장전으로 갈 확률입니다.")
            st.session_state.prob_pk = st.slider("승부차기 확률 (연장 무승부 시)", 0.0, 1.0, st.session_state.get("prob_pk", 0.5), help="연장전에서도 승부가 나지 않아 승부차기로 갈 확률입니다.")
            st.session_state.max_goal_diff = st.slider("최대 골 득실차", 1, 10, st.session_state.get("max_goal_diff", 5), help="경기에서 발생할 수 있는 최대 골 득실차입니다.")
            st.session_state.matchmaking_jitter = st.number_input("매칭 범위 (MMR Jitter)", value=st.session_state.get("matchmaking_jitter", 50.0), help="매칭 시 허용되는 MMR 차이 범위입니다.")

        with st.expander("ELO 시스템 설정"):
            st.session_state.base_k = st.number_input("기본 K-Factor", value=st.session_state.get("base_k", 32), help="ELO 계산에 사용되는 기본 K-Factor입니다.")
            st.session_state.max_k = st.number_input("최대 K-Factor", value=st.session_state.get("max_k", 64), help="K-Factor가 가질 수 있는 최대값입니다.")
            st.session_state.placement_k_factor = st.number_input("배치고사 K-Factor", value=st.session_state.get("placement_k_factor", 64), help="배치고사 기간 동안 적용되는 K-Factor입니다.")
            st.session_state.streak_bonus = st.number_input("연승 보너스", value=st.session_state.get("streak_bonus", 1.0), help="연승 시 추가되는 점수 보너스입니다.")
            st.session_state.streak_threshold = st.number_input("연승 기준", value=st.session_state.get("streak_threshold", 3), help="연승 보너스가 적용되기 시작하는 승리 횟수입니다.")
            st.session_state.gd_bonus_weight = st.number_input("골 득실 가중치", value=st.session_state.get("gd_bonus_weight", 1.0), help="골 득실 차이에 따른 점수 가중치입니다.")
            st.session_state.mmr_compression_correction = st.number_input("MMR 압축 보정", value=st.session_state.get("mmr_compression_correction", 0.0), help="MMR 압축 현상을 완화하기 위한 보정값입니다.")
            
            st.subheader("배치고사")
            st.session_state.placement_matches = st.number_input("배치고사 경기 수", value=st.session_state.get("placement_matches", 10), help="배치고사로 간주되는 초기 경기 수입니다.")
            st.session_state.placement_bonus = st.number_input("배치고사 K-Factor 보너스 배율", value=st.session_state.get("placement_bonus", 4.0), help="배치고사 기간 동안 적용되는 K-Factor 보너스 배율입니다.")
            
            st.subheader("연승/연패 보너스")
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
                st.error(f"연승 규칙 표시 오류: {e}")
                st.session_state.streak_rules = pd.DataFrame(columns=["min_streak", "bonus"])
            
            st.subheader("골 득실 보너스")
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
                st.error(f"골 득실 규칙 표시 오류: {e}")
                st.session_state.goal_diff_rules = pd.DataFrame(columns=["min_diff", "bonus"])
            
            st.subheader("승리 유형별 가중치 (Decay)")
            st.session_state.decay_et = st.slider("연장승 가중치", 0.0, 1.0, st.session_state.get("decay_et", 0.8), help="연장전 승리 시 획득 점수 비율입니다 (1.0 = 정상 점수).")
            st.session_state.decay_pk = st.slider("승부차기승 가중치", 0.0, 1.0, st.session_state.get("decay_pk", 0.6), help="승부차기 승리 시 획득 점수 비율입니다.")

            st.subheader("MMR 압축 보정 (Calibration)")
            st.subheader("MMR 압축 보정 (Calibration)")
            # st.session_state.calibration_enabled is now controlled in the main Control Panel
            if st.session_state.get("calibration_enabled", False):
                st.session_state.calibration_k_bonus = st.number_input("보정 K-Bonus 배율", value=st.session_state.get("calibration_k_bonus", 2.0), help="보정 모드 시 적용할 추가 K-Factor 배율입니다.")
                st.session_state.calibration_match_count = st.number_input("보정 적용 경기 수", value=st.session_state.get("calibration_match_count", 10), help="보정 모드가 적용되는 경기 수입니다.")
            else:
                st.caption("제어판에서 보정 모드를 활성화하면 추가 설정이 표시됩니다.")

        # --- Tier Configuration ---
        if 'tier_config' not in st.session_state:
            st.session_state.tier_config = [
                {"Tier": "Bronze", "Min": 0, "Max": 1200},
                {"Tier": "Silver", "Min": 1200, "Max": 1400},
                {"Tier": "Gold", "Min": 1400, "Max": 1600},
                {"Tier": "Platinum", "Min": 1600, "Max": 1800},
                {"Tier": "Diamond", "Min": 1800, "Max": 2000},
                {"Tier": "Master", "Min": 2000, "Max": 2400},
                {"Tier": "Challenger", "Min": 2400, "Max": 5000}
            ]

        with st.expander("티어 기준 설정 (Tier Config)"):
            st.caption("티어 이름과 점수 구간을 설정하세요.")
            st.session_state.tier_config = st.data_editor(
                st.session_state.tier_config,
                column_config={
                    "Tier": st.column_config.TextColumn("티어 이름 (Name)", required=True),
                    "Min": st.column_config.NumberColumn("최소 점수 (Min)", required=True, step=10),
                    "Max": st.column_config.NumberColumn("최대 점수 (Max)", required=True, step=10),
                },
                num_rows="dynamic",
                hide_index=True,
                key="tier_editor"
            )

        with st.expander("유저 세그먼트 (티어/실력 분포)"):
            st.write("유저 세그먼트 및 특성을 정의하세요.")
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
                # Ensure numeric columns are float to allow decimal input in editor
                df_segments = pd.DataFrame(segment_data)
                if not df_segments.empty:
                    cols_to_float = ["matches_per_day_min", "matches_per_day_max", "true_skill_min", "true_skill_max", "daily_play_prob", "ratio"]
                    for col in cols_to_float:
                        if col in df_segments.columns:
                            df_segments[col] = df_segments[col].astype(float)
                            
                edited_segments = st.data_editor(df_segments, num_rows="dynamic", use_container_width=True, key="segment_editor")
            except Exception as e:
                st.error(f"세그먼트 표시 오류: {e}")
                edited_segments = pd.DataFrame() # Fallback
            
            new_segments = []
            total_ratio = 0
            if not edited_segments.empty:
                for index, row in edited_segments.iterrows():
                    try:
                        s = SegmentConfig(
                            row["name"], float(row["ratio"]), float(row["daily_play_prob"]),
                            float(row["matches_per_day_min"]), float(row["matches_per_day_max"]),
                            float(row["true_skill_min"]), float(row["true_skill_max"]),
                            int(row["active_hour_start"]), int(row["active_hour_end"])
                        )
                        new_segments.append(s)
                        total_ratio += s.ratio
                    except:
                        pass
                st.session_state.segments = new_segments
            st.write(f"총 비율 합계: {total_ratio:.4f}")

        with st.expander("시즌 초기화 규칙 (Season End)"):
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
                st.error(f"초기화 규칙 표시 오류: {e}")
                st.session_state.reset_rules = pd.DataFrame(columns=["tier_name", "min_mmr", "reset_mmr", "soft_reset_ratio"])

        if st.button("설정 저장"):
            save_config()
            st.success("설정이 저장되었습니다!")

    # --- Main Content ---
    st.title("Rank simulation")

    tab1, tab2, tab3 = st.tabs(["시뮬레이션 실행", "분석", "매치 기록"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("제어판")
            st.checkbox("보정 모드 활성화 (Calibration)", key="calibration_enabled", help="MMR 압축 현상을 완화하기 위한 보정 모드를 활성화합니다.")
            hard_reset = st.checkbox("시즌 1부터 시작 (초기화)", value=True, help="체크 시 모든 데이터를 초기화하고 1일차부터 시작합니다. 해제 시 현재 상태를 유지하며 이어서 진행합니다.")
            
            if st.button("시뮬레이션 시작", type="primary"):
                with st.spinner("시뮬레이션 준비 중..."):
                    win_type_decay = {'Regular': 1.0, 'Extra': st.session_state.decay_et, 'PK': st.session_state.decay_pk}
                    
                    elo_config = ELOConfig(
                        base_k=st.session_state.base_k,
                        placement_matches=st.session_state.placement_matches,
                        placement_bonus=st.session_state.placement_bonus,
                        streak_rules=st.session_state.streak_rules.to_dict('records') if isinstance(st.session_state.streak_rules, pd.DataFrame) else st.session_state.streak_rules,
                        goal_diff_rules=st.session_state.goal_diff_rules.to_dict('records') if isinstance(st.session_state.goal_diff_rules, pd.DataFrame) else st.session_state.goal_diff_rules,
                        win_type_decay=win_type_decay,
                        calibration_enabled=st.session_state.get("calibration_enabled", False),
                        calibration_k_bonus=st.session_state.get("calibration_k_bonus", 2.0),
                        calibration_match_count=st.session_state.get("calibration_match_count", 10)
                    )
                    
                    match_config = MatchConfig(
                        basic_draw_prob=st.session_state.basic_draw_prob,
                        min_draw_prob=st.session_state.min_draw_prob,
                        max_draw_prob=st.session_state.max_draw_prob,
                        home_advantage=st.session_state.home_advantage
                    )
                    
                    # Initialize or Update Simulation
                    if hard_reset or st.session_state.simulation is None:
                        st.session_state.simulation = FastSimulation(
                            num_users=st.session_state.num_users,
                            segment_configs=st.session_state.segments,
                            elo_config=elo_config,
                            match_config=match_config,
                            initial_mmr=st.session_state.initial_mmr
                        )
                        st.session_state.stats_history = []
                        st.success(f"시뮬레이션이 초기화되었습니다. (Day 0)")
                    else:
                        # Update existing simulation configs
                        st.session_state.simulation.elo_config = elo_config
                        st.session_state.simulation.match_config = match_config
                        st.session_state.simulation.segment_configs = st.session_state.segments
                        st.success(f"시뮬레이션 설정이 업데이트되었습니다. (Day {st.session_state.simulation.day}부터 계속)")
                    
                    st.session_state.simulation.initialize_users()
                    
                # Run Simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                sim = st.session_state.simulation
                
                stats_history = []
                if not hard_reset and 'stats_history' in st.session_state and st.session_state.stats_history:
                     stats_history = st.session_state.stats_history
                
                for day in range(st.session_state.num_days):
                    sim.run_day()
                    # Collect daily stats for plotting
                    mmrs = sim.mmr
                    stats_history.append({
                        "day": sim.day,
                        "avg_mmr": np.mean(mmrs),
                        "min_mmr": np.min(mmrs),
                        "max_mmr": np.max(mmrs)
                    })
                    
                    progress = (day + 1) / st.session_state.num_days
                    progress_bar.progress(progress)
                    status_text.text(f"시뮬레이션 진행 중: Day {sim.day} (진행률: {int(progress*100)}%)...")
                
                status_text.text("시뮬레이션 완료!")
                st.session_state.stats_history = stats_history
                st.success("시뮬레이션이 성공적으로 종료되었습니다.")

            if st.session_state.simulation is not None:
                st.divider()
                st.subheader("시즌 관리")
                if st.button("시즌 초기화 (Soft Reset)"):
                        try:
                            # Ensure numeric types for sorting and processing
                            rules_df = st.session_state.reset_rules.copy()
                            cols_to_numeric = ["min_mmr", "reset_mmr", "soft_reset_ratio"]
                            for col in cols_to_numeric:
                                rules_df[col] = pd.to_numeric(rules_df[col], errors='coerce')
                            
                            # Drop rows with invalid numbers if any
                            rules_df = rules_df.dropna(subset=cols_to_numeric)
                            
                            rules_df = rules_df.sort_values("min_mmr")
                            
                            rules = []
                            for i in range(len(rules_df)):
                                row = rules_df.iloc[i]
                                min_val = float(row["min_mmr"])
                                if i < len(rules_df) - 1:
                                    max_val = float(rules_df.iloc[i+1]["min_mmr"])
                                else:
                                    max_val = float('inf')
                                
                                rules.append({
                                    "min": min_val,
                                    "max": max_val,
                                    "target": float(row["reset_mmr"]),
                                    "compression": float(row["soft_reset_ratio"])
                                })
                            
                            st.session_state.simulation.apply_tiered_reset(rules)
                            
                            # Day reset is now handled by re-running simulation with hard_reset=True if desired.
                            # Soft reset just compresses MMR.
                                
                            st.success("시즌 초기화가 완료되었습니다. (MMR 압축 및 전적 초기화)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"초기화 중 오류 발생: {e}")

        with col2:
            st.subheader("실시간 통계 (마지막 날)")
            if st.session_state.simulation:
                sim = st.session_state.simulation
                total_matches = np.sum(sim.matches_played)
                avg_mmr = np.mean(sim.mmr)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("총 경기 수", f"{total_matches:,}")
                m2.metric("평균 MMR", f"{avg_mmr:.2f}")
                m3.metric("활성 유저", f"{sim.num_users:,}")
                
                fig = px.histogram(x=sim.mmr, nbins=50, title="최종 MMR 분포", labels={'x': 'MMR', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("시뮬레이션 분석")
        if st.session_state.simulation and 'stats_history' in st.session_state:
            stats_history = st.session_state.stats_history
            
            if not stats_history:
                st.info("시뮬레이션 데이터가 없습니다. 시뮬레이션을 실행해주세요.")
            else:
                # MMR Trends
                df_stats = pd.DataFrame(stats_history)
                fig_trends = go.Figure()
                fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['avg_mmr'], name='평균 MMR'))
                fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['min_mmr'], name='최소 MMR'))
                fig_trends.add_trace(go.Scatter(x=df_stats['day'], y=df_stats['max_mmr'], name='최대 MMR'))
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Tier Distribution
                st.markdown("### 티어 분포")
                
                # Use dynamic tier config
                raw_tiers = st.session_state.tier_config
                # Sort by Min score to ensure correct order
                sorted_tiers = sorted(raw_tiers, key=lambda x: x['Min'])
                
                tiers = [(t['Tier'], t['Min'], t['Max']) for t in sorted_tiers]
                
                sim = st.session_state.simulation
                tier_counts = {name: 0 for name, _, _ in tiers}
                for mmr in sim.mmr:
                    for name, low, high in tiers:
                        if low <= mmr < high:
                            tier_counts[name] += 1
                            break
            
                tier_df = pd.DataFrame(list(tier_counts.items()), columns=["Tier", "Count"])
                fig_tier = px.bar(tier_df, x="Tier", y="Count", title="티어별 유저 수")
                st.plotly_chart(fig_tier, use_container_width=True)

                # Segment Performance Analysis
                st.markdown("### 세그먼트 성과 분석 (Segment Performance)")
                
                # Prepare Data
                sim = st.session_state.simulation
                seg_names_map = {i: name for i, name in enumerate(sim.seg_names)}
                user_seg_names = [seg_names_map[i] for i in sim.segment_indices]
                
                df_users = pd.DataFrame({
                    "MMR": sim.mmr,
                    "True Skill": sim.true_skill,
                    "Segment": user_seg_names
                })
                
                # Filter
                all_segments = list(sim.seg_names)
                selected_segments = st.multiselect("세그먼트 필터 (Filter Segments)", all_segments, default=all_segments)
                
                if selected_segments:
                    filtered_df = df_users[df_users["Segment"].isin(selected_segments)]
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**세그먼트별 MMR 분포**")
                        fig_box = px.box(filtered_df, x="Segment", y="MMR", color="Segment", 
                                         title="MMR Distribution by Segment")
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        
                    with col_b:
                        st.markdown("**실력(True Skill) vs MMR**")
                        # Sample if too large for scatter
                        if len(filtered_df) > 5000:
                            plot_df = filtered_df.sample(5000)
                        else:
                            plot_df = filtered_df
                        
                        fig_scatter = px.scatter(plot_df, x="True Skill", y="MMR", color="Segment",
                                                 title="True Skill vs MMR", hover_data=["Segment"])
                        # Add identity line
                        fig_scatter.add_shape(type="line", line=dict(dash="dash", color="gray"),
                            x0=plot_df["True Skill"].min(), y0=plot_df["True Skill"].min(),
                            x1=plot_df["True Skill"].max(), y1=plot_df["True Skill"].max())
                        st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("분석할 세그먼트를 선택하세요.")
            
        else:
            st.info("분석을 보려면 먼저 시뮬레이션을 실행하세요.")

    with tab3:
        st.subheader("매치 기록")
        if st.session_state.simulation:
            sim = st.session_state.simulation
            # Fast Mode Logic
            st.info("고속 모드 활성: 세그먼트별 샘플 유저 보기")
            dropdown_options = {}
            for idx, name in sim.watched_indices.items():
                label = f"{name} (ID: {idx})"
                dropdown_options[label] = idx
                
            selected_label = st.selectbox("확인할 샘플 유저 선택", list(dropdown_options.keys()))
            
            if selected_label:
                target_idx = dropdown_options[selected_label]
                selected_segment = sim.watched_indices[target_idx]
                
                st.write(f"**샘플 유저 ID: {target_idx} ({selected_segment})**")
                st.write(f"현재 MMR: {sim.mmr[target_idx]:.2f} | 실력(True Skill): {sim.true_skill[target_idx]:.2f}")
                st.write(f"전적: {sim.wins[target_idx]}승 - {sim.draws[target_idx]}무 - {sim.losses[target_idx]}패")
                
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
                    st.info("진행된 경기가 없습니다.")
        else:
            st.info("매치 기록을 보려면 먼저 시뮬레이션을 실행하세요.")

    # --- Comments Section ---
    st.divider()
    st.subheader("Comment")
    comments = st.text_area("메모용:", value=st.session_state.get("user_comments", ""), height=500)
    if st.button("Save"):
        st.session_state.user_comments = comments
        save_config()
        st.success("코멘트가 저장되었습니다!")
