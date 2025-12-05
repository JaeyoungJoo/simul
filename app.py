import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from simulation_core import Simulation, FastSimulation, SegmentConfig, ELOConfig, MatchConfig, TierConfig, TierType, CORE_VERSION
import json
import os
from streamlit_gsheets import GSheetsConnection
import extra_streamlit_components as stx
import time
import datetime
import io


def safe_create_tier_config(**kwargs):
    try:
        return TierConfig(**kwargs)
    except TypeError:
        # Fallback for legacy class definition
        # Create with minimal args and set attributes manually
        try:
            # Try with fields that likely exist
            t = TierConfig(name=kwargs["name"], type=kwargs["type"], 
                           min_mmr=kwargs.get("min_mmr", 0), max_mmr=kwargs.get("max_mmr", 9999))
        except TypeError:
             # Fallback to absolute minimum
             t = TierConfig(name=kwargs["name"], type=kwargs["type"])
        
        # Set all provided attributes
        for k, v in kwargs.items():
            setattr(t, k, v)
        return t

def render_bulk_csv_uploader(label, current_df, key_suffix, header_mapping=None, use_expander=True):
    container = None
    if use_expander:
        container = st.expander(f"{label} - CSV 일괄 입력 (Bulk Input)")
    else:
        if st.checkbox(f"CSV 일괄 입력 열기 ({label})", key=f"check_{key_suffix}"):
            container = st.container()
            
    if container:
        with container:
            st.caption("엑셀이나 CSV 데이터를 복사해서 붙여넣으세요. (첫 줄은 헤더여야 합니다)")
            csv_input = st.text_area(f"CSV 데이터 붙여넣기 ({label})", key=f"csv_input_{key_suffix}")
            if st.button(f"CSV 적용 ({label})", key=f"csv_apply_{key_suffix}"):
                if csv_input:
                    try:
                        # 1. Try reading with Python engine (auto-detect)
                        try:
                            new_df = pd.read_csv(io.StringIO(csv_input), sep=None, engine='python')
                        except:
                            # Fallback to default
                            new_df = pd.read_csv(io.StringIO(csv_input))
                        
                        # 2. Check if it looks like tab-separated but read as comma (1 column)
                        if len(new_df.columns) == 1 and '\t' in csv_input:
                             try:
                                new_df = pd.read_csv(io.StringIO(csv_input), sep='\t')
                             except:
                                pass
    
                        # 3. Apply Header Mapping (Korean Label -> English Key)
                        if header_mapping:
                            # Normalize columns: strip whitespace
                            new_df.columns = new_df.columns.str.strip()
                            
                            # Create a reverse mapping check
                            renamed_cols = {}
                            for col in new_df.columns:
                                if col in header_mapping:
                                    renamed_cols[col] = header_mapping[col]
                            
                            if renamed_cols:
                                new_df = new_df.rename(columns=renamed_cols)
    
                        # 4. Validation
                        if not current_df.empty:
                            # Check for missing required columns (intersection with current_df columns)
                            # We only care if *required* columns are missing, but we don't know which are required here.
                            # Just check overlap with current_df keys.
                            expected_cols = set(current_df.columns)
                            found_cols = set(new_df.columns)
                            missing_cols = list(expected_cols - found_cols)
                            
                            # Filter out missing cols that might be optional or auto-generated if needed?
                            # For now, just warn if high overlap is expected.
                            if missing_cols and len(missing_cols) < len(expected_cols): # If some match but not all
                                 st.warning(f"주의: 다음 컬럼을 찾을 수 없습니다: {missing_cols}. (헤더 이름을 확인하세요)")
                            elif len(missing_cols) == len(expected_cols):
                                 st.error(f"오류: 일치하는 컬럼이 없습니다. 헤더가 올바른지 확인하세요.\n기대하는 컬럼(또는 한글명): {list(header_mapping.keys()) if header_mapping else list(expected_cols)}")
                                 return None
    
                        return new_df
                    except Exception as e:
                        st.error(f"CSV 파싱 오류: {e}")
                else:
                    st.warning("데이터를 입력하세요.")
    return None

st.set_page_config(page_title="Rank Simulation", layout="wide")

# --- Configuration Persistence ---
CONFIG_FILE = "sim_config.json"
def load_config(current_username=None):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        # Read Config with fallback
        df = pd.DataFrame()
        try:
            df = conn.read(worksheet="Simul_Config", ttl=0)
        except:
            try:
                df = conn.read(worksheet="simul_config", ttl=0)
            except:
                try:
                    df = conn.read(worksheet="Config", ttl=0)
                except:
                    try:
                        df = conn.read(worksheet="config", ttl=0)
                    except Exception as e:
                        st.error(f"설정 시트 로드 실패 (Simul_Config, simul_config, Config, config 모두 실패): {e}")
                        return {}
        
        if df.empty:
            return {}
            
        # Normalize columns
        df.columns = [str(c).strip() for c in df.columns]
        
        # Debug: Show loaded dataframe info
        # st.write("Debug: Config Sheet Columns:", df.columns.tolist())
        # st.write("Debug: Config Sheet Data (Head):", df.head())
        
        target_config_json = None
        
        # 1. Try to find config for current user
        if current_username and "username" in df.columns and "ConfigJSON" in df.columns:
            # Normalize for comparison
            df["username_norm"] = df["username"].astype(str).str.strip().str.lower()
            target_user_norm = str(current_username).strip().lower()
            
            user_config = df[df["username_norm"] == target_user_norm]
            if not user_config.empty:
                target_config_json = user_config.iloc[0]["ConfigJSON"]
        
            if not user_config.empty:
                target_config_json = user_config.iloc[0]["ConfigJSON"]
                
                # Load comment from column if available
                if "Comment" in user_config.columns:
                    comment_val = user_config.iloc[0]["Comment"]
                    if pd.notna(comment_val):
                        # We will inject this into the loaded config later
                        pass
        
        # 2. If no user config (or not logged in), try to find Default (Admin) config
        if not target_config_json:
            # We need to find who is admin. This requires reading Users sheet again.
            # Optimization: If we stored admin list in session state? No, let's read it to be safe/fresh.
            try:
                users_df = conn.read(worksheet="Users", ttl=0)
                users_df.columns = [str(c).strip().lower() for c in users_df.columns]
                
                if 'admin' in users_df.columns and 'username' in users_df.columns:
                    # Filter for admins
                    # Handle various truthy values
                    is_admin_mask = users_df['admin'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])
                    admin_users = users_df[is_admin_mask]['username'].tolist()
                    
                    if admin_users:
                        # Find config for the first admin
                        # We need to match username in Config sheet
                        if "username" in df.columns and "ConfigJSON" in df.columns:
                            # Filter Config df for rows where username is in admin_users
                            admin_configs = df[df["username"].isin(admin_users)]
                            
                            if not admin_configs.empty:
                                # Pick the first one found (as per requirement)
                                target_config_json = admin_configs.iloc[0]["ConfigJSON"]
            except Exception as e:
                st.error(f"관리자 설정 로드 실패: {e}")
                pass

        # Debug Output
        if target_config_json:
            st.success(f"설정 로드 성공: {current_username if current_username else 'Admin/Default'}")
        else:
            st.error(f"설정 로드 실패: {current_username} (기본값 사용)")
            # st.write(f"Debug Info: User={current_username}, Columns={df.columns.tolist()}")
        
        # 3. Legacy Fallback: If no username column, or just one row exists (old format)
        if not target_config_json:
             if "ConfigJSON" in df.columns and len(df) > 0:
                 # If there is no username column, assume single config mode
                 if "username" not in df.columns:
                     target_config_json = df.iloc[0]["ConfigJSON"]
        
        if target_config_json:
            config_dict = json.loads(target_config_json)
            
            # --- Merge Split Configs (Tier & Segment) ---
            # If TierConfigJSON exists in the row, load and merge/override
            if current_username and "username" in df.columns:
                 # Re-find user row (optimization: we already have user_config df from above if we restructuring)
                 # But let's reuse logic for safety
                df["username_norm"] = df["username"].astype(str).str.strip().str.lower()
                target_user_norm = str(current_username).strip().lower()
                user_config = df[df["username_norm"] == target_user_norm]
                
                if not user_config.empty:
                    row = user_config.iloc[0]
                    # Merge Tier Config
                    if "TierConfigJSON" in row and pd.notna(row["TierConfigJSON"]) and str(row["TierConfigJSON"]).strip():
                         try:
                             tier_c = json.loads(str(row["TierConfigJSON"]))
                             if tier_c:
                                 config_dict["tier_config"] = tier_c
                         except: pass
                    
                    # Merge Segment Config
                    if "SegmentConfigJSON" in row and pd.notna(row["SegmentConfigJSON"]) and str(row["SegmentConfigJSON"]).strip():
                         try:
                             seg_c = json.loads(str(row["SegmentConfigJSON"]))
                             if seg_c:
                                 config_dict["segments"] = seg_c
                         except: pass

                    # Inject Comment
                    if "Comment" in row and pd.notna(row["Comment"]):
                        config_dict["user_comments"] = str(row["Comment"])
            
            return config_dict
            
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

def save_config(current_username=None):
    if not current_username:
        # If no username provided (e.g. not logged in), maybe don't save or save to local?
        # For now, let's require username for cloud save.
        return

    config = {
        # Global
        "num_users": st.session_state.get("num_users", 1000),
        "num_days": st.session_state.get("num_days", 365),
        "initial_mmr": st.session_state.get("initial_mmr", 1000.0),
        "use_true_skill_init": st.session_state.get("use_true_skill_init", False),
        # Match Config
        "draw_prob": st.session_state.get("draw_prob", 0.1),
        "prob_et": st.session_state.get("prob_et", 0.2),
        "prob_pk": st.session_state.get("prob_pk", 0.5),
        "max_goal_diff": st.session_state.get("max_goal_diff", 5),
        "matchmaking_jitter": st.session_state.get("matchmaking_jitter", 50.0),
        "bot_win_rate": st.session_state.get("bot_win_rate", 0.8),
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
        "mmr_compression_correction": st.session_state.get("mmr_compression_correction", 0.0),
        "gd_bonus_weight": st.session_state.get("gd_bonus_weight", 1.0),
        "streak_bonus": st.session_state.get("streak_bonus", 1.0),
        "streak_threshold": st.session_state.get("streak_threshold", 3),
        # Tier Config (Serialize)
        "tier_config": [
            {
                "name": t.name,
                "type": t.type.value,
                "min_mmr": t.min_mmr,
                "max_mmr": t.max_mmr,
                "demotion_lives": t.demotion_lives,
                "points_win": t.points_win,
                "points_draw": t.points_draw,
                "promotion_points": t.promotion_points,
                "capacity": t.capacity,
                "placement_min_mmr": t.placement_min_mmr,
                "placement_max_mmr": t.placement_max_mmr,
                "promotion_points_low": getattr(t, "promotion_points_low", t.promotion_points),
                "promotion_points_high": getattr(t, "promotion_points_high", t.promotion_points),
                "loss_point_correction": getattr(t, "loss_point_correction", 1.0),
                "bot_match_enabled": getattr(t, "bot_match_enabled", False),
                "bot_trigger_goal_diff": getattr(t, "bot_trigger_goal_diff", 99),
                "bot_trigger_loss_streak": getattr(t, "bot_trigger_loss_streak", 99)
            } for t in st.session_state.get("tier_config", [])
        ],
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
    
    # Extract comment for separate column
    user_comment_text = config.get("user_comments", "")
    
    # Save to Google Sheets
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Read existing to preserve other users
        df = pd.DataFrame()
        target_worksheet = "Simul_Config"
        try:
            df = conn.read(worksheet="Simul_Config", ttl=0)
        except:
            try:
                df = conn.read(worksheet="simul_config", ttl=0)
                target_worksheet = "simul_config"
            except:
                try:
                    df = conn.read(worksheet="Config", ttl=0)
                    target_worksheet = "Config"
                except:
                    try:
                        df = conn.read(worksheet="config", ttl=0)
                        target_worksheet = "config"
                    except:
                        df = pd.DataFrame()
                        # If all fail, default to Simul_Config for creation
                        target_worksheet = "Simul_Config"
            
        # Serialize Separate Chunks to avoid 50k char limit
        # 1. Tiers
        tier_data = config.get("tier_config", [])
        tier_json = json.dumps(tier_data, separators=(',', ':')) # Minify
        
        # 2. Segments
        seg_data = config.get("segments", [])
        seg_json = json.dumps(seg_data, separators=(',', ':')) # Minify
        
        # 3. Main Config (Exclude heavy items to keep it light)
        main_config = config.copy()
        main_config.pop("tier_config", None)
        main_config.pop("segments", None) 
        # Keep reset_rules/streak_rules in main for now as they are small
        
        main_json = json.dumps(main_config, separators=(',', ':'))
        
        new_row = {
            "username": current_username, 
            "ConfigJSON": main_json, 
            "TierConfigJSON": tier_json,
            "SegmentConfigJSON": seg_json,
            "Comment": user_comment_text
        }
        
        if df.empty:
            df_to_save = pd.DataFrame([new_row])
        else:
            # Check if username column exists
            if "username" in df.columns:
                # Normalize for comparison
                df["username_norm"] = df["username"].astype(str).str.strip().str.lower()
                target_user_norm = str(current_username).strip().lower()
                
                # Update existing user
                if target_user_norm in df["username_norm"].values:
                    # Find index
                    idx = df.index[df["username_norm"] == target_user_norm].tolist()[0]
                    
                    # Ensure columns exist before assigning
                    if "ConfigJSON" not in df.columns: df["ConfigJSON"] = ""
                    if "TierConfigJSON" not in df.columns: df["TierConfigJSON"] = ""
                    if "SegmentConfigJSON" not in df.columns: df["SegmentConfigJSON"] = ""
                    if "Comment" not in df.columns: df["Comment"] = ""
                    
                    # Use .loc for safety
                    df.loc[idx, "ConfigJSON"] = main_json
                    df.loc[idx, "TierConfigJSON"] = tier_json
                    df.loc[idx, "SegmentConfigJSON"] = seg_json
                    df.loc[idx, "Comment"] = user_comment_text
                    # Drop temp column
                    df = df.drop(columns=["username_norm"])
                    df_to_save = df
                else:
                    # Append new user
                    # Drop temp column before concat if we want to be clean, but df is local.
                    df = df.drop(columns=["username_norm"])
                    df_to_save = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                # Migration: Add username column to existing (assume they are legacy/orphaned or assign to 'admin'?)
                # Safest: Just append new structure. 
                # But we want to keep clean.
                # Let's just append.
                df_to_save = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
        conn.update(worksheet=target_worksheet, data=df_to_save)
        st.toast("설정이 저장되었습니다!")
    except Exception as e:
        st.error(f"설정 저장 실패: {e}")
        # Fallback to local
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except:
            pass

# --- Authentication ---
# --- Authentication & Session Management ---
cookie_manager = stx.CookieManager()

def is_user_admin(username):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = pd.DataFrame()
        try:
            df = conn.read(worksheet="Users", ttl=0)
        except Exception:
            try:
                df = conn.read(worksheet="users", ttl=0)
            except:
                return False
        
        if df.empty: return False
        
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'username' not in df.columns: return False
        
        df['username'] = df['username'].astype(str).str.strip()
        user_row = df[df['username'] == username.strip()]
        
        if not user_row.empty and 'admin' in df.columns:
            admin_val = user_row.iloc[0]['admin']
            # Robust check for True, 'true', '1', '1.0', 'yes'
            if str(admin_val).strip().lower() in ['true', '1', '1.0', 'yes']:
                return True
        return False
    except:
        return False

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
                return False, False
        
        if df.empty:
            st.error("디버그: Users 시트가 비어있습니다.")
            return False, False
            
        # Normalize columns: strip whitespace and lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        if 'username' not in df.columns or 'password' not in df.columns:
            st.error(f"디버그: 'username' 또는 'password' 컬럼이 없습니다. 발견된 컬럼: {df.columns.tolist()}")
            return False, False

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
                # Check Admin Status
                is_admin = False
                if 'admin' in df.columns:
                    admin_val = user_row.iloc[0]['admin']
                    # Handle various truthy values (True, 'TRUE', 'true', 1, '1', '1.0')
                    if str(admin_val).strip().lower() in ['true', '1', '1.0', 'yes']:
                        is_admin = True
                return True, is_admin
            else:
                pass
        else:
             pass
             
        return False, False
    except Exception as e:
        st.error(f"로그인 오류: {e}")
        return False, False

def login_page():
    st.title("로그인")
    st.info("시뮬레이션을 이용하려면 로그인하세요.")
    
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submit = st.form_submit_button("로그인")
        
        if submit:
            st.write(f"Debug: Login attempt with username='{username}'")
            is_valid, is_admin = check_password(username, password)
            if is_valid:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username # Store username
                st.session_state["is_admin"] = is_admin # Store admin status
                st.write(f"Debug: Login successful. User={username}, Admin={is_admin}")
                
                # Set cookies for persistence (expire in 1 day)
                expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
                cookie_manager.set("auth_user", username, expires_at=expires_at, key="set_auth_user")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 잘못되었습니다.")

def logout():
    # Clear all session state
    st.session_state.clear()
    st.session_state["authenticated"] = False
    st.session_state["logged_out"] = True # Flag to prevent immediate restore
        
    # Delete cookies (Overwrite with empty and expire)
    # cookie_manager.delete often fails to sync immediately. Setting to empty is safer.
    cookie_manager.set("auth_user", "", key="logout_auth_user_overwrite")
    cookie_manager.set("last_activity", "", key="logout_last_activity_overwrite")
    
    # Also try delete for good measure
    try:
        cookie_manager.delete("auth_user")
        cookie_manager.delete("last_activity")
    except:
        pass
        
    st.rerun()

# --- Main Execution Flow ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Check Cookies for Persistence
# Only check if not authenticated AND not just logged out
if not st.session_state["authenticated"] and not st.session_state.get("logged_out"):
    auth_user = cookie_manager.get("auth_user")
    last_activity = cookie_manager.get("last_activity")
    
    if st.session_state.get("is_admin", False):
        st.write(f"Debug: Cookie Check - User={auth_user}, LastAct={last_activity}")
    
    if auth_user and last_activity and auth_user != "":
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
                # Check admin status again during restore
                is_admin_restore = is_user_admin(auth_user)
                
                if is_admin_restore: # Use local var for debug print
                    st.write(f"Debug: Restoring session from cookie for user: {auth_user} (Admin)")
                
                st.session_state["authenticated"] = True
                st.session_state["username"] = auth_user # Restore username
                st.session_state["is_admin"] = is_admin_restore # Restore admin status
                
                # Update last activity
                expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
                cookie_manager.set("last_activity", str(current_time), expires_at=expires_at)
                st.rerun()
        except ValueError:
            pass
    else:
        if st.session_state.get("is_admin", False):
            st.write("Debug: No valid auth cookie found.")

if not st.session_state["authenticated"]:
    # Reset logged_out flag if we are showing login page (so next refresh can check cookies if needed? No, keep it until login)
    # Actually, if we are here, we are showing login page.
    login_page()
else:
    # We are authenticated.
    # Reset logged_out flag
    if st.session_state.get("logged_out"):
        st.session_state["logged_out"] = False
        
    # Hide Streamlit Header/Toolbar for Non-Admins
    if not st.session_state.get("is_admin", False):
        st.markdown("""
            <style>
                header[data-testid="stHeader"] {
                    visibility: hidden;
                }
                .stApp > header {
                    display: none;
                }
            </style>
            """, unsafe_allow_html=True)
        
    if st.session_state.get("is_admin", False):
        st.write(f"Debug: Main Logic - Current User: {st.session_state.get('username')}")
    # Update last activity timestamp on every interaction
    current_time = time.time()
    expires_at = datetime.datetime.now() + datetime.timedelta(days=1)
    
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
        st.write(f"로그인됨: {st.session_state.get('username', 'Unknown')}")
        
        # Admin-only Debug Tools
        if st.session_state.get("is_admin", False):
            st.divider()
            st.caption("관리자 도구 (Admin Tools)")
            
            if st.button("설정 다시 불러오기 (Reload Config)", key="reload_btn_top"):
                if 'config_loaded' in st.session_state:
                    del st.session_state['config_loaded']
                st.rerun()
                
            if st.button("데이터베이스 진단 (Debug DB)", key="debug_db_btn"):
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    df = pd.DataFrame()
                    try:
                        df = conn.read(worksheet="Simul_Config", ttl=0)
                        st.success("워크시트 'Simul_Config' 로드 성공")
                    except Exception as e1:
                        st.warning(f"'Simul_Config' 시트 로드 실패: {e1}")
                        try:
                            df = conn.read(worksheet="simul_config", ttl=0)
                            st.success("워크시트 'simul_config' (소문자) 로드 성공")
                        except Exception as e2:
                            st.warning(f"'simul_config' 시트 로드 실패: {e2}")
                            try:
                                df = conn.read(worksheet="Config", ttl=0)
                                st.success("워크시트 'Config' 로드 성공")
                            except Exception as e3:
                                st.warning(f"'Config' 시트 로드 실패: {e3}")
                                try:
                                    df = conn.read(worksheet="config", ttl=0)
                                    st.success("워크시트 'config' (소문자) 로드 성공")
                                except Exception as e4:
                                    st.error(f"모든 시트 로드 실패. 오류: {e4}")
                                    # Try reading default sheet
                                    try:
                                        st.info("기본(첫 번째) 시트를 읽어봅니다...")
                                        df = conn.read(ttl=0)
                                        st.success("기본 시트 로드 성공!")
                                        st.write(f"기본 시트 컬럼: {df.columns.tolist()}")
                                        st.warning("이 시트가 설정 시트라면, 탭 이름을 'Simul_Config'로 변경해 주세요.")
                                    except Exception as e5:
                                        st.error(f"기본 시트 로드도 실패: {e5}")
                                    st.exception(e4)
                            
                    if not df.empty:
                        st.write("--- 데이터베이스 진단 결과 (설정) ---")
                        st.write(f"1. 컬럼 목록: {df.columns.tolist()}")
                        
                        # Normalize
                        df.columns = [str(c).strip() for c in df.columns]
                        if "username" in df.columns:
                            df["username_norm"] = df["username"].astype(str).str.strip().str.lower()
                            current_user = st.session_state.get("username", "").strip().lower()
                            st.write(f"2. 현재 접속 계정: '{current_user}'")
                            
                            user_row = df[df["username_norm"] == current_user]
                            if not user_row.empty:
                                st.success(f"3. 계정 데이터 발견됨! (행 번호: {user_row.index[0]})")
                                json_data = user_row.iloc[0].get("ConfigJSON")
                                st.text(f"4. JSON 데이터 미리보기:\n{str(json_data)[:200]}...")
                                try:
                                    json.loads(json_data)
                                    st.success("5. JSON 파싱 성공: 데이터 형식이 올바릅니다.")
                                except Exception as e:
                                    st.error(f"5. JSON 파싱 실패: {e}")
                            else:
                                st.error(f"3. 계정 데이터 없음: '{current_user}'와 일치하는 행을 찾지 못했습니다.")
                                st.write(f"   (DB에 존재하는 유저 목록: {df['username_norm'].unique().tolist()})")
                        else:
                            st.error("오류: 'username' 컬럼이 시트에 없습니다.")
                        st.dataframe(df.head())
                        
                    # Debug Users Sheet
                    st.write("--- 데이터베이스 진단 결과 (유저) ---")
                    try:
                        users_df = conn.read(worksheet="username", ttl=0)
                        st.success("워크시트 'username' 로드 성공")
                    except:
                        try:
                            users_df = conn.read(worksheet="Users", ttl=0)
                            st.success("워크시트 'Users' 로드 성공")
                        except:
                            try:
                                users_df = conn.read(worksheet="users", ttl=0)
                                st.success("워크시트 'users' 로드 성공")
                            except Exception as ue:
                                st.error(f"유저 시트 로드 실패: {ue}")
                                users_df = pd.DataFrame()
                    
                    if not users_df.empty:
                        st.write(f"유저 시트 컬럼: {users_df.columns.tolist()}")
                        st.dataframe(users_df.head())
                        
                    st.write("----------------------------")
                except Exception as e:
                    st.error(f"진단 중 오류 발생: {e}")
            
            if st.button("쿠키 강제 삭제 (Force Clear Cookies)", type="secondary"):
                cookie_manager.delete("auth_user")
                cookie_manager.delete("last_activity")
                st.session_state.clear()
                st.rerun()
                
            # Session Debug Info
            with st.expander("세션 디버그 정보 (Session Debug)", expanded=False):
                st.write(f"Session Username: {st.session_state.get('username')}")
                st.write(f"Session Authenticated: {st.session_state.get('authenticated')}")
                st.write(f"Cookie Auth User: {cookie_manager.get('auth_user')}")
                st.write(f"Cookie Last Activity: {cookie_manager.get('last_activity')}")
            
            st.divider()
            
        if st.button("로그아웃"):
            logout()
            
    # Load Config at Startup
    if 'config_loaded' not in st.session_state:
        if st.session_state.get("is_admin", False):
            st.write("Debug: Calling load_config...")
        with st.spinner("데이터베이스에서 설정을 불러오는 중..."):
            loaded_config = load_config(st.session_state.get("username"))
            
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
                    try:
                        loaded_tiers = []
                        for t in v:
                            # Handle migration from old format (dict with "Tier", "Min", "Max")
                            if "Tier" in t:
                                loaded_tiers.append(TierConfig(
                                    name=t["Tier"],
                                    type=TierType.MMR,
                                    min_mmr=t.get("Min", 0),
                                    max_mmr=t.get("Max", 9999)
                                ))
                            else:
                                # New format
                                loaded_tiers.append(TierConfig(
                                    name=t["name"],
                                    type=TierType(t["type"]),
                                    min_mmr=t.get("min_mmr", 0),
                                    max_mmr=t.get("max_mmr", 9999),
                                    demotion_lives=t.get("demotion_lives", 0),
                                    points_win=t.get("points_win", 0),
                                    points_draw=t.get("points_draw", 0),
                                    promotion_points=t.get("promotion_points", 100),
                                    capacity=t.get("capacity", 0),
                                    placement_min_mmr=t.get("placement_min_mmr", t.get("min_mmr", 0)),
                                    placement_max_mmr=t.get("placement_max_mmr", t.get("max_mmr", 0)),
                                    promotion_points_low=t.get("promotion_points_low", t.get("promotion_points", 100)),
                                    promotion_points_high=t.get("promotion_points_high", t.get("promotion_points", 100)),
                                    loss_point_correction=t.get("loss_point_correction", 1.0),
                                    bot_match_enabled=t.get("bot_match_enabled", False),
                                    bot_trigger_goal_diff=t.get("bot_trigger_goal_diff", 99),
                                    bot_trigger_loss_streak=t.get("bot_trigger_loss_streak", 99)
                                ))
                        st.session_state.tier_config = loaded_tiers
                    except Exception as e:
                        st.warning(f"티어 설정 로드 중 오류: {e}")
                        st.session_state.tier_config = []
                else:
                    st.session_state[k] = v
            
            # Ensure default if not in config
            if "use_true_skill_init" not in st.session_state:
                st.session_state.use_true_skill_init = False
            if "user_comments" in loaded_config:
                st.session_state.user_comments = loaded_config["user_comments"]
                
        st.session_state.config_loaded = True


    # Initialize Session State (Defaults)
    if 'segments' not in st.session_state:
        # Try loading from CSV first
        loaded_segments = []
        if os.path.exists("segment_balance.csv"):
            try:
                df_seg = pd.read_csv("segment_balance.csv")
                for _, row in df_seg.iterrows():
                    loaded_segments.append(SegmentConfig(
                        row["name"], float(row["ratio"]), float(row["daily_play_prob"]),
                        float(row["matches_per_day_min"]), float(row["matches_per_day_max"]),
                        float(row["true_skill_min"]), float(row["true_skill_max"]),
                        int(row["active_hour_start"]), int(row["active_hour_end"])
                    ))
            except Exception as e:
                st.warning(f"세그먼트 CSV 로드 실패: {e}")
        
        if loaded_segments:
            st.session_state.segments = loaded_segments
        else:
            # Fallback to Defaults
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
            
            st.session_state.use_true_skill_init = st.checkbox("True Skill 기반 초기 배치 (True Skill Based Initialization)", value=st.session_state.get("use_true_skill_init", False), help="체크 시, 유저의 True Skill이 속한 티어의 배치 구간에서 초기 MMR이 결정됩니다.")
            
            if st.session_state.use_true_skill_init:
                st.caption("※ 초기 MMR 설정은 무시되고, 각 유저의 실력에 맞는 티어 배치 구간에서 시작합니다.")
                # We keep initial_mmr in state but maybe disable it or just show it's ignored
                st.session_state.initial_mmr = st.number_input("초기 MMR (기본값)", value=st.session_state.get("initial_mmr", 1000.0), disabled=True, help="True Skill 기반 배치 시에는 사용되지 않습니다 (매칭되지 않는 경우 제외).")
            else:
                st.session_state.initial_mmr = st.number_input("초기 MMR", value=st.session_state.get("initial_mmr", 1000.0), help="모든 유저의 시작 MMR 점수입니다.")

        with st.expander("매치 설정 (Match Configuration)"):
            st.session_state.draw_prob = st.slider("무승부 확률 (Draw Prob)", 0.0, 0.5, st.session_state.get("draw_prob", 0.15), help="정규 시간 내 무승부 확률입니다.")
            st.session_state.prob_et = st.slider("연장전 확률 (Extra Time Prob)", 0.0, 1.0, st.session_state.get("prob_et", 0.2), help="무승부 시 연장전으로 갈 확률입니다.")
            st.session_state.prob_pk = st.slider("승부차기 확률 (PK Prob)", 0.0, 1.0, st.session_state.get("prob_pk", 0.5), help="연장전 무승부 시 승부차기로 갈 확률입니다.")
            st.session_state.max_goal_diff = st.number_input("최대 골 득실", min_value=1, value=st.session_state.get("max_goal_diff", 5), help="경기에서 발생할 수 있는 최대 골 득실 차이입니다.")
            st.session_state.matchmaking_jitter = st.number_input("매칭 범위 (Jitter)", value=st.session_state.get("matchmaking_jitter", 50.0), help="매칭 시 MMR 검색 범위의 표준편차입니다.")
            st.session_state.bot_win_rate = st.slider("봇 매치 승률 (Bot Win Rate)", 0.0, 1.0, st.session_state.get("bot_win_rate", 0.8), help="봇 매치 시 유저가 승리할 확률입니다.")

        with st.expander("ELO 설정 (ELO Configuration)"):
            st.session_state.base_k = st.number_input("기본 K-Factor", value=st.session_state.get("base_k", 20.0), help="ELO 점수 변동폭의 기본값입니다.")
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
                
                # Bulk Input
                streak_map = {"연승 횟수": "min_streak", "보너스 점수": "bonus", "min_streak": "min_streak", "bonus": "bonus"} # Self-map included
                new_streak_df = render_bulk_csv_uploader("연승 규칙", st.session_state.streak_rules, "streak", streak_map)
                if new_streak_df is not None:
                    st.session_state.streak_rules = new_streak_df
                    st.rerun()
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
                
                # Bulk Input
                gd_map = {"골 득실차": "min_diff", "보너스 점수": "bonus", "min_diff": "min_diff", "bonus": "bonus"}
                new_gd_df = render_bulk_csv_uploader("골 득실 규칙", st.session_state.goal_diff_rules, "gd", gd_map)
                if new_gd_df is not None:
                    st.session_state.goal_diff_rules = new_gd_df
                    st.rerun()
            except Exception as e:
                st.error(f"골 득실 규칙 표시 오류: {e}")
                st.session_state.goal_diff_rules = pd.DataFrame(columns=["min_diff", "bonus"])
            
            st.subheader("승리 유형별 가중치 (Decay)")
            st.session_state.decay_et = st.slider("연장승 가중치", 0.0, 1.0, st.session_state.get("decay_et", 0.8), help="연장전 승리 시 획득 점수 비율입니다 (1.0 = 정상 점수).")
            st.session_state.decay_pk = st.slider("승부차기승 가중치", 0.0, 1.0, st.session_state.get("decay_pk", 0.6), help="승부차기 승리 시 획득 점수 비율입니다.")
            
            st.subheader("랭크 포인트 수렴 (Rank Point Convergence)")
            st.session_state.point_convergence_rate = st.slider("랭크 포인트 수렴 속도", 0.0, 1.0, st.session_state.get("point_convergence_rate", 0.5), help="MMR 변동이 랭크 포인트에 반영되는 비율입니다. (1.0 = 즉시 반영, 0.1 = 천천히 반영)")

            st.subheader("MMR 압축 보정 (Calibration)")
            st.subheader("MMR 압축 보정 (Calibration)")
            # st.session_state.calibration_enabled is now controlled in the main Control Panel
            if st.session_state.get("calibration_enabled", False):
                st.session_state.calibration_k_bonus = st.number_input("보정 K-Bonus 배율", value=st.session_state.get("calibration_k_bonus", 2.0), help="보정 모드 시 적용할 추가 K-Factor 배율입니다.")
                st.session_state.calibration_match_count = st.number_input("보정 적용 경기 수", value=st.session_state.get("calibration_match_count", 10), help="보정 모드가 적용되는 경기 수입니다.")
            else:
                st.caption("제어판에서 보정 모드를 활성화하면 추가 설정이 표시됩니다.")

        # --- Tier Configuration ---
    if 'tier_config' not in st.session_state or not st.session_state.tier_config:
        # Try loading from CSV first
        loaded_tiers = []
        if os.path.exists("tier_balance.csv"):
            try:
                df_tier = pd.read_csv("tier_balance.csv")
                for _, row in df_tier.iterrows():
                    # Handle TierType enum conversion
                    t_type_str = row["type"]
                    t_type = TierType.MMR # Default
                    if t_type_str == "Ladder": t_type = TierType.LADDER
                    elif t_type_str == "Ratio": t_type = TierType.RATIO
                    elif t_type_str == "MMR": t_type = TierType.MMR
                    
                    loaded_tiers.append(TierConfig(
                        name=str(row["name"]),
                        type=t_type,
                        min_mmr=float(row["min_mmr"]),
                        max_mmr=float(row["max_mmr"]),
                        demotion_lives=int(row.get("demotion_lives", 0)),
                        points_win=int(row.get("points_win", 0)),
                        points_draw=int(row.get("points_draw", 0)),
                        promotion_points=int(row.get("promotion_points", 100)),
                        capacity=int(row.get("capacity", 0)),
                        placement_min_mmr=float(row.get("placement_min_mmr", 0)),
                        placement_max_mmr=float(row.get("placement_max_mmr", 0)),
                        promotion_points_low=int(row.get("promotion_points_low", row.get("promotion_points", 100))),
                        promotion_points_high=int(row.get("promotion_points_high", row.get("promotion_points", 100))),
                        loss_point_correction=float(row.get("loss_point_correction", 1.0)),
                        bot_match_enabled=bool(row.get("bot_match_enabled", False)),
                        bot_trigger_goal_diff=int(row.get("bot_trigger_goal_diff", 99)),
                        bot_trigger_loss_streak=int(row.get("bot_trigger_loss_streak", 99))
                    ))
            except Exception as e:
                st.warning(f"티어 CSV 로드 실패: {e}")
        
        if loaded_tiers:
            st.session_state.tier_config = loaded_tiers
        else:
            # Default Tiers
            st.session_state.tier_config = [
                TierConfig("Bronze", TierType.MMR, 0, 1200, placement_min_mmr=0, placement_max_mmr=1200),
                TierConfig("Silver", TierType.MMR, 1200, 1400, placement_min_mmr=1200, placement_max_mmr=1400),
                TierConfig("Gold", TierType.MMR, 1400, 1600, placement_min_mmr=1400, placement_max_mmr=1600),
                TierConfig("Platinum", TierType.LADDER, 1600, 1800, 0, 3, 10, 5, 100, placement_min_mmr=1600, placement_max_mmr=1800), # Example Ladder
                TierConfig("Diamond", TierType.RATIO, 1800, 9999, 0, 0, 0, 0, 0, 100, placement_min_mmr=1800, placement_max_mmr=9999) # Example Ratio
            ]

    with st.expander("티어 기준 설정 (Tier Config)"):
            st.caption("티어별 승강등 규칙을 설정하세요. 순서는 낮은 티어부터 높은 티어 순입니다.")
            
            # Convert TierConfig objects to DataFrame for editing
            tier_data = []
            if st.session_state.tier_config:
                for t in st.session_state.tier_config:
                    tier_data.append({
                        "name": t.name,
                        "type": t.type.value,
                        "min_mmr": t.min_mmr,
                        "max_mmr": t.max_mmr,
                        "demotion_mmr": getattr(t, "demotion_mmr", 0.0),
                        "demotion_lives": getattr(t, "demotion_lives", 0),
                        "loss_point_correction": getattr(t, "loss_point_correction", 1.0),
                        "points_win": getattr(t, "points_win", 0),
                        "points_draw": getattr(t, "points_draw", 0),
                        "points_loss": getattr(t, "points_loss", 0),
                        "promotion_points": getattr(t, "promotion_points", 100),
                        "promotion_points_low": getattr(t, "promotion_points_low", 100),
                        "promotion_points_high": getattr(t, "promotion_points_high", 100),
                        "promotion_mmr_2": getattr(t, "promotion_mmr_2", 0.0),
                        "promotion_mmr_3": getattr(t, "promotion_mmr_3", 0.0),
                        "promotion_mmr_4": getattr(t, "promotion_mmr_4", 0.0),
                        "promotion_mmr_5": getattr(t, "promotion_mmr_5", 0.0),
                        "capacity": getattr(t, "capacity", 0),
                        "placement_min_mmr": getattr(t, "placement_min_mmr", 0.0),
                        "placement_max_mmr": getattr(t, "placement_max_mmr", 0.0),
                        "bot_match_enabled": t.bot_match_enabled,
                        "bot_trigger_goal_diff": t.bot_trigger_goal_diff,
                        "bot_trigger_loss_streak": t.bot_trigger_loss_streak
                    })
            
            df_tiers = pd.DataFrame(tier_data)
            
            # Ensure columns exist if empty
            if df_tiers.empty:
                df_tiers = pd.DataFrame(columns=[
                    "name", "type", "min_mmr", "max_mmr", "demotion_mmr", "demotion_lives", "loss_point_correction",
                    "points_win", "points_draw", "points_loss", 
                    "promotion_points", "promotion_points_low", "promotion_points_high",
                    "promotion_mmr_2", "promotion_mmr_3", "promotion_mmr_4", "promotion_mmr_5",
                    "capacity", "placement_min_mmr", "placement_max_mmr", 
                    "bot_match_enabled", "bot_trigger_goal_diff", "bot_trigger_loss_streak"
                ])

            try:
                edited_tiers = st.data_editor(
                    df_tiers,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="tier_editor",
                    column_config={
                        "name": st.column_config.TextColumn("티어 이름", required=True),
                        "type": st.column_config.SelectboxColumn("타입", options=["MMR", "Ladder", "Ratio", "ELO"], required=True),
                        "min_mmr": st.column_config.NumberColumn("최소 MMR", step=10),
                        "max_mmr": st.column_config.NumberColumn("최대 MMR", step=10),
                        "demotion_mmr": st.column_config.NumberColumn("강등 위험 MMR", step=10, help="이 MMR 미만일 때 패배 시 강등 방어 횟수 차감"),
                        "demotion_lives": st.column_config.NumberColumn("강등 방어 횟수", step=1, help="강등 위험 상태에서 패배 시 차감되는 횟수 (0=강등 없음)"),
                        "loss_point_correction": st.column_config.NumberColumn("패배 포인트 보정", step=0.1, help="패배 시 포인트 감소량 보정 (예: 0.8 = 80%만 감소)"),
                        "points_win": st.column_config.NumberColumn("승리 포인트", step=1),
                        "points_draw": st.column_config.NumberColumn("무승부 포인트", step=1),
                        "points_loss": st.column_config.NumberColumn("패배 시 차감 승점", step=1, help="Ladder: 패배 시 차감되는 승점 (0점 도달 시 강등 방어 차감)"),
                        "promotion_points": st.column_config.NumberColumn("승급 포인트", step=10),
                        "promotion_points_low": st.column_config.NumberColumn("승급 포인트 (Low MMR)", step=10),
                        "promotion_points_high": st.column_config.NumberColumn("승급 포인트 (High MMR)", step=10),
                        "promotion_mmr_2": st.column_config.NumberColumn("승점 2배 MMR", step=10, help="이 MMR 미만일 때 승리 시 승점 2배"),
                        "promotion_mmr_3": st.column_config.NumberColumn("승점 3배 MMR", step=10, help="이 MMR 미만일 때 승리 시 승점 3배"),
                        "promotion_mmr_4": st.column_config.NumberColumn("승점 4배 MMR", step=10, help="이 MMR 미만일 때 승리 시 승점 4배"),
                        "promotion_mmr_5": st.column_config.NumberColumn("승점 5배 MMR", step=10, help="이 MMR 미만일 때 승리 시 승점 5배"),
                        "capacity": st.column_config.NumberColumn("정원 (Ratio)", step=1),
                        "placement_min_mmr": st.column_config.NumberColumn("배치 최소 MMR", step=10),
                        "placement_max_mmr": st.column_config.NumberColumn("배치 최대 MMR", step=10),
                        "bot_match_enabled": st.column_config.CheckboxColumn("봇 매치"),
                        "bot_trigger_goal_diff": st.column_config.NumberColumn("봇 트리거 (골득실)"),
                        "bot_trigger_loss_streak": st.column_config.NumberColumn("봇 트리거 (연패)")
                    }
                )
            except Exception as e:
                st.error(f"티어 설정 표시 오류: {e}")
                edited_tiers = pd.DataFrame()

            # Bulk Input for Tiers
            tier_map = {
                "티어 이름": "name", "타입": "type", "최소 MMR": "min_mmr", "최대 MMR": "max_mmr",
                "강등 위험 MMR": "demotion_mmr", "강등 방어 횟수": "demotion_lives", "패배 포인트 보정": "loss_point_correction",
                "승리 포인트": "points_win", "무승부 포인트": "points_draw", "승급 포인트": "promotion_points",
                "패배 차감 승점": "points_loss",
                "승급 포인트 (Low MMR)": "promotion_points_low", "승급 포인트 (High MMR)": "promotion_points_high",
                "승점 2배 MMR": "promotion_mmr_2", "승점 3배 MMR": "promotion_mmr_3", 
                "승점 4배 MMR": "promotion_mmr_4", "승점 5배 MMR": "promotion_mmr_5",
                "정원 (Ratio)": "capacity", "배치 최소 MMR": "placement_min_mmr", "배치 최대 MMR": "placement_max_mmr",
                "봇 매치": "bot_match_enabled", "봇 트리거 (골득실)": "bot_trigger_goal_diff", "봇 트리거 (연패)": "bot_trigger_loss_streak",
                # English mappings
                "name": "name", "type": "type", "min_mmr": "min_mmr", "max_mmr": "max_mmr",
                "demotion_mmr": "demotion_mmr", "demotion_lives": "demotion_lives", "loss_point_correction": "loss_point_correction",
                "points_win": "points_win", "points_draw": "points_draw", "points_loss": "points_loss",
                "promotion_points": "promotion_points",
                "promotion_points_low": "promotion_points_low", "promotion_points_high": "promotion_points_high",
                "promotion_mmr_2": "promotion_mmr_2", "promotion_mmr_3": "promotion_mmr_3",
                "promotion_mmr_4": "promotion_mmr_4", "promotion_mmr_5": "promotion_mmr_5",
                "capacity": "capacity", "placement_min_mmr": "placement_min_mmr", "placement_max_mmr": "placement_max_mmr",
                "bot_match_enabled": "bot_match_enabled", "bot_trigger_goal_diff": "bot_trigger_goal_diff", "bot_trigger_loss_streak": "bot_trigger_loss_streak"
            }
            new_tier_df = render_bulk_csv_uploader("티어 설정", df_tiers, "tier", tier_map)
            if new_tier_df is not None:
                try:
                    bulk_tiers = []
                    for index, row in new_tier_df.iterrows():
                        bulk_tiers.append(safe_create_tier_config(
                            name=str(row["name"]),
                            type=TierType(row["type"]) if isinstance(row["type"], str) else TierType(row["type"]), # Handle string or enum
                            min_mmr=float(row.get("min_mmr", 0)),
                            max_mmr=float(row.get("max_mmr", 9999)),
                            demotion_mmr=float(row.get("demotion_mmr", 0)),
                            demotion_lives=int(row.get("demotion_lives", 0)),
                            loss_point_correction=float(row.get("loss_point_correction", 1.0)),
                            points_win=int(row.get("points_win", 0)),
                            points_draw=int(row.get("points_draw", 0)),
                            points_loss=int(row.get("points_loss", 0)),
                            promotion_points=int(row.get("promotion_points", 100)),
                            promotion_points_low=int(row.get("promotion_points_low", 100)),
                            promotion_points_high=int(row.get("promotion_points_high", 100)),
                            promotion_mmr_2=float(row.get("promotion_mmr_2", 0)),
                            promotion_mmr_3=float(row.get("promotion_mmr_3", 0)),
                            promotion_mmr_4=float(row.get("promotion_mmr_4", 0)),
                            promotion_mmr_5=float(row.get("promotion_mmr_5", 0)),
                            capacity=int(row.get("capacity", 0)),
                            placement_min_mmr=float(row.get("placement_min_mmr", 0)),
                            placement_max_mmr=float(row.get("placement_max_mmr", 0)),
                            bot_match_enabled=bool(row.get("bot_match_enabled", False)),
                            bot_trigger_goal_diff=int(row.get("bot_trigger_goal_diff", 99)),
                            bot_trigger_loss_streak=int(row.get("bot_trigger_loss_streak", 99))
                        ))
                    st.session_state.tier_config = bulk_tiers
                    st.rerun()
                except Exception as e:
                    st.error(f"티어 CSV 적용 오류: {e}")

            # Apply Manual Edits
            if not edited_tiers.equals(df_tiers):
                new_tiers = []
                for index, row in edited_tiers.iterrows():
                    try:
                        t = safe_create_tier_config(
                            name=str(row["name"]),
                            type=TierType(row["type"]) if isinstance(row["type"], str) else TierType(row["type"]),
                            min_mmr=float(row["min_mmr"]),
                            max_mmr=float(row["max_mmr"]),
                            demotion_mmr=float(row.get("demotion_mmr", 0)),
                            demotion_lives=int(row.get("demotion_lives", 0)),
                            loss_point_correction=float(row.get("loss_point_correction", 1.0)),
                            points_win=int(row["points_win"]),
                            points_draw=int(row["points_draw"]),
                            points_loss=int(row.get("points_loss", 0)),
                            promotion_points=int(row["promotion_points"]),
                            promotion_points_low=int(row["promotion_points_low"]),
                            promotion_points_high=int(row["promotion_points_high"]),
                            promotion_mmr_2=float(row.get("promotion_mmr_2", 0)),
                            promotion_mmr_3=float(row.get("promotion_mmr_3", 0)),
                            promotion_mmr_4=float(row.get("promotion_mmr_4", 0)),
                            promotion_mmr_5=float(row.get("promotion_mmr_5", 0)),
                            capacity=int(row["capacity"]),
                            placement_min_mmr=float(row.get("placement_min_mmr", 0)),
                            placement_max_mmr=float(row.get("placement_max_mmr", 0)),
                            bot_match_enabled=bool(row.get("bot_match_enabled", False)),
                            bot_trigger_goal_diff=int(row.get("bot_trigger_goal_diff", 99)),
                            bot_trigger_loss_streak=int(row.get("bot_trigger_loss_streak", 99))
                        )
                        new_tiers.append(t)
                    except Exception as e:
                        # st.warning(f"티어 데이터 오류 (행 {index}): {e}")
                        pass
                st.session_state.tier_config = new_tiers


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
            
            # Bulk Input for Segments
            seg_map = {
                "세그먼트 이름": "name", "비율": "ratio", "일일 플레이 확률": "daily_play_prob",
                "일일 최소 경기": "matches_per_day_min", "일일 최대 경기": "matches_per_day_max",
                "최소 실력 (True Skill)": "true_skill_min", "최대 실력 (True Skill)": "true_skill_max",
                "주요 활동 시작 시간": "active_hour_start", "주요 활동 종료 시간": "active_hour_end",
                # English keys
                "name": "name", "ratio": "ratio", "daily_play_prob": "daily_play_prob",
                "matches_per_day_min": "matches_per_day_min", "matches_per_day_max": "matches_per_day_max",
                "true_skill_min": "true_skill_min", "true_skill_max": "true_skill_max",
                "active_hour_start": "active_hour_start", "active_hour_end": "active_hour_end"
            }
            # Use checkbox instead of expander to avoid nesting issue
            # Also ensure df_segments is available (it should be from above, but if empty, create new)
            if 'df_segments' not in locals(): df_segments = pd.DataFrame()
            
            new_seg_df = render_bulk_csv_uploader("유저 세그먼트", df_segments, "segment", seg_map, use_expander=False)
            if new_seg_df is not None:
                try:
                    bulk_segments = []
                    for index, row in new_seg_df.iterrows():
                        bulk_segments.append(SegmentConfig(
                            str(row["name"]), float(row["ratio"]), float(row["daily_play_prob"]),
                            float(row["matches_per_day_min"]), float(row["matches_per_day_max"]),
                            float(row["true_skill_min"]), float(row["true_skill_max"]),
                            int(row["active_hour_start"]), int(row["active_hour_end"])
                        ))
                    st.session_state.segments = bulk_segments
                    st.rerun()
                except Exception as e:
                    st.error(f"세그먼트 CSV 적용 오류: {e}")

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
                st.session_state.reset_rules = pd.DataFrame(columns=["min_mmr", "max_mmr", "reset_mmr", "soft_reset_ratio"])
            
            # Defensive check & Schema Migration
            if not isinstance(st.session_state.reset_rules, pd.DataFrame):
                st.session_state.reset_rules = pd.DataFrame(st.session_state.reset_rules)
            
            # Check if we need to migrate columns (e.g. if tier_name exists or max_mmr missing)
            current_cols = set(st.session_state.reset_rules.columns)
            required_cols = {"min_mmr", "max_mmr", "reset_mmr", "soft_reset_ratio"}
            
            if not required_cols.issubset(current_cols):
                 # Reset to new schema if columns don't match
                 st.session_state.reset_rules = pd.DataFrame([
                     {"min_mmr": 0, "max_mmr": 9999, "reset_mmr": 1000, "soft_reset_ratio": 0.0}
                 ])
                 
            try:
                st.session_state.reset_rules = st.data_editor(
                    st.session_state.reset_rules, 
                    num_rows="dynamic", 
                    use_container_width=True, 
                    key="reset_editor",
                    column_config={
                        "min_mmr": st.column_config.NumberColumn("최소 MMR", required=True, step=10),
                        "max_mmr": st.column_config.NumberColumn("최대 MMR", required=True, step=10),
                        "reset_mmr": st.column_config.NumberColumn("초기화 목표 MMR", required=True, step=10),
                        "soft_reset_ratio": st.column_config.NumberColumn("압축 비율 (0=완전초기화)", required=True, min_value=0.0, max_value=1.0, step=0.1, help="0이면 목표 MMR로 완전 초기화, 1이면 현재 MMR 유지. 0.5면 중간값.")
                    }
                )
                
                # Bulk Input
                reset_map = {
                    "최소 MMR": "min_mmr", "최대 MMR": "max_mmr", "초기화 목표 MMR": "reset_mmr", "압축 비율 (0=완전초기화)": "soft_reset_ratio",
                    "min_mmr": "min_mmr", "max_mmr": "max_mmr", "reset_mmr": "reset_mmr", "soft_reset_ratio": "soft_reset_ratio"
                }
                new_reset_df = render_bulk_csv_uploader("초기화 규칙", st.session_state.reset_rules, "reset", reset_map)
                if new_reset_df is not None:
                    st.session_state.reset_rules = new_reset_df
                    st.rerun()
            except Exception as e:
                st.error(f"초기화 규칙 표시 오류: {e}")
                st.session_state.reset_rules = pd.DataFrame(columns=["min_mmr", "max_mmr", "reset_mmr", "soft_reset_ratio"])

    if st.button("설정 저장"):
            save_config(st.session_state.get("username"))
            st.success("설정이 저장되었습니다!")

    # --- Main Content ---
    st.title("Rank simulation")

    tab1, tab2, tab3, tab4 = st.tabs(["시뮬레이션 실행", "분석", "매치 기록", "랭크 분석"])

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
                        draw_prob=st.session_state.draw_prob,
                        prob_extra_time=st.session_state.prob_et,
                        prob_pk=st.session_state.prob_pk,
                        max_goal_diff=st.session_state.max_goal_diff,
                        matchmaking_jitter=st.session_state.matchmaking_jitter,
                        bot_win_rate=st.session_state.bot_win_rate
                    )
                    
                    # Ensure segments are objects
                    segment_configs = []
                    for s in st.session_state.segments:
                        if isinstance(s, dict):
                            segment_configs.append(SegmentConfig(**s))
                        else:
                            segment_configs.append(s)

                    # Initialize or Update Simulation
                    if hard_reset or st.session_state.simulation is None:
                        st.session_state.simulation = FastSimulation(
                            num_users=st.session_state.num_users,
                            segment_configs=segment_configs,
                            elo_config=elo_config,
                            match_config=match_config,
                            tier_configs=st.session_state.tier_config,
                            initial_mmr=st.session_state.initial_mmr,
                            use_true_skill_init=st.session_state.get("use_true_skill_init", False),
                            reset_rules=st.session_state.reset_rules.to_dict('records') if 'reset_rules' in st.session_state and isinstance(st.session_state.reset_rules, pd.DataFrame) else []
                        )
                        st.session_state.stats_history = []
                        st.success(f"시뮬레이션이 초기화되었습니다. (Day 0)")
                    else:
                        # Update existing simulation configs
                        st.session_state.simulation.elo_config = elo_config
                        st.session_state.simulation.match_config = match_config
                        st.session_state.simulation.segment_configs = segment_configs
                        st.session_state.simulation.tier_configs = st.session_state.tier_config
                        st.success(f"시뮬레이션 설정이 업데이트되었습니다. (Day {st.session_state.simulation.day}부터 계속)")
                    
                    st.session_state.simulation.initialize_users()
                    
                # Run Simulation
                st.write("Debug: Starting simulation loop...") # Debug
                progress_bar = st.progress(0)
                status_text = st.empty()
                sim = st.session_state.simulation
                
                stats_history = []
                if not hard_reset and 'stats_history' in st.session_state and st.session_state.stats_history:
                     stats_history = st.session_state.stats_history
                
                st.write(f"Debug: Num days = {st.session_state.num_days}") # Debug
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

        with col2:
            st.subheader("시즌 관리")
            if st.session_state.simulation is not None:
                if st.button("시즌 초기화 (Soft Reset)"):
                    try:
                        # Ensure numeric types for sorting and processing
                        rules_df = st.session_state.reset_rules.copy()
                        cols_to_numeric = ["min_mmr", "max_mmr", "reset_mmr", "soft_reset_ratio"]
                        for col in cols_to_numeric:
                            rules_df[col] = pd.to_numeric(rules_df[col], errors='coerce')
                        
                        # Drop rows with invalid numbers if any
                        rules_df = rules_df.dropna(subset=cols_to_numeric)
                        
                        # Sort by min_mmr just in case
                        rules_df = rules_df.sort_values("min_mmr")
                        
                        rules = []
                        for index, row in rules_df.iterrows():
                            rules.append({
                                "min": float(row["min_mmr"]),
                                "max": float(row["max_mmr"]),
                                "target": float(row["reset_mmr"]),
                                "compression": float(row["soft_reset_ratio"])
                            })
                        
                        st.session_state.simulation.apply_tiered_reset(rules)
                            
                        st.success("시즌 초기화가 완료되었습니다. (MMR 압축 및 전적 초기화)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"초기화 중 오류 발생: {e}")
            else:
                st.info("시뮬레이션을 먼저 실행하세요.")

            st.divider()
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
                
                sim = st.session_state.simulation
                
                if sim.tier_configs:
                    # Use tracked tier indices
                    tier_counts = {}
                    for i, config in enumerate(sim.tier_configs):
                        count = np.sum(sim.user_tier_index == i)
                        tier_counts[config.name] = count
                    
                    tier_df = pd.DataFrame(list(tier_counts.items()), columns=["Tier", "Count"])
                    fig_tier = px.bar(tier_df, x="Tier", y="Count", title="티어별 유저 수 (현재)")
                    st.plotly_chart(fig_tier, use_container_width=True)
                else:
                    # Fallback to MMR ranges if no tier config (Legacy)
                    raw_tiers = st.session_state.tier_config
                    # Sort by Min score to ensure correct order
                    sorted_tiers = sorted(raw_tiers, key=lambda x: x.min_mmr if isinstance(x, TierConfig) else x['Min'])
                    
                    tiers = []
                    for t in sorted_tiers:
                        if isinstance(t, TierConfig):
                             tiers.append((t.name, t.min_mmr, t.max_mmr))
                        else:
                             tiers.append((t['Tier'], t['Min'], t['Max']))
                    
                    tier_counts = {name: 0 for name, _, _ in tiers}
                    for mmr in sim.mmr:
                        for name, low, high in tiers:
                            if low <= mmr < high:
                                tier_counts[name] += 1
                                break
                
                    tier_df = pd.DataFrame(list(tier_counts.items()), columns=["Tier", "Count"])
                    fig_tier = px.bar(tier_df, x="Tier", y="Count", title="티어별 유저 수 (MMR 기준)")
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
            
            # Get all available segments from watched indices
            # watched_indices is {user_idx: segment_name}
            available_segments = sorted(list(set(sim.watched_indices.values())))
            
            selected_segments_multi = st.multiselect("확인할 세그먼트 선택 (다중 선택 가능)", available_segments, default=available_segments[:1] if available_segments else None)
            
            if selected_segments_multi:
                # Find all sample users belonging to selected segments
                target_indices = [idx for idx, seg_name in sim.watched_indices.items() if seg_name in selected_segments_multi]
                
                if not target_indices:
                    st.warning("선택한 세그먼트에 해당하는 샘플 유저가 없습니다.")
                else:
                    st.write(f"**선택된 샘플 유저 수: {len(target_indices)}명**")
                    
                    # 1. Aggregated Match Logs
                    all_logs_data = []
                    
                    for target_idx in target_indices:
                        seg_name = sim.watched_indices[target_idx]
                        current_mmr = sim.mmr[target_idx]
                        current_ts = sim.true_skill[target_idx]
                        
                        logs = sim.match_logs.get(target_idx, [])
                        for log in logs:
                            pre_mmr = log.current_mmr - log.mmr_change
                            all_logs_data.append({
                                "User ID": target_idx,
                                "Segment": seg_name,
                                "Current MMR": f"{pre_mmr:.1f}",
                                "True Skill": f"{current_ts:.1f}",
                                "Day": log.day,
                                "Opponent ID": log.opponent_id,
                                "Opponent MMR": f"{log.opponent_mmr:.1f}",
                                "Opponent True Skill": f"{log.opponent_true_skill:.1f}",
                                "Result": f"{log.result} ({log.result_type})",
                                "Goal Diff": log.goal_diff,
                                "Change": f"{log.mmr_change:+.1f}",
                                "New MMR": f"{log.current_mmr:.1f}",
                                "Tier": "Unranked (배치)" if log.current_tier_index == -1 else (sim.tier_configs[log.current_tier_index].name if sim.tier_configs and 0 <= log.current_tier_index < len(sim.tier_configs) else "-"),
                                "Ladder Points": log.current_ladder_points
                            })
                    
                    if all_logs_data:
                        df_logs = pd.DataFrame(all_logs_data)
                        # Reorder columns
                        cols = ["User ID", "Segment", "Current MMR", "True Skill", "Day", "Result", "Change", "New MMR", "Tier", "Opponent MMR", "Goal Diff"]
                        # Add remaining columns
                        cols += [c for c in df_logs.columns if c not in cols]
                        df_logs = df_logs[cols]
                        
                        st.dataframe(df_logs, use_container_width=True)
                    else:
                        st.info("매치 기록이 없습니다.")

                    # 2. Aggregated Segment Statistics
                    st.divider()
                    st.markdown(f"#### 선택된 세그먼트 통합 통계 ({', '.join(selected_segments_multi)})")
                    
                    # Identify all users in these segments (not just sample users)
                    # sim.seg_names stores names by segment index
                    target_seg_indices = []
                    if hasattr(sim, 'seg_names'):
                        for seg_name in selected_segments_multi:
                            try:
                                idx = sim.seg_names.index(seg_name)
                                target_seg_indices.append(idx)
                            except ValueError:
                                pass
                    
                    if target_seg_indices:
                        # Create mask for all users in selected segments
                        combined_mask = np.isin(sim.segment_indices, target_seg_indices)
                        
                        if combined_mask.any():
                            seg_mmr = sim.mmr[combined_mask]
                            seg_ts = sim.true_skill[combined_mask]
                            seg_tier = sim.user_tier_index[combined_mask]
                            seg_matches = sim.matches_played[combined_mask]
                            
                            # Helper to get tier name
                            def get_tier_name(idx):
                                idx = int(round(idx))
                                if idx == -1: return "Unranked"
                                if sim.tier_configs and 0 <= idx < len(sim.tier_configs):
                                    return sim.tier_configs[idx].name
                                return str(idx)

                            min_tier = np.min(seg_tier)
                            med_tier = np.median(seg_tier)
                            max_tier = np.max(seg_tier)

                            stats_data = {
                                "Metric": ["Current MMR", "True Skill", "Tier", "Match Count"],
                                "Min": [
                                    f"{np.min(seg_mmr):.2f}", 
                                    f"{np.min(seg_ts):.2f}", 
                                    get_tier_name(min_tier), 
                                    f"{np.min(seg_matches):.0f}"
                                ],
                                "Median": [
                                    f"{np.median(seg_mmr):.2f}", 
                                    f"{np.median(seg_ts):.2f}", 
                                    get_tier_name(med_tier), 
                                    f"{np.median(seg_matches):.1f}"
                                ],
                                "Max": [
                                    f"{np.max(seg_mmr):.2f}", 
                                    f"{np.max(seg_ts):.2f}", 
                                    get_tier_name(max_tier), 
                                    f"{np.max(seg_matches):.0f}"
                                ]
                            }
                            
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(df_stats, use_container_width=True)
                        else:
                            st.warning("선택된 세그먼트에 속한 유저가 없습니다.")
                    
                    # 3. Aggregated First 3 Matches Analysis
                    st.divider()
                    st.markdown(f"#### 초반 3경기 분석 (통합)")
                    
                    if hasattr(sim, 'first_3_outcomes') and target_seg_indices:
                        # Filter users
                        combined_mask = np.isin(sim.segment_indices, target_seg_indices)
                        
                        if combined_mask.any():
                            outcomes = sim.first_3_outcomes[combined_mask]
                            valid_mask = ~np.any(outcomes == -9, axis=1)
                            valid_outcomes = outcomes[valid_mask]
                            total_valid = len(valid_outcomes)
                            
                            if total_valid > 0:
                                patterns = []
                                for row in valid_outcomes:
                                    wins = np.sum(row == 1)
                                    draws = np.sum(row == 0)
                                    losses = np.sum(row == -1)
                                    
                                    parts = []
                                    if wins > 0: parts.append(f"{wins}승")
                                    if draws > 0: parts.append(f"{draws}무")
                                    if losses > 0: parts.append(f"{losses}패")
                                    
                                    if not parts: p_str = "0승 0무 0패"
                                    else: p_str = " ".join(parts)
                                    patterns.append(p_str)
                                
                                pattern_counts = pd.Series(patterns).value_counts()
                                df_patterns = pd.DataFrame({
                                    "Pattern": pattern_counts.index,
                                    "Count": pattern_counts.values,
                                    "Percentage": (pattern_counts.values / total_valid * 100).round(1)
                                })
                                df_patterns["Percentage"] = df_patterns["Percentage"].apply(lambda x: f"{x}%")
                                
                                st.dataframe(df_patterns, use_container_width=True)
                            else:
                                st.info("초반 3경기를 완료한 유저가 없습니다.")
                    
                    # 4. Rank History Graph (Sample Users)
                    st.divider()
                    st.markdown("#### 랭크 변동 이력 (Sample Users)")
                    
                    history_data = []
                    for target_idx in target_indices:
                        seg_name = sim.watched_indices[target_idx]
                        logs = sim.match_logs.get(target_idx, [])
                        
                        # Sort logs by day just in case
                        # logs.sort(key=lambda x: x.day) # Assuming they are already sorted
                        
                        for i, log in enumerate(logs):
                            history_data.append({
                                "Match Count": i + 1,
                                "Tier Index": log.current_tier_index,
                                "Tier Name": sim.tier_configs[log.current_tier_index].name if sim.tier_configs and 0 <= log.current_tier_index < len(sim.tier_configs) else "Unranked",
                                "User": f"{seg_name} ({target_idx})"
                            })
                            
                    if history_data:
                        df_history = pd.DataFrame(history_data)
                        
                        fig_history = go.Figure()
                        
                        # Plot each user as a separate trace
                        # We use loop to ensure correct labeling and coloring if needed, 
                        # but px.line is easier for multi-line.
                        # However, for custom Y-axis ticks (Tier Names), we need to be careful.
                        
                        fig_history = px.line(df_history, x="Match Count", y="Tier Index", color="User",
                                              title="Rank Variation History (Tier vs Match Count)",
                                              markers=True,
                                              hover_data=["Tier Name"])
                        
                        # Configure Y-axis to show Tier Names
                        if sim.tier_configs:
                            tier_names = [t.name for t in sim.tier_configs]
                            tick_vals = list(range(len(tier_names)))
                            tick_text = tier_names
                            
                            # Add Unranked if present in data (usually -1)
                            # If we have -1 in data, we should include it in ticks
                            if df_history["Tier Index"].min() == -1:
                                tick_vals.insert(0, -1)
                                tick_text.insert(0, "Unranked")
                                
                            fig_history.update_layout(
                                yaxis=dict(
                                    tickmode='array',
                                    tickvals=tick_vals,
                                    ticktext=tick_text,
                                    title="Tier"
                                )
                            )
                        
                        st.plotly_chart(fig_history, use_container_width=True)
                    else:
                        st.info("표시할 이력 데이터가 없습니다.")
            else:
                st.info("세그먼트를 선택하세요.")
        else:
            st.info("매치 기록을 보려면 먼저 시뮬레이션을 실행하세요.")

    with tab4:
        st.subheader("랭크 시스템 분석")
        if st.session_state.simulation and st.session_state.simulation.tier_configs:
            sim = st.session_state.simulation
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 승급/강등 현황")
                # Prepare data
                prom_data = []
                for t_idx, count in sim.promotion_counts.items():
                    if t_idx < len(sim.tier_configs):
                        prom_data.append({"Tier": sim.tier_configs[t_idx].name, "Count": count, "Type": "Promotion"})
                
                dem_data = []
                for t_idx, count in sim.demotion_counts.items():
                    if t_idx < len(sim.tier_configs):
                        dem_data.append({"Tier": sim.tier_configs[t_idx].name, "Count": count, "Type": "Demotion"})
                
                df_trans = pd.DataFrame(prom_data + dem_data)
                if not df_trans.empty:
                    fig_trans = px.bar(df_trans, x="Tier", y="Count", color="Type", barmode="group", title="티어별 승급/강등 횟수")
                    st.plotly_chart(fig_trans, use_container_width=True)
                else:
                    st.info("아직 승급/강등 데이터가 없습니다.")
            
            with col2:
                st.markdown("#### 티어별 도달 난이도 (예상)")
                # This is hard to calculate exactly without full history for everyone.
                # But we can show current distribution as a proxy for difficulty?
                # Or show "Average Matches to Promote" if we tracked it.
                # Since we didn't track "Matches to Reach" globally yet, let's show Demotion Rate?
                
                # Demotion Rate = Demotions / (Promotions + Demotions + Stays) ?
                # Or just Demotions / Total Users in Tier?
                
                rates = []
                for i, config in enumerate(sim.tier_configs):
                    user_count = np.sum(sim.user_tier_index == i)
                    dems = sim.demotion_counts.get(i, 0)
                    proms = sim.promotion_counts.get(i+1, 0) # Promoted TO next tier (from this tier)
                    
                    # Total events
                    # This is cumulative over time, user_count is snapshot.
                    # Ratio might be misleading.
                    # Let's just show raw counts table.
                    rates.append({
                        "Tier": config.name,
                        "Current Users": user_count,
                        "Total Promotions (Out)": proms,
                        "Total Demotions (In)": dems
                    })
                    
                st.dataframe(pd.DataFrame(rates), use_container_width=True)
                
        else:
            st.info("랭크 분석을 보려면 시뮬레이션을 실행하고 티어 설정을 완료하세요.")

        # --- Divergence Analysis (New Request) ---
        st.divider()
        st.markdown("#### 티어별 실력 안착 분석 (Divergence Analysis)")
        st.caption("각 티어의 목표 MMR 중간값과 실제 해당 티어에 안착한 유저들의 실력(True Skill) 중간값의 차이를 분석합니다.")
        
        if st.session_state.simulation and st.session_state.simulation.tier_configs:
            sim = st.session_state.simulation
            div_data = []
            
            for i, config in enumerate(sim.tier_configs):
                # 1. Target Median
                if config.max_mmr >= 9999:
                    target_median = config.min_mmr + 100 # Arbitrary buffer for infinite top tier
                else:
                    target_median = (config.min_mmr + config.max_mmr) / 2
                
                # 2. Actual Users
                indices = np.where(sim.user_tier_index == i)[0]
                
                if len(indices) > 0:
                    actual_median_ts = np.median(sim.true_skill[indices])
                    actual_avg_ts = np.mean(sim.true_skill[indices])
                    divergence = actual_median_ts - target_median
                    user_count = len(indices)
                else:
                    actual_median_ts = 0
                    actual_avg_ts = 0
                    divergence = 0
                    user_count = 0
                
                div_data.append({
                    "Tier": config.name,
                    "Target Median MMR": f"{target_median:.0f}",
                    "Actual Median TS": f"{actual_median_ts:.0f}",
                    "Divergence": f"{divergence:+.0f}",
                    "User Count": user_count,
                    "_divergence_val": divergence # For sorting/coloring if needed
                })
            
            df_div = pd.DataFrame(div_data)
            
            # Display Table
            st.dataframe(
                df_div,
                column_config={
                    "_divergence_val": None # Hide helper column
                },
                use_container_width=True
            )
            
            # Chart
            fig_div = px.bar(df_div, x="Tier", y="_divergence_val", 
                             title="티어별 실력 괴리도 (양수: 실력이 더 높음 / 음수: 실력이 더 낮음)",
                             labels={"_divergence_val": "Divergence (True Skill - Target MMR)"})
            st.plotly_chart(fig_div, use_container_width=True)

    # --- Comments Section ---
    st.divider()
    st.subheader("Comment")
    comments = st.text_area("메모용:", value=st.session_state.get("user_comments", ""), height=500)
    if st.button("Save"):
        st.session_state.user_comments = comments
        save_config(st.session_state.get("username"))
        st.success("코멘트가 저장되었습니다!")
