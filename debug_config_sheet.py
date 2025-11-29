import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import json

# Mock streamlit secrets if needed, but running via `streamlit run` should work if secrets.toml is present.
# Or we can just use the connection if running as a streamlit app.

def debug_sheet():
    print("--- Starting Debug ---")
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        print("Connection object created.")
        
        # Read Config
        print("Reading 'Config' worksheet...")
        df = conn.read(worksheet="Config", ttl=0)
        print("Read complete.")
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Normalize columns
        df.columns = [str(c).strip() for c in df.columns]
        print(f"Normalized Columns: {df.columns.tolist()}")
        
        if "username" in df.columns:
            # Normalize usernames
            df["username_norm"] = df["username"].astype(str).str.strip().str.lower()
            usernames = df["username_norm"].unique().tolist()
            print(f"Usernames found: {usernames}")
            
            # Check specific users
            for target in ["admin", "test1"]:
                print(f"Checking for '{target}'...")
                user_rows = df[df["username_norm"] == target]
                if not user_rows.empty:
                    print(f"  Found {len(user_rows)} row(s).")
                    json_str = user_rows.iloc[0].get("ConfigJSON", "MISSING")
                    print(f"  ConfigJSON length: {len(str(json_str))}")
                    print(f"  ConfigJSON preview: {str(json_str)[:100]}...")
                    try:
                        parsed = json.loads(json_str)
                        print("  JSON Parse: SUCCESS")
                        print(f"  Keys: {list(parsed.keys())}")
                    except Exception as e:
                        print(f"  JSON Parse: FAILED ({e})")
                else:
                    print(f"  NOT FOUND.")
        else:
            print("ERROR: 'username' column missing!")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
    print("--- End Debug ---")

if __name__ == "__main__":
    debug_sheet()
