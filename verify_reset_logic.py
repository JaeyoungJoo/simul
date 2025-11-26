import pandas as pd
import numpy as np

def test_logic():
    print("--- Testing Season Reset Logic ---")
    
    # 1. Simulate Session State Data (Potential String Types)
    data = [
        {"tier_name": "Silver", "min_mmr": "1200", "reset_mmr": "1100", "soft_reset_ratio": "0.5"},
        {"tier_name": "Bronze", "min_mmr": "0", "reset_mmr": "800", "soft_reset_ratio": "0.8"},
        {"tier_name": "Gold", "min_mmr": "1600", "reset_mmr": "1400", "soft_reset_ratio": "0.4"}
    ]
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # 2. Simulate Logic in app.py (Before Fix)
    print("\n[Simulation] Sorting by 'min_mmr' (String Sort check)")
    # Note: If min_mmr is string, "1200" < "1600" but "0" < "1200". 
    # Wait, "1200" < "1600" is True. "0" < "1200" is True.
    # But "10000" < "200" is True in string sort.
    
    # Let's try to reproduce a sorting issue
    data_bad = [
        {"tier_name": "High", "min_mmr": "10000", "reset_mmr": "5000", "soft_reset_ratio": "0.5"},
        {"tier_name": "Low", "min_mmr": "200", "reset_mmr": "100", "soft_reset_ratio": "0.5"}
    ]
    df_bad = pd.DataFrame(data_bad)
    sorted_bad = df_bad.sort_values("min_mmr")
    print("\nBad Sort Result (String):")
    print(sorted_bad[["tier_name", "min_mmr"]])
    
    # 3. Simulate Rule Construction
    print("\n[Simulation] Constructing Rules from 'df' (assuming sorted correctly for now)")
    # Correct sort for the main test
    df["min_mmr"] = df["min_mmr"].astype(float)
    df = df.sort_values("min_mmr")
    
    rules = []
    for i in range(len(df)):
        row = df.iloc[i]
        min_val = float(row["min_mmr"])
        if i < len(df) - 1:
            max_val = float(df.iloc[i+1]["min_mmr"])
        else:
            max_val = float('inf')
        
        rules.append({
            "min": min_val,
            "max": max_val,
            "target": float(row["reset_mmr"]),
            "compression": float(row["soft_reset_ratio"])
        })
        print(f"Rule {i}: Min {min_val} -> Max {max_val} | Target {row['reset_mmr']} | Comp {row['soft_reset_ratio']}")

    # 4. Simulate Application on Users
    print("\n[Simulation] Applying to Users")
    user_mmrs = np.array([500, 1100, 1300, 1700, 2000])
    print(f"User MMRs: {user_mmrs}")
    
    new_mmrs = user_mmrs.copy()
    
    for rule in rules:
        min_val = rule['min']
        max_val = rule['max']
        target = rule['target']
        comp = rule['compression']
        
        mask = (user_mmrs >= min_val) & (user_mmrs < max_val)
        if np.any(mask):
            print(f"Applying Rule ({min_val}-{max_val}) to: {user_mmrs[mask]}")
            # Formula: target + (old - target) * comp
            new_mmrs[mask] = target + (user_mmrs[mask] - target) * comp
            
    print(f"New MMRs:  {new_mmrs}")
    
    # Check specific values
    # 500 (Bronze): 0-1200. Target 800. Comp 0.8.
    # Expected: 800 + (500 - 800) * 0.8 = 800 + (-300)*0.8 = 800 - 240 = 560.
    print(f"Calc for 500: 800 + (500-800)*0.8 = {800 + (500-800)*0.8}")
    
    # 1300 (Silver): 1200-1600. Target 1100. Comp 0.5.
    # Expected: 1100 + (1300 - 1100) * 0.5 = 1100 + 200*0.5 = 1200.
    print(f"Calc for 1300: 1100 + (1300-1100)*0.5 = {1100 + (1300-1100)*0.5}")

if __name__ == "__main__":
    test_logic()
