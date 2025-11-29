import pandas as pd
import numpy as np

def test_pandas_at():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    print("Original DF:")
    print(df)
    
    try:
        # Try setting a new column value with .at
        # This usually raises KeyError if column doesn't exist
        df.at[0, 'C'] = 5
        print("\nAfter .at (Success):")
        print(df)
    except Exception as e:
        print(f"\n.at Failed: {e}")
        
    # Try with .loc
    try:
        df.loc[0, 'D'] = 6
        print("\nAfter .loc (Success):")
        print(df)
    except Exception as e:
        print(f"\n.loc Failed: {e}")

    # Try explicit creation
    if 'E' not in df.columns:
        df['E'] = None
    df.at[0, 'E'] = 7
    print("\nAfter explicit creation + .at:")
    print(df)

if __name__ == "__main__":
    test_pandas_at()
