
import json
import pandas as pd
from config import MAX_ROWS

def load_jsonl(file_path, max_rows=MAX_ROWS):
    """Load JSONL file into pandas DataFrame"""
    print(f"[LOAD] Loading: {file_path}")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    print(f"[WARN] Stopped at {max_rows} rows")
                    break
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(data)
        print(f"[DONE] Loaded {len(df)} rows, {len(df.columns)} columns")
        if not df.empty:
            print(f"[INFO] Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        print("[HINT] Check if files exist at exact paths")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return pd.DataFrame()
