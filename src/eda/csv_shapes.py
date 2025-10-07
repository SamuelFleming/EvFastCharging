from pathlib import Path
import pandas as pd

# Paths
folder_path = Path("data/processed")          # safer than 'data\processed'
out_file = folder_path / "csv-summaries.txt"  # write a single text file

parts = []

for csv_path in folder_path.glob("*.csv"):
    print(f"Processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        parts.append(f"\n\n=== {csv_path.name} ===\nPath: {csv_path}\nFAILED to read: {e}")
        continue

    # Build a human-readable section per file
    section = []
    section.append(f"\n\n=== {csv_path.name} ===")
    section.append(f"Path: {csv_path}")
    section.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    section.append("Columns: " + ", ".join(map(str, df.columns)))

    # Head (first 5 rows)
    try:
        section.append("\nHead (first 5 rows):\n" + df.head().to_string(index=False))
    except Exception as e:
        section.append(f"\nHead (error rendering): {e}")

    # Describe numeric columns (if any)
    try:
        desc_num = df.describe(include="number")
        if not desc_num.empty:
            section.append("\nDescribe (numeric):\n" + desc_num.to_string())
    except Exception:
        pass

    # Describe object columns (if any)
    try:
        desc_obj = df.describe(include="object")
        if not desc_obj.empty:
            section.append("\nDescribe (object):\n" + desc_obj.to_string())
    except Exception:
        pass

    parts.append("\n".join(section))

# Write the summary file
out_file.write_text("".join(parts), encoding="utf-8")
print(f"Wrote to file {out_file}")