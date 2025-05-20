from pathlib import Path
import sys
import logging
import pandas as pd
from collections import defaultdict
from itertools import combinations

# Require at least one argument (file/wildcard)
if len(sys.argv) < 2:
    print(
        "Error: You must specify at least one wildcard or file path for the Excel files.",
        file=sys.stderr,
    )
    print("Usage: python analyze.py '<wildcard1>' '<wildcard2>' ...", file=sys.stderr)
    sys.exit(1)

# Collect all files from all arguments
all_files = set()
for arg in sys.argv[1:]:
    matched = list(Path().glob(arg))
    if not matched:
        print(f"Warning: No files matched the pattern: {arg}", file=sys.stderr)
    all_files.update(matched)

if not all_files:
    print("Error: No files matched any of the provided patterns.", file=sys.stderr)
    sys.exit(1)

files = sorted(all_files)

# Map: filename -> set of entity_names
entities_by_file = {}

for file in files:
    try:
        df = pd.read_excel(file, sheet_name="Entities")
        if "entity_name" in df.columns:
            entities = set(df["entity_name"].dropna().astype(str))
            entities_by_file[file.name] = entities
        else:
            print(f"Warning: 'entity_name' column not found in {file}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# All entity names across all files
all_entities = set.union(*entities_by_file.values()) if entities_by_file else set()

# Entities present in all files
common_entities = set.intersection(*entities_by_file.values()) if entities_by_file else set()

# Entities unique to a single file
unique_entities = defaultdict(list)
for entity in all_entities:
    present_in = [fname for fname, ents in entities_by_file.items() if entity in ents]
    if len(present_in) == 1:
        unique_entities[present_in[0]].append(entity)

# Entities in each subset of files (excluding all and unique)
subset_entities = defaultdict(list)
file_list = list(entities_by_file.keys())
for r in range(2, len(file_list)):
    for combo in combinations(file_list, r):
        intersect = set.intersection(*(entities_by_file[f] for f in combo))
        # Remove those in all files or already counted in smaller combos
        intersect = intersect - common_entities
        for smaller_r in range(1, r):
            for smaller_combo in combinations(combo, smaller_r):
                intersect = intersect - set.intersection(
                    *(entities_by_file[f] for f in smaller_combo)
                )
        if intersect:
            subset_entities[combo] = list(intersect)

# Print results in markdown
print("# Entity Name Comparison\n")

print(f"## Files compared:\n")
for fname in file_list:
    print(f"- {fname}")

print(f"\n## Entities present in all files ({len(common_entities)})\n")
for entity in sorted(common_entities):
    print(f"- {entity}")

print(f"\n## Entities unique to a single file\n")
for fname, ents in unique_entities.items():
    print(f"### {fname} ({len(ents)})")
    for entity in sorted(ents):
        print(f"- {entity}")
    print()

print(f"\n## Entities present in subsets of files\n")
for combo, ents in subset_entities.items():
    print(f"### {', '.join(combo)} ({len(ents)})")
    for entity in sorted(ents):
        print(f"- {entity}")
    print()
