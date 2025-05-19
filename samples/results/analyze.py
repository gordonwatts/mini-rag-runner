from pathlib import Path
import sys
import logging
import pandas as pd
from collections import defaultdict
from itertools import combinations

# Require a wildcard argument
if len(sys.argv) < 2:
    print("Error: You must specify a wildcard pattern for the Excel files.", file=sys.stderr)
    print("Usage: python analyze.py '<wildcard>'", file=sys.stderr)
    sys.exit(1)

wildcard = sys.argv[1]

# Use pathlib to resolve files
files = list(Path().glob(wildcard))

if not files:
    print(f"Error: No files matched the pattern: {wildcard}", file=sys.stderr)
    sys.exit(1)

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

print("\n## Entities present in all files\n")
for entity in sorted(common_entities):
    print(f"- {entity}")

print("\n## Entities unique to a single file\n")
for fname, ents in unique_entities.items():
    print(f"### {fname}")
    for entity in sorted(ents):
        print(f"- {entity}")
    print()

print("\n## Entities present in subsets of files\n")
for combo, ents in subset_entities.items():
    print(f"### {', '.join(combo)}")
    for entity in sorted(ents):
        print(f"- {entity}")
    print()
