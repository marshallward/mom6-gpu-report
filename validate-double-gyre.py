#!/usr/bin/env python3
import sys
import re

# Hardcoded reference values (full precision)
reference_energies = [
    1.4237499351110218E-13,
    4.5804317899616613E-06,
    8.9582059515487157E-06,
    1.1667077649757350E-05,
    1.5422955375147519E-05,
    1.8260222694485033E-05,
    2.3974311229885696E-05,
    2.9279621461764738E-05,
    3.4864407126080987E-05,
    4.1955651758640954E-05,
    4.6047288874347594E-05
]

def extract_energy_mass(filename):
    pattern = re.compile(r"En\s+([+-]?\d+\.\d+E[+-]]?\d+)")
    energies = []

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                energies.append(match.group(1).strip())

    return energies

def main():
    if len(sys.argv) != 2:
        print("Usage: ./validate_energy_strict.py ocean.stats")
        sys.exit(1)

    filename = sys.argv[1]
    extracted = extract_energy_mass(filename)

    ref_str = [f"{v:.16E}" for v in reference_energies]

    if len(extracted) != len(ref_str):
        print(f"Mismatch in number of entries: {len(extracted)} vs {len(ref_str)}")
        sys.exit(1)

    all_good = True
    for i, (val, ref) in enumerate(zip(extracted, ref_str)):
        if val != ref:
            print(f"Mismatch at step {i}: got {val}, expected {ref}")
            all_good = False

    if all_good:
        print("All energy values match exactly.")

if __name__ == "__main__":
    main()
