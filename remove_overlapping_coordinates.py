import numpy as np

input_file = "trajectory.xyz"
output_file = "trajectory.xyz"

# tolerance for coordinate comparison
tol = 1e-6

def has_duplicates(coords, tol):
    """
    Checks for duplicate coordinates within tolerance.
    Returns number of overlapping coordinates.
    """
    duplicates = 0
    n = len(coords)

    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(coords[i], coords[j], atol=tol):
                duplicates += 1

    return duplicates


structures = []
removed_structures = []

with open(input_file) as f:
    lines = f.readlines()

i = 0
structure_index = 1
cleaned_lines = []

while i < len(lines):

    n_atoms = int(lines[i].strip())
    comment = lines[i+1]

    atom_lines = lines[i+2:i+2+n_atoms]

    coords = []
    parsed_atoms = []

    for line in atom_lines:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])

        coords.append([x, y, z])
        parsed_atoms.append(line)

    coords = np.array(coords)

    dup_count = has_duplicates(coords, tol)

    if dup_count > 0:
        print(f"Struktur {structure_index} entfernt: {dup_count} überlappende Koordinaten")
        removed_structures.append(structure_index)
    else:
        cleaned_lines.append(f"{n_atoms}\n")
        cleaned_lines.append(comment)
        cleaned_lines.extend(parsed_atoms)

    structure_index += 1
    i += n_atoms + 2


with open(output_file, "w") as f:
    f.writelines(cleaned_lines)

print("\nFertig.")
print(f"Entfernte Strukturen: {len(removed_structures)}")
