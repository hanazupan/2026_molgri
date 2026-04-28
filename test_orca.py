from rules.orca.smk import OrcaReader

reader = OrcaReader("~/MA/2026_molgri/nobackup/benzene_benzene/test")

# Energie
energy = reader.extract_energy_orca_output()
print("Energy:", energy)

# Laufzeit
time = reader.extract_time_orca_output()
print("Time:", time)

# Check ob fertig
print("Finished:", reader.assert_normal_finish(False))