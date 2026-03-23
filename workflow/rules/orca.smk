from workflow.helpers.orca_reader import QuantumSetup, OrcaReader, QuantumMolecule, OrcaWriter, read_important_stuff_into_csv, nice_str_of, split_xyz_trajectory

from pathlib import Path

BASE_DIR = Path("/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test")
FRAMES = glob_wildcards(
    "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/{frame}/structure.out"
).frame
OUT_PATH = BASE_DIR / "output.csv"
BASE = "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test"

setup = QuantumSetup(
    functional="PBE0",
    basis_set="def2-TZVP",
    solvent=None,
    dispersion_correction="D3",
    num_scf=None,
    num_cores=None,        # ← IMPORTANT
    ram_per_core=None      # ← IMPORTANT
)


checkpoint split_trajectory:
    input:
        xyz="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/trajectory.xyz"
    output:
        flag="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/split_done.txt"
    run:
        split_xyz_trajectory(
            xyz_file=input.xyz,
            output_base_dir="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test",
            structures_per_chunk=1
        )

        with open(output.flag, "w") as f:
            f.write("done")
        # just create a flag so Snakemake knows we're done
        #Path(output[0]).mkdir(exist_ok=True)

def get_frames(wildcards):
    checkpoints.split_trajectory.get()  # ensures checkpoint runs first

    base_dir = Path("/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test")

    return [
        p.name for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]


rule write_orca_input:
    input:
        xyz=lambda wc: f"{BASE}/{wc.frame}/structure.xyz"
    output:
        inp=f"{BASE}/{{frame}}/structure.inp"
    params:
        setup=setup
    run:
        molecule = QuantumMolecule(
            xyz_file=input.xyz,
            charge=0,
            multiplicity=1
        )

        writer = OrcaWriter(molecule, params.setup)
        writer.make_input(geo_optimization=False)
        writer.write_to_file(output.inp)


rule run_orca:
    input:
        inp=f"{BASE}/{{frame}}/structure.inp"
    output:
        out=f"{BASE}/{{frame}}/structure.out"
    log:
        f"{BASE}/{{frame}}/orca.log"
    resources:
        orca=1
    shell:
        """
        cd $(dirname {input.inp})
        orca --replace structure.inp > {log} 2>&1
        """

rule read_orca:
    input:
        orca_out="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/structure.out"

    run:
        reader = OrcaReader(input.orca_out)

        energy = reader.extract_energies_orca_output()
        print("Energy:", energy)

        time = reader.extract_time_orca_output()
        print("Time:", time)

        print("Finished:", reader.assert_normal_finish(False))

rule write_csv:
    input:
        expand(
            "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/{frame}/structure.out",
            frame=FRAMES
        )
    output:
        csv="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/output.csv"
    params:
        setup=setup
    run:
        read_important_stuff_into_csv(
            out_files_to_read=input,
            csv_file_to_write=output.csv,
            setup=params.setup,
            num_points=len(input)
        )