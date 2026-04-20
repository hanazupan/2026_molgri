import os
import re
import subprocess
from subprocess import PIPE, run
import MDAnalysis as mda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from pathlib import Path
from ase.io import read
from datetime import timedelta
from workflow.helpers.io import read_object

HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]

def xtc_to_xyz(xtc_file, gro_file, output_xyz):
    u = mda.Universe(gro_file, xtc_file)

    with open(output_xyz, "w") as f:
        #for i, ts in enumerate(u.trajectory):
        for i, ts in enumerate(u.trajectory):
            n = len(u.atoms)

            f.write(f"{n}\n")
            f.write(f"frame{i}\n")

            for atom in u.atoms:
                x, y, z = atom.position
                element = atom.name[0]
                f.write(f"{element:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def find_invalid_frames_with_overlapping_atoms(
    trajectory_path: str,
    rounding_decimals: int = 6
):
    """
    Detect frames where at least two atoms have identical coordinates.

    Args:
        trajectory_path (str): Path to trajectory.xyz
        rounding_decimals (int): Precision for coordinate comparison

    Returns:
        valid_indices (np.ndarray): frames WITHOUT overlaps
        invalid_indices (np.ndarray): frames WITH overlapping atoms
    """

    traj = read(trajectory_path, index=":")

    valid_indices = []
    invalid_indices = []

    for i, atoms in enumerate(traj):
        coords = atoms.get_positions().round(rounding_decimals)

        # convert each atom position into a tuple
        coord_tuples = [tuple(c) for c in coords]

        # check if duplicates exist within the frame
        if len(coord_tuples) != len(set(coord_tuples)):
            invalid_indices.append(i)
        else:
            valid_indices.append(i)

    print(f"Total frames: {len(traj)}")
    print(f"Valid frames: {len(valid_indices)}")
    print(f"Invalid frames (overlapping atoms): {len(invalid_indices)}")

    return np.array(valid_indices), np.array(invalid_indices)


class QuantumSetup:

    """
    Just a simple class to collect variables connected to QM calculation set-up (that are not molecule-specific).
    """

    def __init__(self, functional: str, basis_set: str, solvent: str = None, mini: str = True, dispersion_correction: str = "",
                 num_scf: int = 15, num_cores: int = None, ram_per_core: int = None):
        self.functional = functional
        self.basis_set = basis_set
        self.solvent = solvent
        self.mini = mini
        self.dispersion_correction = dispersion_correction
        self.num_scf = num_scf
        self.num_cores = num_cores
        self.ram_per_core = ram_per_core

    def get_dir_name(self):
        """
        Calculations of this quantum set-up can be done in a folder that specifies the major settings.
        """
        return f"{nice_str_of(self.functional)}_{nice_str_of(self.basis_set)}_{nice_str_of(self.solvent)}_{nice_str_of(self.dispersion_correction)}/"


class OrcaReader:

    """
    Does not read .inp, but .out files
    """

    def __init__(self, out_file_path: str, is_multi_out: bool = False):
        self.out_file_path = os.path.expanduser(out_file_path)
        path = os.path.normpath(self.out_file_path)
        split_path = path.split(os.sep)
        self.frame_num = split_path[-2]
        self.calculation_directory = os.path.join(*split_path[:-1])
        self.is_multi_out = is_multi_out

    def assert_normal_finish(self, throw_error=True):
        """
        Make sure that the orca calculation finished normally.

        Args:
            throw_error (bool): if True, raise an error, if False, print a warning

        Either throws an error or prints a warning.
        """
        returncode = subprocess.run(f"""grep -q "****ORCA TERMINATED NORMALLY****" {self.out_file_path}""",
                       capture_output=True, shell=True).returncode

        if returncode != 0 and throw_error:
            raise ChildProcessError(f"Orca did not terminate normally; see {self.out_file_path }")
        elif returncode != 0 and not throw_error:
            return False
        return True

    def assert_optimization_complete(self, throw_error=True):
        """
        Make sure that the optimization is complete (not the same as normal finish!). Only relevant for optimizations.

        Args:
            throw_error (bool): if True, raise an error, if False, print a warning

        Either throws an error or prints a warning.
        """
        message_has_converged = """
***********************HURRAY********************
***        THE OPTIMIZATION HAS CONVERGED     ***
*************************************************
    """

        command = ['grep', f'"{message_has_converged}"', f"{self.out_file_path}"]
        returncode = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True).returncode
        if returncode != 0 and throw_error:
            raise ChildProcessError(f"Orca may have finished normally but did not converge; see {self.out_file_path}")
        elif returncode != 0 and not throw_error:
            print("Opt not complete")
            return False
        # also fail optimization if the calculation failed
        elif not self.assert_normal_finish(throw_error=False):
            print("No normal finish")
            return False
        return True

    def extract_time_orca_output(self) -> pd.Timedelta:
        """
        Take any orca output file and give me the time needed for the calculation.

        Args:
            output_file (str): path to the .out file

        Returns:
            Time as days hours:min:sec
        """
        line_time = subprocess.run(f"""grep "^TOTAL RUN TIME:" {self.out_file_path} | sed 's/^TOTAL RUN TIME: //'""",
                                   capture_output=True, text=True, shell=True)
        try:
            line_time = line_time.stdout.strip()
            line_time= line_time.replace("msec", "ms")
            time_h_m_s = line_time
        except AttributeError:
            time_h_m_s = 0

        return time_h_m_s


    def extract_energies_orca_output(self) -> list:
        """
        Extract all FINAL SINGLE POINT ENERGY values from ORCA output.

        Returns:
            List of energies in Hartree
        """
        result = subprocess.run(
            f'grep "^FINAL SINGLE POINT ENERGY" {self.out_file_path}',
            shell=True,
            capture_output=True,
            text=True
        )

        energies = []

        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    energy = float(line.split()[-1])  # letzter Wert in der Zeile
                    energies.append(energy)
                except ValueError:
                    energies.append(np.NaN)

        return energies


    def get_frame_num(self):
        return int(self.frame_num)


def read_important_stuff_into_csv(out_files_to_read: list, csv_file_to_write: str, setup: QuantumSetup,  chunksize: int,
                                  num_points: int, is_pt=True):
    """
    Read a list of orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and generation times. Times can optionally be read from the benchmark files

    Args:
        out_files_to_read (list): a list of paths, usually to a number of .out files calculated along a molgri pt
        csv_file_to_write (str): a path to a csv file where the data will be recorded.
        setup ():

    Returns:

    """

    all_df = []
    all_frame_indices = [int(Path(out_file).parts[-2]) for out_file in out_files_to_read]

    import re

    def get_frame_number(path: str) -> int:
        # adapt regex to your folder structure
        # e.g. ".../10/structure.out" -> 10
        return int(re.search(r"/(\d+)/", path).group(1))

    for out_file in sorted(out_files_to_read, key=get_frame_number):
        my_reader = OrcaReader(out_file)

    # for out_file_to_read in out_files_to_read:
    #     my_reader = OrcaReader(out_file_to_read)

        frame_index = my_reader.get_frame_num()
        time_h_m_s = my_reader.extract_time_orca_output()
        normal_finish = my_reader.assert_normal_finish(throw_error=False)
        optimization_complete = my_reader.assert_optimization_complete(throw_error=False)

        base_path = "/home/nadjar02/MA/2026_molgri/nobackup"
        short_path = out_file.replace(base_path, "")

        energies = my_reader.extract_energies_orca_output()

        rows = []
        for step, energy in enumerate(energies):
            energy_kjmol = energy * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0
            global_index = chunksize * (frame_index-1) + step

            rows.append([
                short_path,
                frame_index,
                step,
                global_index,
                energy,
                energy_kjmol,
                normal_finish,
                optimization_complete,
                setup.functional,
                setup.basis_set,
                setup.dispersion_correction,
                setup.solvent,
                time_h_m_s
            ])

        df = pd.DataFrame(rows, columns=[
            "File",
            "Frame",
            "Step",
            "Global_index",
            "Energy [hartree]",
            "Energy [kJ/mol]",
            "Normal Finish",
            "Optimization Complete",
            "Functional",
            "Basis set",
            "Dispersion correction",
            "Solvent",
            "Time [h:m:s]"
        ])

        all_df.append(df)

    combined_df = pd.concat(all_df)
    try:
        combined_df["Time [h:m:s]"] = pd.to_timedelta(combined_df["Time [h:m:s]"])
        combined_df["Time [s]"] = np.where(combined_df["Normal Finish"], combined_df["Time [h:m:s]"].dt.total_seconds(), np.NaN)
    except:
        # THIS DELTA TIME THING IS A HEADACHE!!!!
        pass
    combined_df.to_csv(csv_file_to_write, index=False)


def read_times_into_txt(out_files_to_read: list, txt_file_to_write: str, setup: QuantumSetup):
    all_times_seconds = []
    #my_setup = QuantumSetup(set_up=QuantumSetup)

    def get_frame_number(path: str) -> int:
        # adapt regex to your folder structure
        # e.g. ".../10/structure.out" -> 10
        return int(re.search(r"/(\d+)/", path).group(1))

    all_energies = []
    for out_file in sorted(out_files_to_read, key=get_frame_number):
        my_reader = OrcaReader(out_file)
        # energies = reader.extract_energies_orca_output()
        # all_energies.extend(energies)

    # for out_file_to_read in out_files_to_read:
    #     my_reader = OrcaReader(out_file_to_read)

        try:
            time_h_m_s = my_reader.extract_time_orca_output()
            energies = my_reader.extract_energies_orca_output()

            if not time_h_m_s:
                raise ValueError("Empty time string")

            td = pd.to_timedelta(time_h_m_s, errors="coerce")

            if pd.isna(td):
                raise ValueError(f"Invalid timedelta: {time_h_m_s}")

            seconds = td.total_seconds()

            if pd.isna(seconds):
                raise ValueError("NaN seconds")

            all_times_seconds.append(seconds)

        except Exception as e:
            print(f"Skipping invalid file: {out_file_to_read} ({e})")

    def format_hms(seconds):
        return str(timedelta(seconds=int(seconds)))

    with open(txt_file_to_write, "w") as f:
        if all_times_seconds:
            # Stats
            longest = max(all_times_seconds)
            shortest = min(all_times_seconds)
            average = sum(all_times_seconds) / len(all_times_seconds)
            time_per_structure = average / len(energies)

            f.write("Summary:\n")
            f.write(f"Longest:  {format_hms(longest)}\n")
            f.write(f"Shortest: {format_hms(shortest)}\n")
            f.write(f"Average:  {format_hms(average)}\n\n")

            f.write(f"Number of structures:  {len(energies)}\n")
            f.write(f"Average time per structure [s]:  {time_per_structure}\n")
            f.write(f"Average time per structure [h:m:s]:  {format_hms(time_per_structure)}\n\n")

            f.write(f"Number of cores: {setup.num_cores}\n")
            f.write(f"Ram per core: {setup.ram_per_core}\n\n")

            # Write individual times
            f.write("Individual times [h:m:s]:\n")
            for t in all_times_seconds:
                f.write(f"{format_hms(t)}\n")

        else:
            f.write("No valid times found.\n")


def extract_frame_indices_from_xyz(trajectory_path: str):
    """
    Extract frame indices from raw XYZ file by reading comment lines manually.
    """

    frame_indices = []

    with open(trajectory_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())   # first line
        comment = lines[i + 1].strip()      # second line

        # 🔍 extract something like "frame12"
        match = re.search(r"(\d+)", comment)

        if match:
            frame_indices.append(int(match.group(1)))
        else:
            raise ValueError(f"Could not parse frame index from comment: '{comment}'")

        # jump to next frame
        i += num_atoms + 2

#    print("Extracted frame indices:", frame_indices)
    return frame_indices

def filter_frame_indices(frame_indices, invalid_indices):
    invalid_indices = np.array(invalid_indices, dtype=int)
    mask = np.ones(len(frame_indices), dtype=bool)
    mask[invalid_indices] = False
    #print("Filtered frame indices:", list(np.array(frame_indices)[mask]))
    return list(np.array(frame_indices)[mask])

def write_energies_with_indices(
    out_files_to_read: list,
    trajectory_path: str,
    csv_file_to_write: str
):
    """
    Create CSV with:
    - frame indices from trajectory.xyz
    - energies in Hartree
    - invalid frames skipped (based on overlapping atoms)

    Assumes:
    - number of energies == number of trajectory frames
    """
    import pandas as pd
    import re
    # 🔹 1. get frame indices
    frame_indices = extract_frame_indices_from_xyz(trajectory_path)

    # 🔹 2. get invalid indices (trajectory positions)
    _, invalid = find_invalid_frames_with_overlapping_atoms(trajectory_path)
    invalid_set = set(invalid)
    frame_indices = filter_frame_indices(frame_indices, invalid)
    # print(frame_indices)

    def get_frame_number(path: str) -> int:
        # adapt regex to your folder structure
        # e.g. ".../10/structure.out" -> 10
        return int(re.search(r"/(\d+)/", path).group(1))

    all_energies = []
    for out_file in sorted(out_files_to_read, key=get_frame_number):
        reader = OrcaReader(out_file)
        energies = reader.extract_energies_orca_output()
        all_energies.extend(energies)

    # 🔹 3. collect all energies (flatten)
    # all_energies = []
    # for out_file in sorted(out_files_to_read):
    #     reader = OrcaReader(out_file)
    #     energies = reader.extract_energies_orca_output()
    #     all_energies.extend(energies)

    # 🔥 sanity check BEFORE filtering
    if len(all_energies) != len(frame_indices):
        raise ValueError(
            f"Mismatch: {len(all_energies)} energies vs {len(frame_indices)} frames"
        )

    # 🔹 4. build rows (skip invalid frames ONLY here)
    rows = []
    for i, (frame_idx, energy) in enumerate(zip(frame_indices, all_energies)):
        energy_kjmol = energy * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0
        rows.append([
            frame_idx,
            energy_kjmol   # converted to kJ/mol
        ])

    # 🔹 5. write CSV
    df = pd.DataFrame(rows, columns=["Total index", "Energy [kJ/mol]"])
    df.to_csv(csv_file_to_write, index=False)

    import pandas as pd

def read_xyzact_dat(dat_files):
    """
    Reads structure.xyzact.dat files.

    Expected format:
        index   energy (Hartree)

    Returns:
        list of (frame_index, energy)
    """
    data = []

    for file in sorted(dat_files):
        with open(file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.split()
                frame_idx = int(float(parts[0]))
                energy = float(parts[1])

                data.append((frame_idx, energy))

    return data

def write_energies_from_xyzact_dat(
        dat_file: str,
        trajectory_path: str,
        csv_file_to_write: str
):
    """
    Create CSV with:
    - frame indices from trajectory.xyz
    - energies from structure.xyzact.dat
    - invalid frames skipped (based on overlapping atoms)
    """

    # 🔹 1. get frame indices from trajectory
    frame_indices = extract_frame_indices_from_xyz(trajectory_path)

    # 🔹 2. detect invalid frames
    _, invalid = find_invalid_frames_with_overlapping_atoms(trajectory_path)
    invalid_set = set(invalid)

    # 🔹 3. filter frame indices
    frame_indices = filter_frame_indices(frame_indices, invalid)

    # 🔹 4. read energies from .dat
    dat_data = read_xyzact_dat(dat_file)

    # split into lists
    dat_indices = [x[0] for x in dat_data]
    dat_energies = [x[1] for x in dat_data]

    # 🔥 sanity check
    if len(dat_energies) != len(frame_indices):
        raise ValueError(
            f"Mismatch: {len(dat_energies)} energies vs {len(frame_indices)} frames"
        )

    # 🔹 5. build rows
    rows = []
    for frame_idx, energy in zip(frame_indices, dat_energies):
        energy_kjmol = energy * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0

        rows.append([
            frame_idx,
            energy_kjmol
        ])

    # 🔹 6. write CSV
    df = pd.DataFrame(rows, columns=["Total index", "Energy [kJ/mol]"])
    df.to_csv(csv_file_to_write, index=False)


def nice_str_of(string: str) -> str:
    """
    Make a string "nice" (ready to use in file names etc) by removing all charcaters that are not alphanumeric.
    Special case: if input is an empty string, the output is "no" because that is easier to include in names.

    Args:
        string (str): input string to be cleaned up

    Returns:
        output string, same as input but without special characters
    """
    if not string:
        return "no"
    return re.sub(r'[^a-zA-Z0-9]', '', string)


class QuantumMolecule:

    """
    Just a simple class to collect variables that define a QM molecule.
    """

    def __init__(self, charge: int, multiplicity: int, xyz_file: str):
        #fragment_1_len: int = None, fragment_2_len: int = None
        self.charge = charge
        self.multiplicity = multiplicity
        # self.fragment_1_len = fragment_1_len
        # self.fragment_2_len = fragment_2_len
        self.xyz_file = xyz_file


class OrcaWriter:

    """
    This class builds orca input files specifically for our typical set-up of two molecules.
    """

    def __init__(self, molecule:  QuantumMolecule, set_up: QuantumSetup):
        self.molecule = molecule
        self.setup = set_up
        with open(self.molecule.xyz_file, "r") as f:
            self.xyz_file_lines = f.readlines()
        self.total_text = ""

    def write_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.total_text)
        self.total_text = ""

    def _write_first_line(self, geo_optimization: bool = False):
        """
        First line looks something like this: ! PBE0 D4 def2-tzvp Opt <- depends on functional, SP/optimization and
        basis set.

        Args:
            geo_optimization (bool): if True option Opt will be selected, else SP
        """
        if geo_optimization:
            optimization_str = "Opt"
        else:
            optimization_str = ""

        self.total_text += f"! {self.setup.functional} {self.setup.dispersion_correction} {self.setup.basis_set} {optimization_str}\n"

    def _write_solvent(self):
        if self.setup.solvent is not None:
            self.total_text += "%CPCM SMD TRUE\n"
            self.total_text += f'SMDSOLVENT "{self.setup.solvent}"\n'
            self.total_text += "END\n"

    def _write_output_minimisation(self):
        if self.setup.mini is True:
            self.total_text += "%output\n"
            self.total_text += "    PrintLevel Mini\n"
            self.total_text += "end\n"

    def _write_resources(self):
        # limit the number of SCF cycles to make hopeless calculations fail quickly
        if self.setup.num_scf is not None:
            self.total_text += f"""
%scf
    MaxIter {self.setup.num_scf}
end\n"""
        if self.setup.num_cores is not None and self.setup.num_cores != "None":
            self.total_text += f"%PAL NPROCS {self.setup.num_cores} END\n"
        if self.setup.ram_per_core is not None and self.setup.ram_per_core != "None":
            self.total_text += f"%maxcore {self.setup.ram_per_core}\n"

    # def make_entire_trajectory_inp(self, geo_optimization: bool):
    #     #constrain_fragments: bool = False
    #     # num_atoms = self.molecule.fragment_1_len + self.molecule.fragment_2_len
    #     # len_segment_pt = num_atoms + 2
    #     # len_segment_pt = num_atoms+2
    #     len_pt_file = len(self.xyz_file_lines) - 1
    #     len_trajectory = len_pt_file // len_segment_pt
    #     for i in range(len_trajectory):
    #         # all that comes before molecule
    #         self._write_first_line(geo_optimization=geo_optimization)
    #         self._write_solvent()
    #         self._write_resources()
    #         # writing this *xyz frame, don't need the num of atoms
    #         # start_line = i * len_segment_pt + 2
    #         # end_line = i * len_segment_pt + len_segment_pt
    #         start_line = i * len_pt_file + 2
    #         end_line = i * len_pt_file + len_pt_file
    #         self._write_molecule_specification("".join(self.xyz_file_lines[start_line:end_line]))
    #         # all that comes after
    #         # if constrain_fragments:
    #         #     self._write_fragment_constraint()


    def _write_molecule_specification(self):
        xyz_filename = Path(self.molecule.xyz_file).name

        self.total_text += f"* xyzfile {self.molecule.charge} {self.molecule.multiplicity} {xyz_filename}\n"

    # def _write_molecule_specification(self):
    #     self.total_text += f"* xyzfile {self.molecule.charge} {self.molecule.multiplicity} {self.molecule.xyz_file}\n"

    def make_input(self, geo_optimization: bool = False):
        self._write_first_line(geo_optimization=geo_optimization)
        self._write_solvent()
        self._write_output_minimisation()
        self._write_resources()

        # # skip first 2 lines of xyz (atom count + comment)
        # coords = "".join(self.xyz_file_lines[2:])

        self._write_molecule_specification()

def split_xyz_trajectory(xyz_file: str, output_base_dir: str, structures_per_chunk: int):
    from pathlib import Path

    with open(xyz_file, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    frame_length = n_atoms + 2

    total_frames = len(lines) // frame_length

    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, total_frames, structures_per_chunk):
        chunk_idx = i // structures_per_chunk + 1
        chunk_dir = base_dir / str(chunk_idx)
        chunk_dir.mkdir(exist_ok=True)

        chunk_lines = []

        for j in range(i, min(i + structures_per_chunk, total_frames)):
            start = j * frame_length
            end = start + frame_length
            chunk_lines.extend(lines[start:end])

        # write new xyz file for this chunk
        chunk_xyz = chunk_dir / "structure.xyz"
        with open(chunk_xyz, "w") as f:
            f.writelines(chunk_lines)

############################################################################################################
def load_required_indices(path):
    indices = read_object(path)
    return list(map(int, indices))

def build_energy_map(energy_csv):
    df = pd.read_csv(energy_csv)
    df.columns = df.columns.str.strip()

    if "Total index" in df.columns:
        return dict(zip(df["Total index"], df["Energy [kJ/mol]"]))
    else:
        return dict(zip(df.index, df["Energy [kJ/mol]"]))

def iterate_xyz_frames(traj_path):
    with open(traj_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())
        comment = lines[i + 1].strip()
        block = lines[i:i + natoms + 2]

        yield natoms, comment, block

        i += natoms + 2

def extract_frame_number(comment):
    match = re.search(r"frame(\d+)", comment)
    return int(match.group(1)) if match else None

def write_structure(block, frame_number, energy, out_path):
    block = block.copy()
    block[1] = f"frame{frame_number}, E = {energy} kJ/mol\n"

    with open(out_path, "w") as f:
        f.writelines(block)

def extract_structures_with_orca(
    traj_path,
    indices,
    energy_map,
    output_dir,
    setup
):
# takes trajectory and extracts the structures with the lowest energies according to orca

    os.makedirs(output_dir, exist_ok=True)

    # --- collect frames ---
    frame_dict = {}
    for _, comment, block in iterate_xyz_frames(traj_path):
        frame_number = extract_frame_number(comment)
        if frame_number in indices:
            frame_dict[frame_number] = block

    # --- write in correct order ---
    for i, frame_number in enumerate(indices):
        block = frame_dict.get(frame_number)

        if block is None:
            print(f"Warning: frame {frame_number} not found")
            continue

        energy = energy_map.get(frame_number, "NaN")

        # 🔹 create subdirectory (1/, 2/, ...)
        subdir = os.path.join(output_dir, str(i))
        os.makedirs(subdir, exist_ok=True)

        xyz_path = os.path.join(subdir, f"{i}.xyz")

        # --- write xyz ---
        block = block.copy()
        block[1] = f"frame{frame_number}, E = {energy} kJ/mol\n"

        with open(xyz_path, "w") as f:
            f.writelines(block)

        # --- create ORCA input ---
        molecule = QuantumMolecule(
            xyz_file=xyz_path,
            charge=0,
            multiplicity=1
        )

        writer = OrcaWriter(molecule, setup)
        writer.make_input(geo_optimization=True)

        inp_path = os.path.join(subdir, "opt.inp")
        writer.write_to_file(inp_path)

def copy_xyz_to_curta(source_dir, target_dir):
    import subprocess
    subprocess.run(f"scp -r {source_dir} {target_dir}", shell=True, check=True)


def read_xyz_trajectory(xyz_file):
    frames = []
    with open(xyz_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            f.readline()  # comment line

            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            frames.append(np.array(coords))

    return frames


def compute_com_distance(frame):
    mol1 = frame[:12]
    mol2 = frame[12:]

    com1 = mol1.mean(axis=0)
    com2 = mol2.mean(axis=0)

    return np.linalg.norm(com1 - com2)


def plot_energy_vs_distance(xyz_file, energy_csv, output_png, n_lowest, cutoff=1e10):
    frames = read_xyz_trajectory(xyz_file)
    energy_df = pd.read_csv(energy_csv, index_col=0)

    energies = energy_df.iloc[:, 0].to_numpy()
    # convert to relative energies
    energies = energies - np.min(energies)
    distances = np.array([compute_com_distance(f) for f in frames])

    # 🔹 FILTER: remove very large energies
    valid_mask = energies <= cutoff

    energies = energies[valid_mask]
    distances = distances[valid_mask]

    if len(distances) != len(energies):
        raise ValueError("Mismatch between frames and energies")

    lowest_indices = np.argsort(energies)[:n_lowest]

    # Subset for plotting
    distances_lowest = distances[lowest_indices]
    energies_lowest = energies[lowest_indices]

    # Mask for all other frames
    mask = np.ones(len(energies), dtype=bool)
    mask[lowest_indices] = False
    min_idx = np.argmin(energies)

    # --- plotting ---
    plt.figure()

    # normal points (blue)
    plt.scatter(distances[mask], energies[mask], label="All")
    plt.scatter(distances_lowest, energies_lowest, label=f"Lowest {n_lowest}")

    plt.scatter(
        distances[min_idx],
        energies[min_idx],
        marker="x",
        color='red',
        s=100,
        label="Global minimum"
    )

    plt.xlabel("COM distance [nm]")
    plt.ylabel("Energy [kJ/mol]")
    plt.title("Energy vs COM Distance")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_lowest_energy_vs_distance(xyz_file, energy_csv, output_png, N_lowest, n_lowest, cutoff=1e10):
    frames = read_xyz_trajectory(xyz_file)
    energy_df = pd.read_csv(energy_csv, index_col=0)

    energies = energy_df.iloc[:, 0].to_numpy()
    # convert to relative energies
    energies = energies - np.min(energies)
    distances = np.array([compute_com_distance(f) for f in frames])

    # 🔹 FILTER: remove very large energies
    valid_mask = energies <= cutoff

    energies = energies[valid_mask]
    distances = distances[valid_mask]

    if len(distances) != len(energies):
        raise ValueError("Mismatch between frames and energies")

    # 🔹 select N lowest points (to plot)
    N = min(N_lowest, len(energies))
    lowest_N_idx = np.argsort(energies)[:N]

    # subset everything to N lowest
    energies = energies[lowest_N_idx]
    distances = distances[lowest_N_idx]

    # 🔹 select n lowest within those
    n = min(n_lowest, len(energies))
    highlight_idx = np.argsort(energies)[:n]
    min_idx = np.argmin(energies)

    # 🔹 subset
    distances_lowest = distances[highlight_idx]
    energies_lowest = energies[highlight_idx]

    # --- plotting ---
    plt.figure()
    if N != n:
        plt.scatter(distances, energies, label=f"Lowest {len(lowest_N_idx)}")

    plt.scatter(
        distances_lowest,
        energies_lowest,
        label=f"Lowest {n}"
    )

    plt.scatter(
        distances[min_idx],
        energies[min_idx],
        marker="x",
        color='red',
        s=100,
        label="Global minimum"
    )

    plt.xlabel("COM distance [nm]")
    #plt.gca().ticklabel_format(useOffset=False, axis='y')
    plt.ylabel("Energy [kJ/mol]")
    plt.title(f"{N} Lowest Energies vs COM Distance")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_plane_normal(coords):
    """Return normal vector of a molecule using PCA (robust plane fit)."""
    coords_centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords_centered)
    return vh[-1]


def angle_between(v1, v2):
    """Return angle (deg) between two vectors."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


def compute_angles(frame):
    """Compute angles of molecule 2 relative to xy, xz, yz planes."""
    mol2 = frame[12:]
    normal = compute_plane_normal(mol2)

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    return (
        angle_between(normal, z),  # xy-plane
        angle_between(normal, y),  # xz-plane
        angle_between(normal, x),  # yz-plane
    )

def make_plot(angles, energies, lowest, min_idx, title, filename):
    plt.figure()
    plt.scatter(angles, energies, label="All")
    plt.scatter(angles[lowest], energies[lowest], label=f"Lowest {len(lowest)}")
    plt.scatter(angles[min_idx], energies[min_idx], marker="x", color='red', s=100, label="Global minimum")
    plt.xlabel("Angle [deg]")
    plt.ylabel("Energy [kJ/mol]")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_energy_vs_angles(xyz_file, energy_csv, xy_path, xz_path, yz_path, n_lowest, cutoff=1e10):
    """Plot energy vs angles and save directly to given output paths."""

    frames = read_xyz_trajectory(xyz_file)
    energies = pd.read_csv(energy_csv, index_col=0).iloc[:, 0].to_numpy()
    # convert to relative energies
    energies = energies - np.min(energies)


    if len(frames) != len(energies):
        raise ValueError("Mismatch between frames and energies")

    angles_xy, angles_xz, angles_yz = [], [], []

    for frame in frames:
        a_xy, a_xz, a_yz = compute_angles(frame)
        angles_xy.append(a_xy)
        angles_xz.append(a_xz)
        angles_yz.append(a_yz)

    angles_xy = np.array(angles_xy)
    angles_xz = np.array(angles_xz)
    angles_yz = np.array(angles_yz)

    # 🔹 FILTER: remove very large energies
    valid_mask = energies <= cutoff

    energies = energies[valid_mask]
    angles_xy = angles_xy[valid_mask]
    angles_xz = angles_xz[valid_mask]
    angles_yz = angles_yz[valid_mask]

    if len(energies) == 0:
        raise ValueError("No valid energies left after filtering")

    # 🔹 recompute lowest after filtering
    n = min(n_lowest, len(energies))

    lowest = np.argsort(energies)[:n]
    mask = np.ones(len(energies), dtype=bool)
    mask[lowest] = False
    min_idx = np.argmin(energies)

    # make_plot now takes full paths
    make_plot(angles_xy, energies, lowest, min_idx, "XY plane", xy_path)
    make_plot(angles_xz, energies, lowest, min_idx, "XZ plane", xz_path)
    make_plot(angles_yz, energies, lowest, min_idx, "YZ plane", yz_path)



def plot_lowest_energy_vs_angles(xyz_file, energy_csv, xy_path, xz_path, yz_path, N_lowest, n_lowest, cutoff=1e10):
    """
    Plot the lowest `n_lowest` energies vs angles relative to XY, XZ, YZ planes.

    Parameters
    ----------
    xyz_file : str
        Path to XYZ trajectory file
    energy_csv : str
        Path to CSV containing energies
    xy_path : str
        Output path for XY plane plot
    xz_path : str
        Output path for XZ plane plot
    yz_path : str
        Output path for YZ plane plot
    n_lowest : int
        Number of lowest energies to highlight
    """

    def make_plot(angles, energies, lowest_N_idx, lowest_idx, min_idx, title, filename):
        mask = np.ones(len(energies), dtype=bool)
        mask[lowest_idx] = False

        plt.figure()
        if {len(lowest_N_idx)} != {len(lowest_idx)}:
            plt.scatter(angles[mask], energies[mask], label=f"Lowest {len(lowest_N_idx)}")

        plt.scatter(angles[lowest_idx], energies[lowest_idx], label=f"Lowest {len(lowest_idx)}")
        plt.scatter(angles[min_idx], energies[min_idx], marker="x", color='red', s=100, label="Global minimum")
        plt.xlabel("Angle [deg]")
        plt.ylabel("Energy [kJ/mol]")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # --- main computation ---
    frames = read_xyz_trajectory(xyz_file)
    energies = pd.read_csv(energy_csv, index_col=0).iloc[:, 0].to_numpy()
    # convert to relative energies
    energies = energies - np.min(energies)


    if len(frames) != len(energies):
        raise ValueError("Mismatch between frames and energies")

    angles_xy, angles_xz, angles_yz = [], [], []
    for frame in frames:
        a_xy, a_xz, a_yz = compute_angles(frame)  # uses external compute_angles
        angles_xy.append(a_xy)
        angles_xz.append(a_xz)
        angles_yz.append(a_yz)

    angles_xy = np.array(angles_xy)
    angles_xz = np.array(angles_xz)
    angles_yz = np.array(angles_yz)

    # 🔹 FILTER: remove very large energies
    valid_mask = energies <= cutoff

    energies = energies[valid_mask]
    angles_xy = angles_xy[valid_mask]
    angles_xz = angles_xz[valid_mask]
    angles_yz = angles_yz[valid_mask]

    if len(energies) == 0:
        raise ValueError("No valid energies left after filtering")

    # 🔹 select N lowest points (to plot)
    N = min(N_lowest, len(energies))
    lowest_N_idx = np.argsort(energies)[:N]

    # subset everything to N lowest
    energies = energies[lowest_N_idx]
    angles_xy = angles_xy[lowest_N_idx]
    angles_xz = angles_xz[lowest_N_idx]
    angles_yz = angles_yz[lowest_N_idx]

    # 🔹 select n lowest within those
    n = min(n_lowest, len(energies))
    highlight_idx = np.argsort(energies)[:n]
    min_idx = np.argmin(energies)

    # plot
    make_plot(angles_xy, energies, lowest_N_idx, highlight_idx, min_idx, "XY plane", xy_path)
    make_plot(angles_xz, energies, lowest_N_idx, highlight_idx, min_idx, "XZ plane", xz_path)
    make_plot(angles_yz, energies, lowest_N_idx, highlight_idx, min_idx, "YZ plane", yz_path)

