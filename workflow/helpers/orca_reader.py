import os
import re
import subprocess
from subprocess import PIPE, run
import MDAnalysis as mda
import pandas as pd
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import sparse
from scipy.constants import physical_constants
from MDAnalysis import Merge
from pathlib import Path
from ase.io import read

#from molgri.molecules.pts import Pseudotrajectory
import MDAnalysis.transformations as trans

#from molgri.space.translations import TranslationParser

HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]

def xtc_to_xyz(xtc_file, gro_file, output_xyz):
    u = mda.Universe(gro_file, xtc_file)

    with open(output_xyz, "w") as f:
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

    def __init__(self, functional: str, basis_set: str, solvent: str = None, dispersion_correction: str = "",
                 num_scf: int = 15, num_cores: int = None, ram_per_core: int = None):
        self.functional = functional
        self.basis_set = basis_set
        self.solvent = solvent
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

    def extract_last_energy_orca_output(self) -> list:
        """
        Take any orca output file and give me the total energy resulting from the calculation.

        Returns:
            Energy in the unit of Hartrees
        """
        # need the last one so use tail
        line_energy = subprocess.run(f'grep "^FINAL SINGLE POINT ENERGY" {self.out_file_path} | tail -n 1  | sed '
                                     f'"s/^FINAL SINGLE POINT '
                                     f'ENERGY //"', shell=True,
                                     capture_output=True, text=True)
        try:
            line_energy = line_energy.stdout.strip()
            energy_hartree = float(line_energy)
        except ValueError:
            energy_hartree = np.NaN

        return energy_hartree

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

    def extract_num_atoms(self):
        line = subprocess.run(
            f'grep "^Number of atoms" {self.out_file_path}| head -n 1 ',
            shell=True, capture_output=True, text=True)
        number_atoms = line.stdout
        number_atoms = int(number_atoms.strip().split()[-1])
        return number_atoms

    def extract_optimized_xyz(self) -> str:
        if self.assert_optimization_complete(throw_error=False):
            # find the line number with the last occurence of CARTESIAN COORDINATES (ANGSTROEM)
            line = subprocess.run(
                f'grep -n "CARTESIAN COORDINATES (ANGSTROEM)" {self.out_file_path} | cut -d: -f1 | tail -n 1 ',
                shell=True, capture_output=True, text=True)
            line_number_last_coo = int(line.stdout)
            # start two lines after that, finish two lines + molecule length later
            start_point = 2 + line_number_last_coo
            end_point = 2 + line_number_last_coo + self.extract_num_atoms() -1
            command = ['head', '-n', f"{end_point}", f"{self.out_file_path}", "|", "tail", "-n", f"+{start_point}"]

            line = subprocess.run(
                f'head -n {end_point} {self.out_file_path} | tail -n +{start_point}',
                shell=True, capture_output=True, text=True)

            # starting with num of atoms and comment line
            result = f"{self.extract_num_atoms()}\n"
            result += "\n"
            result += line.stdout
            return result
        else:
            # to indicate an error while preserving pt length the initial structure is copied but all element names are
            # changed to X
            print(f"Not complete {self.out_file_path}")
            return ""

    def extract_last_coordinates_from_opt(self) -> str:
        """
        Extract the last structure of the optimization.
        """
        # try to find _trj.xyz in the directory
        for file in os.listdir(self.calculation_directory):
            if str(file).endswith("_trj.xyz"):
                orca_traj_xyz_file = os.path.join(self.calculation_directory, file)
                line_number_last_coo = subprocess.run(
                    f"""grep -n "Coordinates from" {orca_traj_xyz_file} | tail -n 1 | cut -d: -f1""",
                    shell=True, capture_output=True).stdout
                line_with_num_of_atoms = int(line_number_last_coo) - 1
                command = ['tail', '-n', f"+{line_with_num_of_atoms}", f"{orca_traj_xyz_file}"]
                result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                return result.stdout
        else:
            raise FileNotFoundError(f"Cannot find any _trj.xyz file in {self.calculation_directory}")

    def extract_last_coordinates_to_file(self, file_path: str):
        """
        Same as extract_last_coordinates_from_opt, but immediately write to a file.

        Args:
            file_path (str): a path where the new file should be
        """
        file_contents = self.extract_last_coordinates_from_opt()
        with open(file_path, "w") as f:
            f.write(file_contents)

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

    columns = ["File", "Frame", "Global_index", "Functional", "Basis set", "Dispersion correction", "Solvent",
               "Energy [hartree]", "Time [h:m:s]", "Normal Finish", "Optimization Complete"]

    all_df = []


    all_frame_indices = [int(Path(out_file).parts[-2]) for out_file in out_files_to_read]
    #all_frame_indices = [0]

    for out_file_to_read in out_files_to_read:
        my_reader = OrcaReader(out_file_to_read)

        frame_index = my_reader.get_frame_num()
        energy_hartree = my_reader.extract_last_energy_orca_output()
        time_h_m_s = my_reader.extract_time_orca_output()
        normal_finish = my_reader.assert_normal_finish(throw_error=False)
        optimization_complete = my_reader.assert_optimization_complete(throw_error=False)

        base_path = "/home/nadjar02/MA/2026_molgri/nobackup"
        short_path = out_file_to_read.replace(base_path, "")

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

    # for i, out_file_to_read in enumerate(out_files_to_read):
    #     my_reader = OrcaReader(out_file_to_read)
    #     energy_hartree = my_reader.extract_energy_orca_output()
    #
    #     if is_pt:
    #         frame = my_reader.get_frame_num()
    #     else:
    #         frame = None
    #
    #     time_h_m_s = my_reader.extract_time_orca_output()
    #
    #     all_data = [[out_file_to_read, frame, setup.functional, setup.basis_set, setup.dispersion_correction,
    #                  setup.solvent, energy_hartree, time_h_m_s]]
    #
    #     df = pd.DataFrame(all_data, columns=columns)
    #     df["Energy [kJ/mol]"] = df["Energy [hartree]"] / 1000.0 * (HARTREE_TO_J * AVOGADRO_CONSTANT)
    #
    #     df["Normal Finish"] = my_reader.assert_normal_finish(throw_error=False)
    #     df["Optimization Complete"] = my_reader.assert_optimization_complete(throw_error=False)
    #     all_df.append(df)

    combined_df = pd.concat(all_df)
    try:
        combined_df["Time [h:m:s]"] = pd.to_timedelta(combined_df["Time [h:m:s]"])
        combined_df["Time [s]"] = np.where(combined_df["Normal Finish"], combined_df["Time [h:m:s]"].dt.total_seconds(), np.NaN)
    except:
        # THIS DELTA TIME THING IS A HEADACHE!!!!
        pass
    combined_df.to_csv(csv_file_to_write, index=False)
    #write_output_file(combined_df, csv_file_to_write, file_type="csv")

def read_energies_into_csv(out_files_to_read: list, csv_file_to_write: str, setup: QuantumSetup,  chunksize: int,
                                  num_points: int,  invalid_indices: set = None, is_pt=True):
    """
    Read a list of orca .out files that were created with the same set-up (functional, basis set ...). Save the
    energies and global indices.

    Args:
        out_files_to_read (list): a list of paths, usually to a number of .out files calculated along a molgri pt
        csv_file_to_write (str): a path to a csv file where the data will be recorded.
        setup ():

    Returns:

    """
    if invalid_indices is None:
        invalid_indices = set()

    all_df = []

    current_traj_index = 0

    for out_file_to_read in out_files_to_read:
        my_reader = OrcaReader(out_file_to_read)
        energies = my_reader.extract_energies_orca_output()

        rows = []
        for step, energy in enumerate(energies):

            # 🔥 skip invalid frames
            if current_traj_index in invalid_indices:
                current_traj_index += 1
                continue

            energy_kjmol = energy * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0

            rows.append([
                current_traj_index,
                energy_kjmol,
            ])

            current_traj_index += 1
        print("Total energies processed:", current_traj_index)
        print("Invalid indices:", sorted(invalid_indices))

        df = pd.DataFrame(rows, columns=[
            "Total index",
            "Energy [kJ/mol]"
        ])
        all_df.append(df)

    combined_df = pd.concat(all_df)
    combined_df.to_csv(csv_file_to_write, index=False)


import re
from ase.io import read

import re

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


import numpy as np
import pandas as pd

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

    # 🔹 1. get frame indices
    frame_indices = extract_frame_indices_from_xyz(trajectory_path)

    # 🔹 2. get invalid indices (trajectory positions)
    _, invalid = find_invalid_frames_with_overlapping_atoms(trajectory_path)
    invalid_set = set(invalid)
    frame_indices = filter_frame_indices(frame_indices, invalid)
    print(frame_indices)

    # 🔹 3. collect all energies (flatten)
    all_energies = []
    for out_file in sorted(out_files_to_read):
        reader = OrcaReader(out_file)
        energies = reader.extract_energies_orca_output()
        all_energies.extend(energies)

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
            energy_kjmol   # ✅ keep in Hartree
        ])

    # 🔹 5. write CSV
    df = pd.DataFrame(rows, columns=["Total index", "Energy [kJ/mol]"])
    df.to_csv(csv_file_to_write, index=False)

def filter_frame_indices(frame_indices, invalid_indices):
    invalid_indices = np.array(invalid_indices, dtype=int)
    mask = np.ones(len(frame_indices), dtype=bool)
    mask[invalid_indices] = False
    #print("Filtered frame indices:", list(np.array(frame_indices)[mask]))
    return list(np.array(frame_indices)[mask])


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

def write_output_file(data, output_path: str, file_type: str = "csv", index: bool = False):
    """
    Allgemeine Funktion zum Schreiben von Output-Dateien.

    Args:
        data: pandas DataFrame ODER String
        output_path (str): Zielpfad (inkl. Dateiname)
        file_type (str): "csv" oder "txt"
        index (bool): ob DataFrame-Index geschrieben werden soll
    """

    output_path = Path(output_path)

    # 📁 Stelle sicher, dass der Ordner existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_type == "csv":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Für CSV muss 'data' ein pandas DataFrame sein.")
        data.to_csv(output_path, index=index)

    elif file_type == "txt":
        with open(output_path, "w") as f:
            f.write(str(data))

    else:
        raise ValueError(f"Unbekannter file_type: {file_type}")

    print(f"✅ Datei geschrieben: {output_path}")



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

#     def _write_fragment_constraint(self):
#         """
#
#         Returns:
#
#         """
#         assert self.molecule.fragment_1_len is not None, "Need to know fragment lengths to constrain them!"
#         assert self.molecule.fragment_2_len is not None, "Need to know fragment lengths to constrain them!"
#
#
#         self.total_text += f"""
# %geom
#
#
#     ConnectFragments
#      {{1 2 C}}      # constrain the internal coordinates
#               #  connecting fragments 1 and 2
#     end
#
#     Fragments
#       1 {{0:{self.molecule.fragment_1_len-1}}} end
#       2 {{{self.molecule.fragment_1_len}:{self.molecule.fragment_1_len+self.molecule.fragment_2_len-1}}} end
#     end
#
# end\n"""

    def _write_solvent(self):
        if self.setup.solvent is not None:
            self.total_text += "%CPCM SMD TRUE\n"
            self.total_text += f'SMDSOLVENT "{self.setup.solvent}"\n'
            self.total_text += "END\n"

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

    def make_entire_trajectory_inp(self, geo_optimization: bool):
        #constrain_fragments: bool = False
        # num_atoms = self.molecule.fragment_1_len + self.molecule.fragment_2_len
        # len_segment_pt = num_atoms + 2
        # len_segment_pt = num_atoms+2
        len_pt_file = len(self.xyz_file_lines) - 1
        len_trajectory = len_pt_file // len_segment_pt
        for i in range(len_trajectory):
            # all that comes before molecule
            self._write_first_line(geo_optimization=geo_optimization)
            self._write_solvent()
            self._write_resources()
            # writing this *xyz frame, don't need the num of atoms
            # start_line = i * len_segment_pt + 2
            # end_line = i * len_segment_pt + len_segment_pt
            start_line = i * len_pt_file + 2
            end_line = i * len_pt_file + len_pt_file
            self._write_molecule_specification("".join(self.xyz_file_lines[start_line:end_line]))
            # all that comes after
            # if constrain_fragments:
            #     self._write_fragment_constraint()


    # def _write_molecule_specification(self, use_string):
    #     """
    #     Here we don't reference the .xyz file but write coordinates directly into .inp.
    #     """
    #     self.total_text += f"* xyz {self.molecule.charge} {self.molecule.multiplicity}\n"
    #     self.total_text += use_string
    #     self.total_text += "*\n"

    def _write_molecule_specification(self):
        xyz_filename = Path(self.molecule.xyz_file).name

        self.total_text += f"* xyzfile {self.molecule.charge} {self.molecule.multiplicity} {xyz_filename}\n"

    # def _write_molecule_specification(self):
    #     self.total_text += f"* xyzfile {self.molecule.charge} {self.molecule.multiplicity} {self.molecule.xyz_file}\n"

    def make_input(self, geo_optimization: bool = False):
        self._write_first_line(geo_optimization=geo_optimization)
        self._write_solvent()
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

