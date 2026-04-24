import os
import subprocess
import pandas as pd
import numpy as np
from scipy.constants import physical_constants
from pathlib import Path


HARTREE_TO_J = physical_constants["Hartree energy"][0]
AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]

class OrcaReader:

    """
    Does not read .inp, but .out files
    """

    def __init__(self, out_file_path: str, is_multi_out: bool = False):
        self.out_file_path = os.path.expanduser(out_file_path)
        path = os.path.normpath(self.out_file_path)
        split_path = path.split(os.sep)
        self.frame_num = int(split_path[-2].split("_")[-1])
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
        returncode = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).returncode
        if returncode != 0 and throw_error:
            raise ChildProcessError(f"Orca may have finished normally but did not converge; see {self.out_file_path}")
        elif returncode != 0 and not throw_error:
            print("Opt not complete")
            return False
        # also fail optimization if the calculation failed
        elif not self.assert_normal_finish(throw_error=False):
            print("Not normal finish")
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
        print("Frame number: ", self.frame_num)
        return int(self.frame_num)

def read_important_stuff_into_csv(out_files_to_read: list, csv_file_to_write: str, chunksize: int):
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
    #all_frame_indices = [int(Path(out_file).parts[-2]) for out_file in out_files_to_read]

    for out_file_to_read in out_files_to_read:
        my_reader = OrcaReader(out_file_to_read, is_multi_out=True)

        frame_index = my_reader.get_frame_num()
        time_h_m_s = my_reader.extract_time_orca_output()
        normal_finish = my_reader.assert_normal_finish(throw_error=False)
        optimization_complete = my_reader.assert_optimization_complete(throw_error=False)

        #base_path = "/home/nadjar02/MA/2026_molgri/nobackup"
        #short_path = out_file_to_read.replace(base_path, "")

        energies = my_reader.extract_energies_orca_output()

        rows = []
        # need to write out all, even failed ones
        for step, energy in enumerate(energies):
            energy_kjmol = energy * HARTREE_TO_J * AVOGADRO_CONSTANT / 1000.0
            global_index = chunksize * (frame_index) + step

            rows.append([
                #short_path,
                frame_index,
                step,
                global_index,
                energy,
                energy_kjmol,
                normal_finish,
                optimization_complete,
                time_h_m_s
            ])

        df = pd.DataFrame(rows, columns=[
            "Frame",
            "Step",
            "Total index",
            "Energy [hartree]",
            "Energy [kJ/mol]",
            "Normal Finish",
            "Optimization Complete",
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

