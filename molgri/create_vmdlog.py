"""
Vmdlogs are instruction files provided to VMD. Here we create vmdlogs that help display specific structures of the
(pseudo)trajectories, often the ones corresponding to eigenvector extremes.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike

from molgri.utils.arrays import k_argmax_in_array, k_argmin_in_array

VMD_COLOR_DICT = {"black": 16, "yellow": 4, "orange": 3, "green": 7, "blue": 0, "cyan": 10, "purple": 11,
              "gray": 2, "pink": 9, "red": 1, "magenta": 27, "silver": 6, "gold": 5, "lime": 12}


class TrajectoryIndexingTool:
    """
    A tool that helps getting specific indices from the trajectory
    """
    def __init__(self):
        self.adjacency = None
        self.distances = None

    def set_adjecency_array(self, adjacency_array):
        self.adjacency=adjacency_array.tolil()

    def set_distances(self, distances):
        self.distances=distances.tolil()

    def set_labels(self, cluster_labels):
        self.labels=cluster_labels

    def set_energies(self, energies):
        self.energies=energies

    def set_eigenvectors(self, eigenvectors, index_list = None):
        if not np.any(index_list):
            self.index_list = None
        else:
            self.index_list = list(index_list)
        self.eigenvectors = eigenvectors

    def get_dominant_structures_eigenvector_0(self, num_extremes):
        """
        Get the num_extremes most prominent structures in the 0th eigenvector (the ones with largest absolute value)

        Args:
            num_extremes (int): how many indices to return

        Returns:
            a list of trajectory indices
        """
        if self.eigenvectors is None:
            raise ValueError("Cannot tell dominant eigenvectors if no eigenvectors provided!")
        return find_indices_of_largest_eigenvectors(self.eigenvectors[0], which="abs",
                                             index_list=self.index_list,
                                             num_extremes=num_extremes,
                                             add_one=False)

    def get_dominant_structures_eigenvector_i(self, i, num_extremes):
        """
        Get the num_extremes most prominent structures in the i-th eigenvector (the ones with largest positive and
        negative value)

        Args:
            i (int): which eigenvector
            num_extremes (int): how many indices to return

        Returns:
            a list of trajectory indices
        """
        if self.eigenvectors is None:
            raise ValueError("Cannot tell dominant eigenvectors if no eigenvectors provided!")

        pos_eigenvector = find_indices_of_largest_eigenvectors(self.eigenvectors[i], which="pos",
                                                               index_list=self.index_list, num_extremes=num_extremes,
                                                               add_one=False)
        neg_eigenvector = find_indices_of_largest_eigenvectors(self.eigenvectors[i], which="neg",
                                                               index_list=self.index_list, num_extremes=num_extremes,
                                                               add_one=False)
        return pos_eigenvector, neg_eigenvector

    def get_all_dominant_structures(self, num_extremes, num_eigenvec):
        # first zeroth eigenvector
        eigenvector_zero = self.get_dominant_structures_eigenvector_0(2*num_extremes)

        all_pos_eigenvectors = []
        all_neg_eigenvectors = []
        for i in range(1, num_eigenvec):
            pos_eigenvector, neg_eigenvector = self.get_dominant_structures_eigenvector_i(i=i, num_extremes= num_extremes)
            all_pos_eigenvectors.append(list(pos_eigenvector))
            all_neg_eigenvectors.append(list(neg_eigenvector))
        return eigenvector_zero, all_pos_eigenvectors, all_neg_eigenvectors

    def get_neighbours_of(self, index: int) -> NDArray:
        """
        Provide an index along a trajectory, get indices of all direct neighbours as a flat array.
        """
        if self.adjacency is None:
            raise ValueError("Cannot determine neighbours if adjacency array is not provided.")
        else:
            return np.array(self.adjacency.rows[index], dtype=int)

    def get_k_closest(self, index: int, k=5) -> NDArray:
        """
        Provide an index along a trajectory, get indices of all direct neighbours as a flat array.
        """
        if self.distances is None:
            raise ValueError("Cannot determine closest neighbours if distances are not provided.")
        else:
            argmins = k_argmin_in_array(self.distances.data[index], k=k)
            selected_indices = [self.distances.rows[index][int(am)] for am in argmins]
            return np.array(selected_indices, dtype=int)

    def find_all_orientations_at_same_position(self, index: int, num_orientations: int) -> NDArray:
        """
        Provide an index along a trajectory, get indices of all structures with same position.
        """
        position_index = index//num_orientations
        return np.arange(position_index*num_orientations, (position_index+1)*num_orientations)

    def find_structures_lowest_energy(self, num_structures: int = 5) -> NDArray:
        if self.energies is None:
            raise ValueError("Cannot find lowest energy structures if energies not given!")
        else:
            return k_argmin_in_array(self.energies, num_structures)

    def get_all_cluster_elements(self, max_num_per_cluster: int = 50, ignore_smaller_than: int = 5,
                                 ignore_larger_than: int = 200) -> list:
        if self.labels is None:
            raise ValueError("Cannot report clusters if no labels provided.")

        unique, counts = np.unique(self.labels, return_counts=True)
        cluster_list=[]
        for i, label in enumerate(unique[np.where(counts>1)[0]]):
            cluster = np.where(self.labels == label)[0]
            population = len(cluster)

            if population > ignore_smaller_than and population < ignore_larger_than:
                # if less than max add all to list
                if population < max_num_per_cluster:
                    cluster_list.append(list(cluster))
                # else only add randomly selected sample of existing structures
                else:
                    cluster_list.append([x for x in np.random.choice(cluster, max_num_per_cluster)])
        return cluster_list



def find_num_extremes(eigenvector_array: NDArray, explains_x_percent: float = 40, only_positive: bool = True) -> int:
    """
    In a 1D array that may contain positive and negative elements, we want to know how many of the largest positive
    values we need to sum to get at least X % of the total sum of positive elements (option only_positive=True) or
    how many of the most negative values we need to sum to get at least X % of the total sum of negative elements (
    option only_positive=False

    Args:
        eigenvector_array (NDArray):
        explains_x_percent (float): percentage we wanna explain, usually around 70-90
        only_positive (bool): if True we will only consider positive values in array, if False only negative values

    Returns:
        The number (int) of elements in eigenvector_array that we need to sum to get to X % (pos or neg). Note that
        is eigenvector_array is not sorted, these elements may not be the first elements.
    """
    if only_positive:
        allowed_elements = eigenvector_array[eigenvector_array > 0]
    else:
        allowed_elements = eigenvector_array[eigenvector_array < 0]
        allowed_elements = - allowed_elements

    # case where e.g. all elements positive but you are looking at the needed number of neg elements
    if len(allowed_elements) == 0:
        return 0

    total_sum = np.sum(allowed_elements)
    sorted_by_size = -np.sort(-allowed_elements)  # this is to sort from largest to smallest
    partial_sum = np.cumsum(sorted_by_size)
    percentage_explained = 100 * partial_sum / total_sum
    # first el that reaches x_percent
    larger_index = np.argmax(percentage_explained > explains_x_percent)
    # because of zero-indexing we use +1 -> e.g. if we reach 80% with the first element, the index will be 0 but we
    # need 1 element
    return larger_index + 1


def find_indices_of_largest_eigenvectors(eigenvector_array: NDArray, which: str, num_extremes: int,
                                         index_list: list = None, add_one: bool = True) -> NDArray:
    """
    Given a 1D eigenvector array, find indices with the largest values. This is a help function so that you can create
    VMD inputs that display most positive and most negative parts of an eigenvector with molecular stuctures.

    Args:
        eigenvector_array: array of shape (N_d,) of eigenvector values for a single eigenvector
            (here: N_d is the num of cells in the grid)
        which: "abs", "pos" or "neg" <- largest in absolute sense or most positive or most negative
        num_extremes: how many of the largest-value indices of eigenvector array to consider
        index_list: only used if indices of eigenvector array are not directly transferrable to assignment/Pt indices,
            perhaps because some pt frames with very high energies were filtered out or similar
        add_one: add +1 to indices, used because VMD has the structure file as the 0th frame
    """
    # determine the max values based on which parameter you sort by
    if num_extremes is None:
        if which == "abs" or which == "pos":
            num_extremes = find_num_extremes(np.abs(eigenvector_array), only_positive=True)
        else:
            num_extremes = find_num_extremes(eigenvector_array, only_positive=False)
    if which == "abs":
        most_populated = k_argmax_in_array(np.abs(eigenvector_array), num_extremes)
    elif which == "pos":
        most_populated = k_argmax_in_array(eigenvector_array, num_extremes)
    elif which == "neg":
        most_populated = k_argmax_in_array(-eigenvector_array, num_extremes)
    else:
        raise ValueError(f"The only available options for 'which' are 'abs', 'pos' and 'neg', not {which}.")
    # now possibly perform re-indexing if index_list is involved
    original_index_populated = []
    if index_list is None:
        original_index_populated = most_populated
    else:
        for mp in most_populated:
            original_index_populated.append(index_list[mp])

    original_index_populated = np.array(original_index_populated)

    if add_one:
        original_index_populated += 1

    original_index_populated = original_index_populated.reshape((-1,))
    return original_index_populated


class VMDCreator:
    """
    From pieces of strings build a long vmdlog file that can be used inside vmd to automatically execute commands
    like: create a CPK representation, rotate the molecule, render a picture, change the color, render again etc.
    """

    def __init__(self, index_first_molecule: str = "all", index_second_molecule: str = "all", is_protein: bool = False):
        """
        Because we specialize in two-molecule systems we need to know how to find the first and the second molecule,
        because we will typically represent each of them individually.

        Args:
            index_first_molecule (str): command that selects the first molecule, like 'index < 3'
            index_second_molecule (str): command that selects the second molecule, like 'index >= 3'
            is_protein (bool): use True to automatically select more protein-like representations (secondary
            structure rather than ball and stick)
        """
        self.index_first_molecule = index_first_molecule
        self.index_second_molecule = index_second_molecule
        self.translations_rotations_script = None
        self.is_protein = is_protein

        if is_protein:
            self.default_coloring_method = "Structure"
            self.default_drawing_method = "NewCartoon"
        else:
            self.default_coloring_method = "Type"
            self.default_drawing_method = "DynamicBonds 1.600000 0.300000 6.000000"

        self.num_representations = 0
        self._start_new_file()
    
    def _start_new_file(self):
        """
        Just a helper for starting a new string and adding default settings.
        """
        self.total_file_text = ""
        self._add_pretty_plot_settings()

    def write_text_to_file(self, output_file_path: str) -> None:
        """
        Because basically every method here writes to internal self.total_file_text my_property, in the end we just
        need to transfer it to a file.
        """
        with open(output_file_path, "w") as f:
            f.write(self.total_file_text)

        # in case we want to build another file after this one
        self._start_new_file()

    def _add_pbc_box(self):
        self.total_file_text += """pbc box\n"""

    def _add_pretty_plot_settings(self) -> None:
        """
        Delete the initial default representation. Don't add any representations yet.
        Additionally, some nice settings so that pictures look good.
        """
        #display projection Orthographic

        self.total_file_text += f"""
history keep 0
mol delrep 0 0
color Display Background white
axes location Off
mol material Opaque
display shadows on
display ambientocclusion on
material add copy AOChalky
material change shininess Material22 0.000000
display depthcue off
display projection Orthographic
color Type C gray
display nearclip 0.001
display farclip 1000.0
display resize 1800 1800
"""

    def add_box(self, length_x, length_y, length_z) -> None:
        self.total_file_text += f"""
set mol top
set x0 0.0
set y0 0.0
set z0 0.0
set x1 {str(float(length_x))}
set y1 {str(float(length_y))}
set z1 {str(float(length_z))}

# define corners
set c000 [list $x0 $y0 $z0]
set c100 [list $x1 $y0 $z0]
set c010 [list $x0 $y1 $z0]
set c110 [list $x1 $y1 $z0]
set c001 [list $x0 $y0 $z1]
set c101 [list $x1 $y0 $z1]
set c011 [list $x0 $y1 $z1]
set c111 [list $x1 $y1 $z1]

# draw edges
graphics $mol cylinder $c000 $c100 radius 0.1 resolution 20
graphics $mol cylinder $c000 $c010 radius 0.1 resolution 20
graphics $mol cylinder $c000 $c001 radius 0.1 resolution 20

graphics $mol cylinder $c100 $c110 radius 0.1 resolution 20
graphics $mol cylinder $c100 $c101 radius 0.1 resolution 20

graphics $mol cylinder $c010 $c110 radius 0.1 resolution 20
graphics $mol cylinder $c010 $c011 radius 0.1 resolution 20

graphics $mol cylinder $c001 $c101 radius 0.1 resolution 20
graphics $mol cylinder $c001 $c011 radius 0.1 resolution 20

graphics $mol cylinder $c110 $c111 radius 0.1 resolution 20
graphics $mol cylinder $c101 $c111 radius 0.1 resolution 20
graphics $mol cylinder $c011 $c111 radius 0.1 resolution 20
"""""

    def _zoom_on_box(self, length_x, length_y, zoom_level=1):
        self.total_file_text += f"""
set xmin 0
set ymin 0
set xmax {length_x}
set ymax  {length_y}

set cx [expr {{($xmin + $xmax)/2.0}}]
set cy [expr {{($ymin + $ymax)/2.0}}]

molinfo top set center [list $cx $cy 0.0]

set dx [expr {{$xmax - $xmin}}]
set dy [expr {{$ymax - $ymin}}]
set maxxy [expr {{max($dx, $dy)}}]

scale to [expr {{ {zoom_level} * 0.1 / $maxxy}}]
"""


    def _add_representation(self, first_molecule: bool = False, second_molecule: bool = True, coloring: str = None,
                            color: str = None, representation: str = None, trajectory_frames: ArrayLike = None,
                            periodic: str = None):
        """
        Use this to add a new representation of molecule 1, molecule 2 or both.

        Args:
            first_molecule (bool): True to show molecule 1 in this representation
            second_molecule (bool): True to show molecule 2 in this representation
            coloring (str): keyword understood by VMD how to group for coloring like 'Name', 'Type',
                'SecondaryStructure' or 'ColorId'
            color (str): only used if coloring='ColorId' is selected, color is translated to VMD's internal color ID
            representation (str): keyword understood by VMD how to show structure like 'CPK', 'VDW',
                'NewCartoon' ...
            trajectory_frames (ArrayLike): which frames to use, can be 'now', can be a single frame number (int) or a
                list-like object of multiple frame numbers
        """
        if coloring is None:
            coloring = self.default_coloring_method
        if coloring == "ColorID":
            if color is None:
                color = "black"
            # if the coloring type is ColorID, we need an additional parameter that specifies the color
            coloring = f"{coloring} {VMD_COLOR_DICT[color]}"
        if representation is None:
            representation = self.default_drawing_method

        if first_molecule and not second_molecule:
            molecular_index = self.index_first_molecule
        elif second_molecule and not first_molecule:
            molecular_index = self.index_second_molecule
        elif first_molecule and second_molecule:
            molecular_index = "all"
        else:
            raise ValueError("Trying to add a molecule but first_molecule=False and second_molecule=False.")

        # because trajectory frames may be an int, a string, or a list-like object we ned to pre-process it
        if isinstance(trajectory_frames, np.integer) or isinstance(trajectory_frames, int):
            trajectory_frames_as_str = str(trajectory_frames)
        elif isinstance(trajectory_frames, list):
            trajectory_frames_as_str = ', '.join(map(str, [int(x) for x in trajectory_frames]))
        elif isinstance(trajectory_frames, np.ndarray):
            trajectory_frames_as_str = ', '.join(map(str, trajectory_frames.flatten().astype(int)))
        elif isinstance(trajectory_frames, str):
            trajectory_frames_as_str = trajectory_frames
        else:
            raise ValueError(f"Trajectory frame indices of type {type(trajectory_frames)} cannot be read.")

        self.total_file_text += f"""
mol addrep 0
mol modstyle {self.num_representations} 0 {representation}
mol modselect {self.num_representations} 0 {molecular_index}
mol modcolor {self.num_representations} 0 {coloring}
mol drawframes 0 {self.num_representations} {{ {trajectory_frames_as_str} }}
        """

        if periodic is not None:
            self.total_file_text += f"""
mol showperiodic 0 {self.num_representations} {periodic}
mol numperiodic 0 {self.num_representations} 1
            """

        self.num_representations += 1

    def _add_dot(self, coordinate, radius=0.2):
        self.total_file_text += f"""
graphics top sphere {{ {coordinate[0]} {coordinate[1]} {coordinate[2]} }} radius {radius} resolution 12
"""

    def add_grid(self, coordinates, radius=0.2):
        for coordinate in coordinates:
            self._add_dot(coordinate, radius)

    def _render_representations(self, list_representation_indices: ArrayLike, plot_path: str) -> None: 
        """
        Show representations that are in the list_representation_indices and hide all others. Save the rendered plot
        to plot_path.

        Args:
            list_representation_indices (ArrayLike): provide a list-like object of integers, each pointing to an (
            already added) representation we want to use for this plot
            plot_path (str): path to the plot that will be created
        """
        # show and hide as needed
        for repr_index in list_representation_indices:
            self._show_representation(repr_index)
        not_on_list = set(range(self.num_representations)) - set(list_representation_indices)
        for repr_index in not_on_list:
            self._hide_representation(repr_index)

        # render
        self.total_file_text += f"render TachyonInternal {plot_path}"

    def _show_representation(self, representation_index: int) -> None:
        """
        Helper function, show a particular representation among already created ones. If not hidden does nothing.
        
        Args:
            representation_index (int): which representation to show.
        """
        self.total_file_text += f"\nmol showrep 0 {representation_index} 1\n"

    def _hide_representation(self, representation_index: int) -> None:
        """
        Helper function, hide a particular representation among already created ones. If already hidden does nothing.

        Args:
            representation_index (int): which representation to hide.
        """
        self.total_file_text += f"\nmol showrep 0 {representation_index} 0\n"

    def load_translation_rotation_script(self, path_translation_rotation_script: str = None) -> None:
        """
        Optionally, if there is a (manually created) sequence of translations/rotations/scaling in VMD format, you can 
        load it here. Useful to create multiple plots in exactly same orientation. If not provided, the default VMD 
        orientation is used.
        
        Args:
            path_translation_rotation_script (str): the path to the VMD script
        """
        self.translations_rotations_script = path_translation_rotation_script
        
    def prepare_frame_script(self, vmd_name: str, plot_name: str, num_frames: int, box_limits: list,
                             draw_m1: bool = True, draw_m2: bool = True, draw_rectangular_box: bool = True,
                             gridpoints: NDArray = None, zoom_level: int = 1, translation_rotation_script: str = None):
        """
        Plot one or multiple (overlapping) frames. Assumes the script will be run with additional "structure" first 
        frame which should be ignored (used for the reference so that the initial zoom level is always the same) and 
        one or more additional frames that should all be plotted.
        """
        self._start_new_file()
        if translation_rotation_script:
            self.load_translation_rotation_script(translation_rotation_script)
        if draw_m1:
            # plot only one non-zero frame since they are all the same
            self._add_representation(first_molecule=True, second_molecule=False, periodic="Z",
                                       representation="DynamicBonds 1.600000 0.300000 6.000000", trajectory_frames=[1])
        if draw_m2:
            # plot all provided frames except 0
            self._add_representation(first_molecule=False, second_molecule=True, periodic="xyzXYZ",
                                       representation="Licorice", trajectory_frames=list(range(1, num_frames+1)))
        if draw_rectangular_box is not None:
            self.add_box(*box_limits)
        if gridpoints is not None:
            self.add_grid(gridpoints)
        if translation_rotation_script:
            self._add_rotations_translations()
        self._zoom_on_box(box_limits[0], box_limits[1], zoom_level)
        # we only have two representations even if the second one potentially uses multiple frames
        self._render_representations([0, 1], plot_path=plot_name)
        self.write_text_to_file(vmd_name)

    def prepare_eigenvector_script(self, abs_eigenvector_frames: NDArray, pos_eigenvector_frames: NDArray,
                                   neg_eigenvector_frames: NDArray, plot_names: list) -> None:
        """
        Everything you need to plot eigenvectors:
        - make plotting pretty
        - translate, scale and rotate appropriately to have an optimal fig
        - add indices of zeroth eigenvector at max absolute values
        - add indices of higher eigenvectors at most positive/most negative values
        - render the plots

        INDICES OF FRAMES ALREADY MUST HAVE +1 IF NEEDED

        Args:
            abs_eigenvector_frames (NDArray): a 1D array of integers for representative cells of the 0th eigenvector
            pos_eigenvector_frames (NDArray): a 2D array of integers, each row representing most positive cells of the
                ith eigenvector with i=1,2,3... Must have same length as neg_eigenvector_frames.
            neg_eigenvector_frames (NDArray): a 2D array of integers, each row representing most negative cells of the
                ith eigenvector with i=1,2,3... Must have same length as pos_eigenvector_frames.
            plot_names (list): file paths for all the renders. Must have the length of neg_eigenvector_frames + 1
        """
        assert len(pos_eigenvector_frames) == len(neg_eigenvector_frames)
        assert len(plot_names) == len(neg_eigenvector_frames) + 1

        # add first molecule without any special colors etc
        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=1)

        # add zeroth eigenvector without any special colors
        self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=abs_eigenvector_frames, representation="Licorice")

        # for the rest add one red, one blue
        for pos_frames, neg_frames in zip(pos_eigenvector_frames, neg_eigenvector_frames):
            self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorID", color="blue",
                                     trajectory_frames=pos_frames, representation="Licorice")
            self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorID", color="red",
                                     trajectory_frames= neg_frames, representation="Licorice")

        self._add_rotations_translations()

        # render the zeroth eigenvector
        self._render_representations([0, 1], plot_names[0])

        # render the rest of eigenvectors
        last_used_representation = 1
        for i, plot_name in enumerate(plot_names[1:]):
            # each render contains first molecule in representation 0 and second molecule in representation 1, 2, 3 ...
            self._render_representations([0, last_used_representation+1, last_used_representation+2], plot_name)
            last_used_representation += 2

    def prepare_evec_0(self, num_structures: int, plot_name: str) -> None:
        """
        This is for evoking vmd with all individual structures relating to eigenvector 0, eg

        vmd structure_22.xyz structure 90723.xyz structure_7356.xyz
        """

        self._add_representation(first_molecule=True, second_molecule=False,
                                 trajectory_frames=[0])
        self._add_representation(first_molecule=False, second_molecule=True,
                                 trajectory_frames=list(range(1, num_structures + 1)))
        self._add_rotations_translations()
        # render the zeroth eigenvector
        self._render_representations([0, 1], plot_name)

    def prepare_evec_pos_neg(self, num_structures_pos: int, num_structures_neg: int, plot_name: str) -> None:
        """
        This is for evoking vmd with all individual structures relating to other eigenvectors, eg

        vmd structure_22.xyz structure 90723.xyz structure_7356.xyz
        """
        self._add_representation(first_molecule=True, second_molecule=False,
                                 trajectory_frames=list(range(1, num_structures_pos+num_structures_neg+1)))
        self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorID", color="blue",
                                 trajectory_frames=list(range(1, num_structures_pos+1)))
        self._add_representation(first_molecule=False, second_molecule=True, coloring="ColorID", color="red",
                                 trajectory_frames=list(range(num_structures_pos+1,
                                                              num_structures_pos+num_structures_neg+1)))
        self._add_rotations_translations()
        # render the zeroth eigenvector
        self._render_representations([0, 1, 2], plot_name)

    def prepare_clustering_script(self, indices_per_cluster: list, color_per_cluster: list, plot_name_all_together: str,
                                  plot_names_individual: list) -> None:
        """
        Make a script that shows you each cluster in its representative color and also all clusters together.

        INDICES OF FRAMES ALREADY MUST HAVE +1 IF NEEDED

        Args:
            indices_per_cluster (list): a list of sublists (cannot be an array because different lengths), each sublist
            containing trajectory indices belonging to 0th, 1st, 2nd .... cluster
            color_per_cluster (list): same length as indices_per_cluster list, provides the color to be used for the
            corresponding cluster eg. ["red", "blue", "black" ....]
            plot_name_all_together (str): path to the plot of all clusters together
            plot_names_individual (list): paths to the plots of 0th, 1st, 2nd ... cluster
        """
        assert len(color_per_cluster) == len(indices_per_cluster) == len(plot_names_individual), \
            f"{len(color_per_cluster)}!={len(indices_per_cluster)}!={len(plot_names_individual)}"

        # first molecule in a normal color
        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=0)

        # now add second molecule separately for each cluster:
        for cluster_indices, cluster_color in zip(indices_per_cluster, color_per_cluster):
            self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=cluster_indices,
                                     coloring="ColorID", color=cluster_color)

        self._add_rotations_translations()

        # plot all together
        all_representations = list(range(self.num_representations))
        self._render_representations(all_representations, plot_name_all_together)

        # plot individually
        for cluster_i, plot_name in enumerate(plot_names_individual):
            self._render_representations([0, cluster_i+1], plot_name)

    def plot_frames_individually(self, my_indices: list, plot_names: list) -> None:
        """
        Renders a figure with usual coloring for each grid point in my_indices.

        INDICES OF FRAMES ALREADY MUST HAVE +1 IF NEEDED

        Args:
            my_indices (list): a list of indices like [15, 7, 385, 22] describing a path or sequence on a grid,
            integers are grid indices
            plot_names (list): a list of file names to which renders should be saved, should have the same length as
                my_path
        """
        assert len(my_indices) == len(plot_names)

        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=1,
                                 representation="DynamicBonds 1.600000 0.300000 6.000000")

        for path_index in my_indices:
            self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=path_index, representation="Licorice")

        self._add_rotations_translations()

        for i, plot_path in enumerate(plot_names):
            # each render contains first molecule in representation 0 and second molecule in representation 1, 2, 3 ...
            self._render_representations([0, i+1], plot_path)

    def plot_m2_frames_individually(self, my_indices: list, plot_names: list) -> None:
        """
        Renders a figure with usual coloring for each grid point in my_indices.

        INDICES OF FRAMES ALREADY MUST HAVE +1 IF NEEDED

        Args:
            my_indices (list): a list of indices like [15, 7, 385, 22] describing a path or sequence on a grid,
            integers are grid indices
            plot_names (list): a list of file names to which renders should be saved, should have the same length as
                my_path
        """
        assert len(my_indices) == len(plot_names)

        for path_index in my_indices:
            self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=path_index, representation="Licorice")

        self._add_rotations_translations()

        for i, plot_path in enumerate(plot_names):
            # each render contains first molecule in representation 0 and second molecule in representation 1, 2, 3 ...
            self._render_representations([i], plot_path)

    def _add_COM_of_m2_overlappig_frames(self, trajectory_frames: ArrayLike = None):
            """
            Add a little dot at COm of molecule2 for the selected frames.
            """
            trajectory_frames = [str(int(frame)) for frame in trajectory_frames]
            self.total_file_text += f"""                                                          
    set sel [atomselect top "{self.index_second_molecule}"]                                   
    $sel set occupancy [$sel get mass]
    set frames {{ {' '.join(trajectory_frames)} }}
    foreach i $frames {{
        $sel frame $i
        set com [measure center $sel weight mass]
        graphics top sphere $com radius 0.3
        graphics top color red
    }}                                                                        
    """


    def plot_multiple_overlappig_frames(self, frame_indices: list, plot_path: str, only_COM_of_m2: bool = False) -> None:
        """
        Renders a figure with usual coloring for each grid point in my_indices.

        INDICES OF FRAMES ALREADY MUST HAVE +1 IF NEEDED
        """

        self._add_representation(first_molecule=True, second_molecule=False, trajectory_frames=1,
                                 representation="DynamicBonds 1.600000 0.300000 6.000000")

        if only_COM_of_m2:
            self._add_COM_of_m2_overlappig_frames(frame_indices)
        else:
            for path_index in frame_indices:
                self._add_representation(first_molecule=False, second_molecule=True, trajectory_frames=path_index,
                                         representation="Licorice")

        self._add_rotations_translations()
        self._render_representations(list(range(len(frame_indices)+1)), plot_path=plot_path)

    def _add_rotations_translations(self):
        """
        If the script was loaded before, it will be used, else nothing happens.
        """
        if self.translations_rotations_script:
            with open(self.translations_rotations_script, "r") as f:
                contents = f.read()
            self.total_file_text += contents
    
