![Coverage Status](https://img.shields.io/coverallsCoverage/github/hanazupan/2026_molgri)
![issues](https://img.shields.io/github/issues/hanazupan/2026_molgri)
![license](https://img.shields.io/github/license/hanazupan/2026_molgri)
![activity](https://img.shields.io/github/last-commit/hanazupan/2026_molgri)
![release](https://img.shields.io/github/v/release/hanazupan/2026_molgri)

A rewrite of the molgri package.

## Dependencies

It is best to create a virtual environment to deal with dependencies.

## Where to start

To use the code, the best starting point is the workflow/Snakefile. It defines the end products of a pipeline (eg. implied timescales plots, molecular structures with lowest energies ...). The Snakefile depends on other snakemake files in workflows/rules. The rules define how some input files (e.g. rate matrix) are connected to output files (e.g. eigenvalues and eigenvectors). The imports in snakemake files use the molgri package, where the functionality is implemented. To use molgri functionality, you can do the following:
- edit the Snakefile by commenting out the outputs you don't need or modifying the parameters or the outputs you want
- Provide input files - see below
- Prepare a config file - see examples in workflow/config and edit yours as needed

When this is done you simply need to run


    snakemake --configfile <path_to_your_configfile> 

Let's talk about these three steps a bit more.


### Edit Snakemake file

The Snakefile contains a rule all which has no outputs, only inputs. These inputs must all be generated for a run to succeed. When starting with a new system, it is best to begin with only the first few inputs and slowly expand after debugging any occurring problems. Also don't forget the commas between inputs!

### Input files

Input files are provided in the inputs/ directory. Note that we are always working with two molecules at the same time.

In inputs/one_molecule_structures/, provide separate structure files of the molecules in your system (one molecule per file).  They do not need to be centered, but should already be optimized on their own (bond lengths etc.). Name the structures meaningfully, eg. water.xyz or graphene.gro. The name of your molecule files is referenced in the config file as molecule_1 (the stationary molecule) and molecule_2 (the moving molecule).

In inputs/default_gromacs/ create a folder named after your two molecules: <molecule1>_<molecule2>. Inside this folder you need gromacs input files: topology.top for the system of both molecules, production.mdp for run parameters, and index.mdp. The first five elements of the index must be: System (containing all atoms), Other, MOL1 (containing all atoms of molecule 1), MOL2 (containing all atoms of molecule 2) and atom_1 (containing only the very first atom). See examples for other files.

In inputs/vmd_scripts, create a folder named after your two molecules: <molecule1>_<molecule2>. Then you can create as many script{i}.log files as you need. Each of these files contains instructions for vmd how to rotate, translate and scale the structure to get the desired view. This view is referenced in many plot output names as view_i. For a start, you can just create an empty file scrip1.log and use the default view. 

Although strictly speaking not an input, it makes sense to create a trajectory of your system separately, e.g. on the computer cluster.The structure and trajectory of combined, simulated system should be provided in nobackup/<molecule1>_<molecule2>/SIMULATION as simulation.gro, simulation.tpr and raw_trajectory.xtc.

### Edit config file

In the config file, make sure molecule_1 and molecule_2 refer to the names of your molecular structures. You can also adapt the grid and some other parameters here.

## Other notes
Pseudotrajectories are prepared in such a way that the center of mass of molecule 1 is at (0,0,0). The simulation starts at (0,0,0) and extends to (Lx, Ly, Lz), so the structures are NOT centered in the box. In the simulation, the structures automatically get centered during the simulation but after the simulation they are again aligned so that the center of mass of molecule 1 is at (0, 0, 0).

Units of distance are A. Units of energy are kJ/mol.