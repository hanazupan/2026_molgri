![Coverage Status](https://img.shields.io/coverallsCoverage/github/hanazupan/2026_molgri)
![issues](https://img.shields.io/github/issues/hanazupan/2026_molgri)
![license](https://img.shields.io/github/license/hanazupan/2026_molgri)
![activity](https://img.shields.io/github/last-commit/hanazupan/2026_molgri)
![release](https://img.shields.io/github/v/release/hanazupan/2026_molgri)

A rewrite of the molgri package.

## Input files
Prepare two .gro files of each molecule separately. They do not need to be centered, but should already be optimized on their own (bond lengths etc.). Also prepare an index file. The first four elements must be: System (containing all atoms), Other, MOL1 (containing elements of molecule 1) and MOL2 (containing elements of molecule 2)

## Gromacs
Pseudotrajectories are prepared in such a way that the center of mass of molecule 1 is at (0,0,0). The simulation starts at (0,0,0) and extends to (Lx, Ly, Lz), so the structures are NOT centered in the box. In the simulation, the structures automatically get centered during the simulation but after the simulation they are again aligned so that the center of mass of molecule 1 is at (0, 0, 0).
