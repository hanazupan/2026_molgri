![Coverage Status](https://img.shields.io/coverallsCoverage/github/hanazupan/2026_molgri)
![issues](https://img.shields.io/github/issues/hanazupan/2026_molgri)
![license](https://img.shields.io/github/license/hanazupan/2026_molgri)
![activity](https://img.shields.io/github/last-commit/hanazupan/2026_molgri)
![release](https://img.shields.io/github/v/release/hanazupan/2026_molgri)

A rewrite of the molgri package.

## Input files
Prepare two .gro files of each molecule separately. They do not need to be centered, but should already be optimized on their own (bond lengths etc.). Also prepare an index file. The first four elements must be: System (containing all atoms), Other, MOL1 (containing elements of molecule 1) and MOL2 (containing elements of molecule 2)