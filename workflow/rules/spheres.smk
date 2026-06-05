import numpy as np
import pandas as pd
from molgri.utils.spheres import random_sphere_points

rule sphere_patch_xyz:
    output:
        "sphere_patch_100_new.xyz"
    params:
        n=100,
        patch_angle_deg=60
    run:
        points = random_sphere_points(params.n)

        # north pole direction
        patch_center = np.array([0, 0, 1])

        # convert angle threshold to cosine
        cos_threshold = np.cos(np.deg2rad(params.patch_angle_deg))

        atom_types = []

        for p in points:
            # dot product gives cos(theta)
            cos_theta = np.dot(p, patch_center)

            if cos_theta >= cos_threshold:
                atom_types.append("Pat")   # patch atoms
            else:
                atom_types.append("Sph")   # normal sphere atoms

        with open(output[0], "w") as f:
            f.write(f"{len(points)}\n")
            f.write("Patchy sphere\n")

            for atom, p in zip(atom_types, points):
                f.write(f"{atom} {p[0]} {p[1]} {p[2]}\n")