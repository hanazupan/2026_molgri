
import numpy as np

from molgri.transgrid import TranslationObject


def test_radial_grid():
    """
    Test that the inputs of R_min, R_max and N_rad are interpreted correctly.
    """
    N_ray = 1
    R_min = 2.5
    R_max = 7.3
    N_rad = 15

    my_translations = TranslationObject(N_ray, R_min, R_max, N_rad)
    assert len(my_translations.radial_points) == N_rad
    assert np.isclose(my_translations.radial_points[0], R_min)
    assert np.isclose(my_translations.radial_points[-1], R_max)

def test_ray_grid():
    """
    Test that the inputs of R_min, R_max and N_rad are interpreted correctly.
    """
    N_ray = 22
    R_min = 1
    R_max = 1
    N_rad = 1

    my_translations = TranslationObject(N_ray, R_min, R_max, N_rad)
    assert len(my_translations.ray_points) == N_ray
    # all should have radius 1
    assert np.allclose(np.linalg.norm(my_translations.ray_points, axis=1), 1.0)


# test that no matter how many levels, each ico gridpoint has 5 to 6 neighbours


if __name__ == "__main__":
    test_radial_grid()
    test_ray_grid()
    print("All tests successful.")