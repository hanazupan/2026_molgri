
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


def test_in_betweeen():
    N_ray = 1
    R_min = 2.5
    R_max = 7.5
    N_rad = 3

    my_translations = TranslationObject(N_ray, R_min, R_max, N_rad)

    between_radii = my_translations._get_between_radii()
    assert np.allclose([0.0, 3.75, 6.25, 8.75], between_radii)


def test_adjacency():
    np.random.seed(1)
    to = TranslationObject(20, 2, 7, 3)
    adj = to.get_adjacencies().toarray()

    # is symmetric
    assert np.allclose(adj, adj.T)

    # in row (and column) 13, the only neighbours should be 1, 14, 19, 12, 18, 33
    array_length = 20*3
    true_indices = [1, 14, 19, 12, 18, 33]
    expected = np.zeros(array_length, dtype=bool)
    expected[true_indices] = True

    assert np.array_equal(adj[13], expected)

    # in row (and column) 33, the only neighbours should be 13, 53, 21, 34, 39, 32, 38
    true_indices = [13, 53, 21, 34, 39, 32, 38]
    expected = np.zeros(array_length, dtype=bool)
    expected[true_indices] = True

    assert np.array_equal(adj[33], expected)



    # if you want to check visually
    #to.plot(show_node_numbers=True)


if __name__ == "__main__":
    test_radial_grid()
    test_ray_grid()
    test_in_betweeen()
    test_adjacency()
    print("All tests successful.")