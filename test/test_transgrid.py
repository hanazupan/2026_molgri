
import numpy as np

from molgri.transgrid import TranslationObject
from molgri.constants import ICO_PERFECT_NUM


def test_radial_grid():
    """
    Test that the inputs of R_min, R_max and N_radial are interpreted correctly.
    """
    N_ray = 1
    R_min = 2.5
    R_max = 7.3
    N_rad = 15

    my_translations = TranslationObject(N_ray, R_min, R_max, N_rad)
    assert len(my_translations.radial_grid) == N_rad
    assert np.isclose(my_translations.radial_grid[0], R_min)
    assert np.isclose(my_translations.radial_grid[-1], R_max)



def test_ray_grid():
    """
    Test that the inputs of R_min, R_max and N_radial are interpreted correctly.
    """
    N_ray = 22
    R_min = 1
    R_max = 1
    N_rad = 1

    my_translations = TranslationObject(N_ray, R_min, R_max, N_rad)
    assert len(my_translations.unit_sphere_grid) == N_ray
    # all should have radius 1
    assert np.allclose(np.linalg.norm(my_translations.unit_sphere_grid, axis=1), 1.0)


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
    adj = to.adjacency.toarray()

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


def test_single_direction():
    my_translations = TranslationObject(1, 2, 3, 5)

    # only the two side diagonals
    expected_adjacency = [[0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 1],
                          [0, 0, 0, 1, 0]]

    # the only adjacency is in radial name
    assert np.allclose(expected_adjacency, my_translations.adjacency.toarray())

    # todo test one grid visually
    radii = np.array([0.0, 2.125, 2.375, 2.625, 2.875, 3.125])
    assert np.allclose(radii, my_translations._get_between_radii())
    expected_volumes = 4/3*np.pi * radii**3

    expected_volumes_shells = expected_volumes[1:]-expected_volumes[:-1]
    assert np.allclose(expected_volumes_shells, my_translations.volumes)
    assert np.allclose(0.25*np.array(expected_adjacency), my_translations.distances.toarray())

    expected_surfaces = 4 * np.pi * radii ** 2

    expected_surfaces_array  = [[0, expected_surfaces[1], 0, 0, 0],
                                [expected_surfaces[1], 0, expected_surfaces[2], 0, 0],
                                [0, expected_surfaces[2], 0, expected_surfaces[3], 0],
                                [0, 0, expected_surfaces[3], 0, expected_surfaces[4]],
                                [0, 0, 0, expected_surfaces[4], 0]]
    assert np.allclose(expected_surfaces_array, my_translations.surfaces.toarray())


def test_perfect_division():
    #approximate volumes, areas etc from perfect division
    for N_dir in ICO_PERFECT_NUM:
        my_translations = TranslationObject(N_dir, 2.5, 3.5, 3)
        hull_radii = np.array([0, 2.75, 3.25, 3.75])

        # volumes
        volume_difference = hull_radii[1:]**3 - hull_radii[:-1]**3
        ideal_volume = 4/3 * np.pi * volume_difference / N_dir

        ideal_side_surface = 1.10714872 / 2 * hull_radii[1:]**2

        neighbour_types = my_translations.neighbour_types.copy().toarray()

        for i in range(3):
            assert np.isclose(np.mean(my_translations.volumes[i*N_dir:(i+1)*N_dir]), ideal_volume[i])
            assert np.isclose(np.sum(my_translations.volumes[i * N_dir:(i+1)*N_dir]), N_dir*ideal_volume[i])
            assert np.std(my_translations.volumes[i * N_dir:(i+1)*N_dir]) < 0.12

            surfaces = my_translations.surfaces.toarray()
            side_surfaces = surfaces[neighbour_types == 1]
            side_surfaces = side_surfaces[i * len(side_surfaces) // 3:(i + 1) * len(side_surfaces) // 3]
            sphere_surfaces = surfaces[neighbour_types == 2]
            sphere_surfaces = sphere_surfaces[i*len(sphere_surfaces)//3:(i+1)*len(sphere_surfaces)//3]
            print(surfaces[neighbour_types == 2].shape)
            ideal_side_surface =  np.arccos(np.sqrt(5)/3) / 2 * (hull_radii[i+1]**2 - hull_radii[i]**2)
            print("side", np.mean(side_surfaces), ideal_side_surface)
            print(np.mean(sphere_surfaces), 4*np.pi*hull_radii[i+1]**2/N_dir)
            # TODO: finish


def test_visual_example():
    to = TranslationObject(20, 2, 7, 3)
    #to.plot(show_node_numbers=True, show_hulls=False, show_distances=False, show_hulls_of_cells=[0, 3, 5, 11])

    assert to.volumes[33] > to.volumes[0]
    assert to.volumes[5] > to.volumes[11]
    assert to.volumes[24] > to.volumes[33]

    areas = to.surfaces.toarray()
    assert areas[33][53] > areas[33][13] > areas[33][31] > areas[33][0]
    assert areas[0][5] < areas[11][9] < areas[0][9]

    dist = to.distances.toarray()
    assert np.allclose([dist[8][28], dist[48][28], dist[21][41]], dist[8][28])
    assert dist[54][53] < dist[54][45]
    assert dist[51][42] < dist[51][53]


if __name__ == "__main__":
    test_radial_grid()
    test_ray_grid()
    test_in_betweeen()
    test_adjacency()
    test_single_direction()
    test_perfect_division()
    test_visual_example()
    print("All tests successful.")