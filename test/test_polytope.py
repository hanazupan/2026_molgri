import numpy as np

from molgri.polytope import IcosahedronPolytope
from molgri.utils import all_rows_unique, all_row_norms_equal_k

def test_full_division_ico():
    """
    This applies to polytopes with full number of points. We test that after each level of division, the number of
    points is as expected.
    """
    # ico tests
    expected_num_of_points = [12, 42, 162]
    # the first 12 nodes have 5 edges, the rest 6; divide by two since you count each edge twice
    expected_num_of_edges = [12*5/2, (12*5+(42-12)*6)/2, (12*5+(162-12)*6)/2]

    ico = IcosahedronPolytope()

    for i in range(3):
        if i != 0:
            ico.divide_edges()
        assert ico.G.number_of_nodes() == expected_num_of_points[i], f"At level {i} ico should have {expected_num_of_points[i]} nodes, not {ico.G.number_of_nodes()}"
        # those points are unique
        all_rows_unique(ico.get_nodes(projection=True))

        assert ico.G.number_of_edges() == expected_num_of_edges[i], f"At level {i} ico should have {expected_num_of_edges[i]} edges, not {ico.G.number_of_edges()}"


def test_removing_points_ico():

    for wished_num_points in [5, 12, 33, 59]:
        ico = IcosahedronPolytope()
        ico.create_exactly_N_points(wished_num_points)
        my_nodes = ico.get_nodes()
        my_projected_points = ico.get_nodes(projection=True)

        # you get the expected number of points
        assert len(my_nodes) == wished_num_points
        assert len(my_projected_points) == wished_num_points

        # these points are on unit sphere
        all_row_norms_equal_k(my_projected_points, 1)

        # these points are unique
        all_rows_unique(my_nodes)
        all_rows_unique(my_projected_points)


if __name__ == "__main__":
    test_full_division_ico()
    test_removing_points_ico()