import timeit

import numpy as np
import pandas as pd

OUTPUT_FOLDER = "outputs/timing/"

N_SPH = np.linspace(5, 500, 20, dtype=int)
N_RAD = np.linspace(2, 20, 10, dtype=int)
N_ROT = np.linspace(5, 500, 20, dtype=int)

rule all:
    input:
        f"{OUTPUT_FOLDER}network_creation_timing.csv"


rule time_network_generation:
    output:
        timing_record = f"{OUTPUT_FOLDER}network_creation_{{N_sph}}_{{N_rad}}_{{N_rot}}.csv",
    run:
        from molgri.full_network import create_full_network
        from molgri.utils import random_sphere_points, random_quaternions

        NUM_REPEATS = 10
        NUM_RADIAL = int(wildcards.N_rad)
        NUM_SPHERICAL = int(wildcards.N_sph)
        NUM_ROTATIONAL = int(wildcards.N_rot)

        columns = ["N spherical", "N radial", "N rotational", "Time network creation [ms]", "Time network + edge creation [ms]"]

        # create random points outside of the timing, that's the matter of polytope generation
        radial_points = np.linspace(1.5,4.5,num=NUM_RADIAL)
        spherical_points = random_sphere_points(NUM_SPHERICAL)
        quaternions = random_quaternions(NUM_ROTATIONAL)

        def process_to_time():
            create_full_network(spherical_points,radial_points,quaternions)

        elapsed = timeit.repeat(process_to_time, number=1, repeat=NUM_REPEATS)

        def second_process_to_time():
            full_network = create_full_network(spherical_points,radial_points,quaternions)
            full_network.distance_matrix()
            full_network.surface_matrix()
            full_network.volumes()

        elapsed2 = timeit.repeat(lambda: second_process_to_time, number=1, repeat=NUM_REPEATS)

        data = []
        for time1, time2 in zip(elapsed, elapsed2):
            data.append([NUM_SPHERICAL, NUM_RADIAL, NUM_ROTATIONAL, time1, time2])
        data = np.array(data)

        pd.DataFrame(data, columns=columns).to_csv(output.timing_record)


rule df_all_creation_times:
    input:
        expand(f"{OUTPUT_FOLDER}network_creation_{{N_sph}}_{{N_rad}}_{{N_rot}}.csv",N_sph=N_SPH,N_rad=N_RAD,N_rot=N_ROT)
    output:
        full_timing = f"{OUTPUT_FOLDER}network_creation_timing.csv"
    run:
        all_dataframes = []
        for file in input:
            df = pd.read_csv(file, index_col=0)
            all_dataframes.append(df)
        full_df = pd.concat(all_dataframes, ignore_index=True)
        full_df.to_csv(output.full_timing)

rule plot_timing:
    input:
        dataframe = f"{OUTPUT_FOLDER}{{some_df_name}}.csv"
    output:
        plot = f"{OUTPUT_FOLDER}{{some_df_name}}.png"
    run:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        df = pd.read_csv(file,index_col=0)

        columns = ["N spherical", "N radial", "N rotational", "Time network creation [ms]",
                   "Time network + edge creation [ms]"]

        # three plots relative to each number of points
        fig = make_subplots(rows=1,cols=3)

        all_cols = (1,2,3)
        all_x_variables = columns[:3]

        for col_i, col_des in zip(all_cols, all_x_variables):
            fig.add_trace(
                go.Scatter(x=df[col_des],y=df["Time network creation [ms]"], marker=dict(color="black")),
                row=1,col=col_i)
            fig.add_trace(
                go.Scatter(x=df[col_des],y=df["Time network + edge creation [ms]"], marker=dict(color="blue")),
                row=1,col=col_i)

        # todo convert units of time if needed


        fig.update_layout(height=600,width=800)
        fig.write_image(output.plot,dpi=600)

