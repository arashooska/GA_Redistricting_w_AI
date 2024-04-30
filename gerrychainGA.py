# ------------------------- Import Libraries -------------------------
import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import geopandas as gpd
import time, statistics
import numpy as np
from gerrychain.metrics import mean_median, efficiency_gap

# ------------------------- Define Constants -------------------------
POPULATION_TOLERANCE = 0.2
NUM_GA_DISTS = 14
NUM_STEPS = 10

# ---------------------------- Load Graph ----------------------------
def load_graph(file_path):
    return Graph.from_json(file_path)

# ------------------ Calculate The Total Population ------------------
def find_tot_pop(graph):
    return sum([graph.nodes()[v]['TOTPOP'] for v in graph.nodes()])

# ------------------- Set Up The Initial Partition -------------------
def initialize_partition(elections, graph):
    my_updaters = {"population": updaters.Tally("TOTPOP", alias="population"), 
            "cut_edges": cut_edges, 
            "district Black": Tally("BVAP", alias="district Black")
        }
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    initial_partition = Partition(
    graph,
    assignment="CD",
    updaters=my_updaters
    )
    return initial_partition

# ----------------------- Execute Random Walk ------------------------
def execute_random_walk(initial_partition, num_steps, num_dist, graph):
    tot_pop = find_tot_pop(graph)
    ideal_pop = tot_pop / num_dist

    random_walk_partial = partial(recom,
                                  pop_col="VAP",
                                  epsilon=POPULATION_TOLERANCE,
                                  pop_target=ideal_pop,
                                  node_repeats=2,
                                  )

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, POPULATION_TOLERANCE, pop_key="population")
    
    random_walk = MarkovChain(proposal=random_walk_partial,
                              constraints=[pop_constraint],
                              accept=always_accept,
                              initial_state=initial_partition,
                              total_steps=num_steps)

    print(f"Successfully ran chain for {num_steps} steps!")
    return random_walk

# ----------------------- Gather Relevant Data -----------------------
# Gather relevant data for each partition in the random walk
def gather_data(random_walk, election, data_type):
    gathered_data = []
    for partition in random_walk:
        if data_type == "eg":
            gathered_data.append(efficiency_gap(partition[election.name]))
        elif data_type == "mm":
            gathered_data.append(mean_median(partition[election.name]))
        elif data_type == "D_wins":
            gathered_data.append(partition[election.name].wins("Democratic"))
        elif data_type == "cut_edges":
            gathered_data.append(len(partition["cut_edges"]))
    return gathered_data

# -------------------------- Main Function ---------------------------
def main():
    # Load graph
    ga_graph = load_graph("./GA_clean.json")

    #Start timer
    start_time = time.time()

    # Define elections
    """
    Elections from GA_clean.json:
        "G18GOVRKEM": 193,
        "G18GOVDABR": 309,
        "G18ATGRCAR": 206,
        "G18ATGDBAI": 301,
        "G18AGRRBLA": 210,
        "G18AGRDSWA": 295,
    """
    elections = [
        Election("G18GOV", {"Democratic": "G18GOVDABR", "Republican": "G18GOVRKEM"}),
        Election("G18ATG", {"Democratic": "G18ATGDBAI", "Republican": "G18ATGRCAR"}),
        Election("G18AGR", {"Democratic": "G18AGRDSWA", "Republican": "G18AGRRBLA"}),
    ]

    # Create initial partition
    initial_partition = initialize_partition(elections, ga_graph)

    # Execute random walk
    random_walk = execute_random_walk(initial_partition, NUM_STEPS, NUM_GA_DISTS, ga_graph)

    # Gather relevant data
    EGs = []
    MMs = []
    D_wins = []
    cut_edges = []
    for election in elections:
        # Efficency Gap
        EGs = gather_data(random_walk, election, "eg")
        # Mean Median
        MMs = gather_data(random_walk, election, "mm")
        # Dem-won Districts
        D_wins = gather_data(random_walk, election, "D_wins")

    # Cut Edges
    cut_edges = gather_data(random_walk, election, "cut_edges")

    # Print the gathered data
    for election, EG, MM, D_win, cut_edge in zip(elections, EGs, MMs, D_wins, cut_edges):
        print(f"\n{election.name}:")
        print(f"Efficiency Gap: {EG}")
        print(f"Mean Median: {MM}")
        print(f"Democratic-won Districts: {D_win}")
        print(f"Cut Edges: {cut_edge}")
    
    # Measure execution time
    end_time = time.time()
    print("The time of execution of the program is:", (end_time - start_time) / 60, "mins")

if __name__ == "__main__":
    main()