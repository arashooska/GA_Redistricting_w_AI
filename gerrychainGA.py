# ------------------------- Import Libraries -------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from gerrychain import Graph, Partition, updaters, constraints, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import time
import numpy as np
import pandas as pd
from gerrychain.metrics import mean_median, efficiency_gap

# ------------------------- Define Constants -------------------------
POPULATION_TOLERANCE = 0.2
NUM_GA_DISTS = 14
NUM_STEPS = [10, 1000, 2000, 5000]

DATA_DICT_FORMAT = {"eg": [], "mm": [], "d_wins": [], "r_wins": [], "cut_edges": [], "dem_vote_share_pd": [], "num_majority_black": []}

# ---------------------------- Load Graph ----------------------------
def load_graph(file_path):
    return Graph.from_json(file_path)

# ------------------ Calculate The Total Population ------------------
def find_tot_pop(graph):
    return sum([graph.nodes()[v]['TOTPOP'] for v in graph.nodes()])

def majority_bvap_districts(partition):
    """
    Returns the number of districts in the partition that have a majority
    Black voting-age population (BVAP > 50%).
    """
    num_majority_black = 0
    for district_id, district_dict in partition["BVAP"].items():
        vap = partition["VAP"][district_id]
        bvap = district_dict
        if bvap / vap > 0.5:
            num_majority_black += 1
    return num_majority_black

# ------------------- Set Up The Initial Partition -------------------
def initialize_partition(elections, graph):
    my_updaters = {"population": updaters.Tally("TOTPOP", alias="population"), 
            "cut_edges": cut_edges, 
            "VAP": Tally("VAP", alias="VAP"),
            "BVAP": Tally("BVAP", alias="BVAP"),
            "majority_bvap_districts": majority_bvap_districts
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

def get_plan_stats(dict_to_update, partition, election_name):
    # return_dict = {"eg": [], "mm": [], "d_wins": [], "r_wins": [], "cut_edges": [], "dem_vote_share_pd": [], "num_majority_black": []}
    
    # Calculate statistics on the initial partition
    dict_to_update["eg"].append(efficiency_gap(partition[election_name]))
    dict_to_update["mm"].append(mean_median(partition[election_name]))
    dict_to_update["d_wins"].append(partition[election_name].wins("Democratic"))
    dict_to_update["r_wins"].append(partition[election_name].wins("Republican"))
    dict_to_update["cut_edges"].append(len(partition["cut_edges"]))
    dict_to_update["dem_vote_share_pd"].append(sorted(partition[election_name].percents("Democratic")))
    dict_to_update["num_majority_black"].append(partition["majority_bvap_districts"])

    return dict_to_update



def generate_output_fname(election, num_steps, metric):
    return f"./output/{num_steps}steps/{election}_{metric}.png"

# ------------------- Graphing Fns ------------------------
def graph_marginal_box_plots(vote_shares, election, steps):
        
    data = pd.DataFrame(vote_shares)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw 50% line
    ax.axhline(0.5, color="#cccccc")

    # Draw boxplot
    data.boxplot(ax=ax, positions=range(len(data.columns)))

    # Draw initial plan's Democratic vote %s (.iloc[0] gives the first row, which corresponds to the initial plan)
    plt.plot(data.iloc[0], "ro")

    # Annotate
    ax.set_title(f"{election} election vote shares ({steps} Steps)")
    ax.set_ylabel("Democratic vote %")
    ax.set_xlabel("Sorted districts")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    #plt.show()
    plt.savefig(generate_output_fname(election, steps, "district_vs_boxplot"))
    plt.close()

def graph_eg(election, data_dict, initial_eg, steps):
    plt.figure(figsize=(10, 8))
    plt.hist(data_dict['eg'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
    plt.title(f"Efficiency Gap Histogram - {election}")
    plt.xlabel("Efficiency Gap")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(x=initial_eg[0], color='r', linestyle='--', label='Initial Plan')
    plt.legend()

    plt.savefig(generate_output_fname(election, steps, "eg_histogram"))
    plt.close()

def graph_mm(election, data_dict, initial_mm, steps):
    plt.figure(figsize=(10, 8))
    plt.hist(data_dict['mm'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
    plt.title(f"Mean Median Histogram - {election}")
    plt.xlabel("Mean Median")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(x=initial_mm[0], color='r', linestyle='--', label='Initial Plan')
    plt.legend()

    plt.savefig(generate_output_fname(election, steps, "mm_histogram"))
    plt.close()

def graph_d_vs_r_wins(election, data_dict, initial_d_wins, initial_r_wins, steps):
    plt.figure(figsize=(10, 8))
    plt.hist(data_dict['d_wins'], bins=20, color='lightblue', edgecolor='black', alpha=0.7, align='left', label='Democratic')
    plt.hist(data_dict['r_wins'], bins=20, color='salmon', edgecolor='black', alpha=0.7, align='left', label='Republican')
    plt.title(f"Districts Won by Party Histogram - {election}")
    plt.xlabel("Number of Districts Won")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(x=initial_d_wins[0], color='b', linestyle='--', label='Initial Plan - Democratic')
    plt.axvline(x=initial_r_wins[0], color='r', linestyle='--', label='Initial Plan - Republican')
    plt.legend()

    plt.savefig(generate_output_fname(election, steps, "district_wins_histogram"))
    plt.close()

def graph_cut_edges(election, data_dict, initial_cut_edges, steps):
    plt.figure(figsize=(10, 8))
    plt.hist(data_dict['cut_edges'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
    plt.title(f"Cut Edges Histogram - {election}")
    plt.xlabel("Number of Cut Edges")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.axvline(x=initial_cut_edges[0], color='r', linestyle='--', label='Initial Plan')
    plt.legend()

    plt.savefig(generate_output_fname(election, steps, "cut_edges_histogram"))
    plt.close()

def graph_num_majority_black(election, data_dict, initial_num_majority_black, steps):

    plt.figure(figsize=(10, 8))
    num_majority_black_values = data_dict['num_majority_black']
    min_value = min(num_majority_black_values)
    max_value = max(num_majority_black_values)
    bin_range = (min_value - 0.5, max_value + 0.5)  # Adjust the range to include all values
    num_bins = max_value - min_value + 1  # Set the number of bins to the range of values

    plt.hist(num_majority_black_values, bins=num_bins, range=bin_range, color='skyblue', edgecolor='black', alpha=0.7, rwidth=0.8, align='mid')
    plt.title(f"Majority Black Districts - {election}")
    plt.xlabel("Number of Districts")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(min_value, max_value + 1))  # Set x-ticks to integer values

    plt.axvline(x=initial_num_majority_black, color='r', linestyle='--', label='Initial Plan')
    plt.legend()

    plt.savefig(generate_output_fname(election, steps, "num_majority_black"))
    plt.close()

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

    initial_plan_dict = {}

    for election in elections:
        initial_plan_stats = get_plan_stats(DATA_DICT_FORMAT.copy(), initial_partition, election.name)
        initial_plan_dict[election.name] = initial_plan_stats



    for steps in NUM_STEPS:

        # Execute random walk
        random_walk = execute_random_walk(initial_partition, steps, NUM_GA_DISTS, ga_graph)

        # Gather data for each election
        election_data_dict = {}

        for election in elections:
            election_data_dict[election.name] = DATA_DICT_FORMAT.copy()

        for election, data_dict in election_data_dict.items():
        # for election in elections:
            for partition in random_walk:
                # data_dict = get_plan_stats(data_dict, partition, election)
                #election_data_dict[election] = data_dict
                data_dict['eg'].append(efficiency_gap(partition[election]))
                data_dict['mm'].append(mean_median(partition[election]))
                data_dict['d_wins'].append(partition[election].wins("Democratic"))
                data_dict['r_wins'].append(partition[election].wins("Republican"))
                data_dict['cut_edges'].append(len(partition["cut_edges"]))
                data_dict["dem_vote_share_pd"].append(sorted(partition[election].percents("Democratic"))) # --> Gives a list with the percent vote share in each district
                data_dict["num_majority_black"].append(partition["majority_bvap_districts"])


        # Plot histograms
        for election, data_dict in election_data_dict.items():

            graph_marginal_box_plots(data_dict["dem_vote_share_pd"], election, steps)

            graph_eg(election, data_dict, initial_plan_dict[election]["eg"], steps)

            graph_mm(election, data_dict, initial_plan_dict[election]["mm"], steps)

            graph_d_vs_r_wins(election, data_dict, initial_plan_dict[election]["d_wins"], initial_plan_dict[election]["r_wins"], steps)

            graph_cut_edges(election, data_dict, initial_plan_dict[election]["cut_edges"], steps)

            graph_num_majority_black(election, data_dict, initial_plan_dict[election]["num_majority_black"][0], steps)

        # Measure execution time
        end_time = time.time()
    print("The time of execution of the program is:", (end_time - start_time) / 60, "mins")

if __name__ == "__main__":
    main()