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
from gerrychain.metrics import mean_median, efficiency_gap

# ------------------------- Define Constants -------------------------
POPULATION_TOLERANCE = 0.2
NUM_GA_DISTS = 14
NUM_STEPS = [10, 1000, 2000, 5000]

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


def generate_output_fname(election, num_steps, metric):
    return f"./output/{num_steps}steps/{election}_{metric}.png"

# Separates out the data by district for each election.

# Returns a data structure set up like this

# {"ELECTION1" : [ [All steps for district 0], [All steps for district 1], ... [All steps for district n] ], 
#   "ELECTION2" : [ [All steps for district 0], [All steps for district 1], ... [All steps for district n] ], ... 
#       "ELECTIONn" : [ [All steps for district 0], [All steps for district 1], ... [All steps for district n] ]}
def prepare_percent_dem_data(election_data_dict):
    percent_dem_dict = {}
    for election, data_dict in election_data_dict.items():
        percent_dem_dict[election] = []
        for i in range(NUM_GA_DISTS):
            percent_dem_dict[election].append([])

    for election, data_dict in election_data_dict.items():
        for i, percent_dem in enumerate(data_dict["dem_vote_share_pd"]):
            for j, percent in enumerate(percent_dem):
                percent_dem_dict[election][j].append(percent)
    return percent_dem_dict

def graph_marginal_box_plots(vote_shares, election, steps):
        
        #Graph in order of lowest vote share to highest vote share
        medians = [np.median(district_votes) for district_votes in vote_shares]
        sorted_data = sorted(zip(range(NUM_GA_DISTS), medians, vote_shares), key=lambda x: x[1])
        x_axis = [x[0] for x in sorted_data]
        vote_shares_sorted = [x[2] for x in sorted_data]

        fig, ax = plt.subplots(figsize=(10, 6))
        box_plots = ax.boxplot(vote_shares_sorted, patch_artist=True)

        # Align x-axis ticks with box plots
        xtick_locations = range(1, NUM_GA_DISTS + 1)
        ax.set_xticks(xtick_locations)
        ax.set_xticklabels([x+1 for x in x_axis])

        # Add horizontal line at 0.5
        ax.axhline(y=0.5, linestyle='--', color='gray')

        ax.set_title(f"Democratic Vote Share Per District - {election}")
        ax.set_xlabel("District")
        ax.set_ylabel("Percent Vote Share for Democratic Party")
        plt.tight_layout()

        plt.savefig(generate_output_fname(election, steps, "district_vs_boxplot"))
        #plt.show()




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

    for steps in NUM_STEPS:

        # Execute random walk
        random_walk = execute_random_walk(initial_partition, steps, NUM_GA_DISTS, ga_graph)

        # Gather data for each election
        election_data_dict = {}

        for election in elections:
            election_data_dict[election.name] = {"eg": [], "mm": [], "d_wins": [], "r_wins": [], "cut_edges": [], "dem_vote_share_pd": []}

        for election, data_dict in election_data_dict.items():
            for partition in random_walk:
                data_dict['eg'].append(efficiency_gap(partition[election]))
                data_dict['mm'].append(mean_median(partition[election]))
                data_dict['d_wins'].append(partition[election].wins("Democratic"))
                data_dict['r_wins'].append(partition[election].wins("Republican"))
                data_dict['cut_edges'].append(len(partition["cut_edges"]))
                data_dict["dem_vote_share_pd"].append(partition[election].percents("Democratic")) # --> Gives a list with the percent vote share in each district


    

        #Separate out the data per district
        percent_dem_dict = prepare_percent_dem_data(election_data_dict)


        # Plot histograms
        for election, data_dict in election_data_dict.items():

            graph_marginal_box_plots(percent_dem_dict[election], steps, election)

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['eg'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Efficiency Gap Histogram - {election}")
            plt.xlabel("Efficiency Gap")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(generate_output_fname(election, steps, "eg_histogram"))
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['mm'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Mean Median Histogram - {election}")
            plt.xlabel("Mean Median")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(generate_output_fname(election, steps, "mm_histogram"))
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['d_wins'], bins=20, color='lightblue', edgecolor='black', alpha=0.7, align='left', label='Democratic')
            plt.hist(data_dict['r_wins'], bins=20, color='salmon', edgecolor='black', alpha=0.7, align='left', label='Republican')
            plt.title(f"Districts Won by Party Histogram - {election}")
            plt.xlabel("Number of Districts Won")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(generate_output_fname(election, steps, "district_wins_histogram"))
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['cut_edges'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Cut Edges Histogram - {election}")
            plt.xlabel("Number of Cut Edges")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(generate_output_fname(election, steps, "cut_edges_histogram"))
            plt.close()

            # Measure execution time
            end_time = time.time()
    print("The time of execution of the program is:", (end_time - start_time) / 60, "mins")

if __name__ == "__main__":
    main()