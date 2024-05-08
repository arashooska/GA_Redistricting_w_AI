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
#NUM_STEPS = [100]
NUM_STEPS = [10, 1000, 2000, 5000]


INITIAL_STATS = {
    "G18GOV" : {"eg": -0.13095959014815894, "mm": -0.09354198365629696, "d_wins": 5, "r_wins": 9, "cut_edges": 770, "dem_vote_share_pd": [0.3674232718487633, 0.7433496671820495, 0.880689150673209, 0.3742786929387745, 0.4318773811229532, 0.2888524364503992, 0.3286347888667593, 0.5505948279821782, 0.3666696902201483, 0.3079425034099255, 0.6933510189146589, 0.79942686787918, 0.34336716576907145, 0.4244864735830819]},
    "G18ATG" : {"eg": -0.11948233856414585, "mm": -0.08900580763630966, "d_wins": 5, "r_wins": 9, "cut_edges": 770, "dem_vote_share_pd": [0.3569436438094159, 0.724370815855996, 0.855133275288737, 0.37420680194262435, 0.4306069903101282, 0.29209322419632316, 0.33313639233348075, 0.5476505302737765, 0.36238982708739825, 0.3100618418222774, 0.6853021187734837, 0.7825960079374343, 0.3441267783940198, 0.4221815545863983]},
    "G18AGR" : {"eg": -0.08437353648256268, "mm": -0.08861756472901389, "d_wins": 5, "r_wins": 9, "cut_edges": 770, "dem_vote_share_pd": [0.33772350632359355, 0.7070301030466576, 0.8318842316354114, 0.3535547416729783, 0.41455432538937886, 0.2777743357006219, 0.31352866518708555, 0.5317588203698694, 0.34251630273019995, 0.2912026999928111, 0.6716535573510265, 0.7679919535686471, 0.3266118383823141, 0.40870933057225917]}
}

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

def get_plan_stats(partition, election):
    return_dict = {"eg": [], "mm": [], "d_wins": [], "r_wins": [], "cut_edges": [], "dem_vote_share_pd": [], "num_majority_black": []}
    
    # Calculate statistics on the initial partition
    return_dict["eg"].append(efficiency_gap(partition[election.name]))
    return_dict["mm"].append(mean_median(partition[election.name]))
    return_dict["d_wins"].append(partition[election.name].wins("Democratic"))
    return_dict["r_wins"].append(partition[election.name].wins("Republican"))
    return_dict["cut_edges"].append(len(partition["cut_edges"]))
    return_dict["dem_vote_share_pd"].append(partition[election.name].percents("Democratic"))
    return_dict["num_majority_black"].append(partition["majority_bvap_districts"])
    return return_dict



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

        # Add initial partition data as red dots
        initial_dem_shares = sorted(INITIAL_STATS[election]["dem_vote_share_pd"])
        ax.scatter([i+1 for i in range(NUM_GA_DISTS)], initial_dem_shares, color='r', marker='x', label='Initial Plan')

        plt.legend()

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

    initial_plan_dict = {}

    for election in elections:
        initial_plan_stats = get_plan_stats(initial_partition, election)
        initial_plan_dict[election.name] = initial_plan_stats



    for steps in NUM_STEPS:

        # Execute random walk
        random_walk = execute_random_walk(initial_partition, steps, NUM_GA_DISTS, ga_graph)

        # Gather data for each election
        election_data_dict = {}

        for election in elections:
            election_data_dict[election.name] = {"eg": [], "mm": [], "d_wins": [], "r_wins": [], "cut_edges": [], "dem_vote_share_pd": [], "num_majority_black": []}

        for election, data_dict in election_data_dict.items():
            for partition in random_walk:
                data_dict['eg'].append(efficiency_gap(partition[election]))
                data_dict['mm'].append(mean_median(partition[election]))
                data_dict['d_wins'].append(partition[election].wins("Democratic"))
                data_dict['r_wins'].append(partition[election].wins("Republican"))
                data_dict['cut_edges'].append(len(partition["cut_edges"]))
                data_dict["dem_vote_share_pd"].append(partition[election].percents("Democratic")) # --> Gives a list with the percent vote share in each district
                data_dict["num_majority_black"].append(partition["majority_bvap_districts"])


    

        #Separate out the data per district
        percent_dem_dict = prepare_percent_dem_data(election_data_dict)


        # Plot histograms
        for election, data_dict in election_data_dict.items():

            graph_marginal_box_plots(percent_dem_dict[election], election, steps)

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['eg'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Efficiency Gap Histogram - {election}")
            plt.xlabel("Efficiency Gap")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            initial_eg = initial_plan_dict[election]["eg"]
            plt.axvline(x=initial_eg[0], color='r', linestyle='--', label='Initial Plan')
            plt.legend()

            plt.savefig(generate_output_fname(election, steps, "eg_histogram"))
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['mm'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Mean Median Histogram - {election}")
            plt.xlabel("Mean Median")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            initial_mm = initial_plan_dict[election]["mm"]
            plt.axvline(x=initial_mm[0], color='r', linestyle='--', label='Initial Plan')
            plt.legend()

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

            initial_d_wins = initial_plan_dict[election]["d_wins"]
            initial_r_wins = initial_plan_dict[election]["r_wins"]
            plt.axvline(x=initial_d_wins[0], color='b', linestyle='--', label='Initial Plan - Democratic')
            plt.axvline(x=initial_r_wins[0], color='r', linestyle='--', label='Initial Plan - Republican')
            plt.legend()

            plt.savefig(generate_output_fname(election, steps, "district_wins_histogram"))
            plt.close()



            plt.figure(figsize=(10, 8))
            plt.hist(data_dict['cut_edges'], bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
            plt.title(f"Cut Edges Histogram - {election}")
            plt.xlabel("Number of Cut Edges")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            initial_cut_edges = initial_plan_dict[election]["cut_edges"]
            plt.axvline(x=initial_cut_edges[0], color='r', linestyle='--', label='Initial Plan')
            plt.legend()

            plt.savefig(generate_output_fname(election, steps, "cut_edges_histogram"))
            plt.close()

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

            initial_num_majority_black = initial_plan_dict[election]["num_majority_black"][0]
            plt.axvline(x=initial_num_majority_black, color='r', linestyle='--', label='Initial Plan')
            plt.legend()

            plt.savefig(generate_output_fname(election, steps, "num_majority_black"))
            plt.close()

            # Measure execution time
            end_time = time.time()
    print("The time of execution of the program is:", (end_time - start_time) / 60, "mins")

if __name__ == "__main__":
    main()