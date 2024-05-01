# ------------------------- Import Libraries -------------------------
import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, updaters, constraints, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import time
from gerrychain.metrics import mean_median, efficiency_gap

# ------------------------- Define Constants -------------------------
POPULATION_TOLERANCE = 0.2
NUM_GA_DISTS = 14
NUM_STEPS = 1000

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
        elif data_type == "R_wins":
            gathered_data.append(partition[election.name].wins("Republican"))
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
    R_wins = []
    R_wins = []
    cut_edges = []
    for election in elections:
        # Efficency Gap
        EGs = gather_data(random_walk, election, "eg")
        # Mean Median
        MMs = gather_data(random_walk, election, "mm")
        # Dem-won Districts
        D_wins = gather_data(random_walk, election, "D_wins")
        # R_won Districts
        R_wins = gather_data(random_walk, election, "R_wins")

    # Cut Edges
    cut_edges = gather_data(random_walk, election, "cut_edges")

    # Print the gathered data
    # for election, EG, MM, D_win, R_win, cut_edge in zip(elections, EGs, MMs, D_wins, R_wins, cut_edges):
    #     print(f"\n{election.name}:")
    #     print(f"Efficiency Gap: {EG}")
    #     print(f"Mean Median: {MM}")
    #     print(f"Democratic-won Districts: {D_win}")
    #     print(f"Republican-won Districts: {R_win}")
    #     print(f"Cut Edges: {cut_edge}")

    # Write the output to a text file
    output_file = "./output/1000steps/outlier_analysis.txt"
    with open(output_file, "w") as f:
        for election, EG, MM, D_win, R_win, cut_edge in zip(elections, EGs, MMs, D_wins, R_wins, cut_edges):
            f.write(f"\n{election.name}:\n")
            f.write(f"Efficiency Gap: {EG}\n")
            f.write(f"Mean Median: {MM}\n")
            f.write(f"Democratic-won Districts: {D_win}\n")
            f.write(f"Republican-won Districts: {R_win}\n")
            f.write(f"Cut Edges Count: {cut_edge}\n")

    

    # Plot histograms
    plt.figure(figsize=(10, 8))
    plt.hist(EGs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
    plt.title("Efficiency Gap Histogram")
    plt.xlabel("Efficiency Gap")
    plt.ylabel("Probability Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./output/1000steps/eg.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(MMs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left')
    plt.title("Mean Median Histogram")
    plt.xlabel("Mean Median")
    plt.ylabel("Probability Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./output/1000steps/mm.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(D_wins, bins=20, color='lightblue', edgecolor='black', alpha=0.7, align='left', label='Republican', density=True)
    plt.hist(R_wins, bins=20, color='salmon', edgecolor='black', alpha=0.7, align='left', label='Democratic', density=True)
    plt.title("Districts Won by Party Histogram")
    plt.xlabel("Number of Districts Won")
    plt.ylabel("Probability Distribution")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./output/1000steps/DvsR_wins.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(cut_edges, bins=20, color='skyblue', edgecolor='black', alpha=0.7, align='left', density=True)
    plt.title("Cut Edges Histogram")
    plt.xlabel("Number of Cut Edges")
    plt.ylabel("Probability Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./output/1000steps/cut_edges.png")
    plt.close()

    # Measure execution time
    end_time = time.time()
    print("The time of execution of the program is:", (end_time - start_time) / 60, "mins")

if __name__ == "__main__":
    main()
