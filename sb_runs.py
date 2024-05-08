import numpy as np
import pickle
from gerrychain import Graph, Partition, Election, updaters
from gerrychain.updaters import Tally, cut_edges
from gingleator import Gingleator

score_functs = {0: None, 1: Gingleator.reward_partial_dist, 
                2: Gingleator.reward_next_highest_close,
                3: Gingleator.penalize_maximum_over,
                4: Gingleator.penalize_avg_over}

BURST_LEN = 3
NUM_DISTRICTS = 14
ITERS = 50
POP_COL = "TOTPOP"
N_SAMPS = 10
SCORE_FUNCT = None
EPS = 0.045
MIN_POP_COL = "district Black"

print("Reading in Data/Graph", flush=True)

graph = Graph.from_json("./GA_clean.json")

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

my_updaters = {"population": updaters.Tally(POP_COL, alias="population"), 
        "VAP": Tally("VAP"),
        "cut_edges": cut_edges, 
        "district Black": Tally("BVAP", alias="district Black")
    }
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = Partition(graph, assignment="CD", updaters=my_updaters)

gingles = Gingleator(initial_partition, pop_col=POP_COL,
                     threshold=0.5, score_funct=SCORE_FUNCT, epsilon=EPS,
                     minority_perc_col="{}_perc".format(MIN_POP_COL))

gingles.init_minority_perc_col(MIN_POP_COL, "VAP", 
                               "{}_perc".format(MIN_POP_COL))

num_bursts = int(ITERS/BURST_LEN)

print("Starting Short Bursts Runs", flush=True)

for n in range(N_SAMPS):
    sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=BURST_LEN,
                                     maximize=True, verbose=False)
    print("\tFinished chain {}".format(n), flush=True)

    print("\tSaving results", flush=True)

    f_out = "./output/sb5000/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}.npy".format("GA",
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, SCORE_FUNCT, n)
    np.save(f_out, sb_obs[1])

    f_out_part = "./output/sb5000/{}_dists{}_{}opt_{:.1%}_{}_sbl{}_score{}_{}_max_part.p".format("GA",
                                                        NUM_DISTRICTS, MIN_POP_COL, EPS, 
                                                        ITERS, BURST_LEN, SCORE_FUNCT, n)

    max_stats = {"VAP": sb_obs[0][0]["VAP"],
                 "BVAP": sb_obs[0][0]["district Black"]}

    with open(f_out_part, "wb") as f_out:
        pickle.dump(max_stats, f_out)