import numpy as np
import pickle
from gerrychain import Graph, Partition, Election, updaters
from gerrychain.updaters import Tally, cut_edges
from gingleator import Gingleator

score_functs = {0: None, 
                1: Gingleator.reward_partial_dist, 
                2: Gingleator.reward_next_highest_close,
                3: Gingleator.penalize_maximum_over,
                4: Gingleator.penalize_avg_over,
                5: Gingleator.num_opportunity_dists}

# BURST_LENS = [10, 25, 50, 100, 200]
BURST_LENS = [25]
NUM_DISTRICTS = 14
ITERS = 8000
POP_COL = "TOTPOP"
N_SAMPS = 10
SCORE_FUNCT = score_functs[5]
EPS = 0.045
THRESHOLDS = [0.5, 0.45, 0.4]
MIN_POP_COL = "BVAP"

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
        "BVAP": Tally("BVAP", alias="BVAP")
    }
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = Partition(graph, assignment="CD", updaters=my_updaters)

for burst_len in BURST_LENS:
    for threshold in THRESHOLDS:
        gingles = Gingleator(initial_partition, pop_col=POP_COL,
                            threshold=threshold, score_funct=SCORE_FUNCT, epsilon=EPS,
                            minority_perc_col="{}_perc".format(MIN_POP_COL))

        gingles.init_minority_perc_col(MIN_POP_COL, "VAP", 
                                    "{}_perc".format(MIN_POP_COL))

        num_bursts = int(ITERS/burst_len)

        print("Starting Short Bursts Runs", flush=True)

        for n in range(N_SAMPS):
            sb_obs = gingles.short_burst_run(num_bursts=num_bursts, num_steps=burst_len,
                                            maximize=True, verbose=False)
            print("\tFinished chain {}".format(n), flush=True)

            print("\tSaving results", flush=True)

            f_out = "./output/short-burst/sb-runs/{}_dists{}_{}_opt_{:.1%}_{}_sbl{}_score_{}_{}.npy".format("GA",
                                                                NUM_DISTRICTS, MIN_POP_COL, threshold, 
                                                                ITERS, burst_len, "num_opportunity_dists", n)
            np.save(f_out, sb_obs[1])

            f_out_part = "./output/short-burst/sb-runs/{}_dists{}_{}_opt_{:.1%}_{}_sbl{}_score_{}_{}_max_part.p".format("GA",
                                                                NUM_DISTRICTS, MIN_POP_COL, threshold, 
                                                                ITERS, burst_len, "num_opportunity_dists", n)

            max_stats = {"VAP": sb_obs[0][0]["VAP"],
                        "BVAP": sb_obs[0][0]["BVAP"]}

            with open(f_out_part, "wb") as f_out:
                pickle.dump(max_stats, f_out)