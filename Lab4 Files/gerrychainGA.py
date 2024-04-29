#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:26:42 2024

@author: eveomett

Author: Ellen Veomett

for AI for Redistricting

Lab 3, spring 2024
"""


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

# def mm(part, election):
    
#     """
#     TODO: Calculate the mean-median difference of the districting map corresponding to partition `part`
#     Return the value of the mean-median difference, from the perspective of the Democratic party.
#     """
 
#     vote_share_dem = part[election].percents("Democratic")
    
#     mean_vote_share = np.mean(vote_share_dem)
#     median_vote_share = statistics.median(vote_share_dem)

#     print(f"Mean vote share is {mean_vote_share} and median vote share is {median_vote_share}")

#     return median_vote_share - mean_vote_share


# def eg(part, election):
#     """
#     TODO: Calculate the efficiency gap of the districting map corresponding to partition `part`, 
#     using the wasted votes definition of the efficiency gap.
#     Return the value of the efficiency gap, from the perspective of the Democratic party.
#     """

#     wv_dem = 0
#     wv_rep = 0
#     all_votes = 0

#     for d in range(len(part[election].percents("Democratic"))):
#         total_votes = part[election].votes("Democratic")[d] + part[election].votes("Republican")[d]

#         all_votes += total_votes

#         rep_vs = part[election].percents("Republican")[d]
#         dem_vs = part[election].percents("Democratic")[d]

#         #If republicans won, wasted votes for democrats = all democrat votes
#         # Wasted republican votes is every vote over 50%
#         if rep_vs > dem_vs:
#             wv_dem += part[election].votes("Democratic")[d]
#             wv_rep += part[election].votes("Republican")[d] - total_votes / 2
        
#         #Else, the opposite
#         else:
#             wv_rep += part[election].votes("Republican")[d]
#             wv_dem += part[election].votes("Democratic")[d] - total_votes / 2

#     return (wv_rep - wv_dem) / all_votes


#Determine number of democratic-won districts for a given partition
# def dem_winning_dists(partition):

#       num_wins = 0

#       for p_dem in partition["PRES20"].percents("Democratic"):
#             if p_dem > 0.5:
#                   num_wins += 1

#       return num_wins

# #Determine number of majority latino districts for a given partition
# def get_num_maj_latino_dists(partition):
         
#         num_dists = 0
    
#         for dist in partition.parts:
#                 if partition["dist_pop"][dist] > 0 and partition["dist_latino_pop"][dist]/partition["dist_pop"][dist] > 0.5:
#                     num_dists += 1
    
#         return num_dists

# Load in graph from JSON
# gdf = gpd.read_file("./GA_clean.json")
# gdf = gdf.dropna()
# ga_graph = Graph.from_geodataframe(gdf)


ga_graph = Graph.from_json("./GA_clean.json")


#Define constants
POPULATION_TOLERANCE = 0.02
NUM_GA_DISTS = 14
TOT_POP = sum([ga_graph.nodes()[v]['TOTPOP'] for v in ga_graph.nodes()])


#Start timer
start_time = time.time()


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

#Set updaters so that we can make histograms for mean median and efficiency gap, dem/rep winning districts, and cut edges
#Also marginal box plots

# Declare updaters to be used per partition in the random walk
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population"), 
               "cut_edges": cut_edges, 
               "district Black": Tally("BVAP", alias="district Black")
            }

election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

# Set up the initial partition object
initial_partition = Partition(
    ga_graph,
    assignment="CD",
    updaters=my_updaters
)

# Create the partial to iterate over in the random walk
random_walk_partial = partial(recom,
                              pop_col = "VAP",
                              epsilon = POPULATION_TOLERANCE,
                              pop_target=TOT_POP/NUM_GA_DISTS,
                              node_repeats=2,
                            )

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, POPULATION_TOLERANCE, pop_key="population")


NUM_STEPS = 10

# Execute the random walk
random_walk = MarkovChain( proposal=random_walk_partial,
                            constraints=[pop_constraint],
                            accept=always_accept,
                            initial_state=initial_partition,
                            total_steps=NUM_STEPS)


#NOTE: Skipping below; we just want to show that we can succesfully run the chain.

# Gather relevant data for each partition in the random walk
cut_edge_list = []
dem_winning_districts = []
maj_latino_districts = []
# for partition in random_walk:
#     cut_edge_list.append(len(partition["cut_edges"]))
    #dem_winning_districts.append(partition["dem_winning_dists"])
    #maj_latino_districts.append(get_num_maj_latino_dists(partition))


# Create a histogram for the number of cut edges in the plan, the number of democratic districts, and the number of majority lation districts in the plan

print(f"Successfully ran chain for {NUM_STEPS} steps!")

# plt.figure()
# plt.hist(cut_edge_list, align='left')
# plt.title("Cut edges")
# plt.show()
# plt.savefig("IL_cutedges.png")

# plt.figure()
# plt.hist(dem_winning_districts, align='left')
# plt.title("Democratic winning districts")
# plt.savefig("IL_dem_winning_dists.png")

# plt.figure()
# plt.hist(maj_latino_districts, bins=range(10))
# plt.title("Majority Latino districts")
# plt.savefig("IL_maj_latino_dists.png")


end_time = time.time()
print("The time of execution of above program is :",
      (end_time-start_time)/60, "mins")
