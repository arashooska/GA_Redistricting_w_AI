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
import time

# Load in graph from JSON
gdf = gpd.read_file("./GA/GA.shp")
gdf = gdf.dropna()
ga_graph = Graph.from_geodataframe(gdf)


#ga_graph = Graph.from_json("./GA/GA.geojson")


#Define constants
POPULATION_TOLERANCE = 0.02
NUM_IL_DISTS = 17
TOT_POP = sum([ga_graph.nodes()[v]['TOTPOP'] for v in ga_graph.nodes()])

#Determine number of democratic-won districts for a given partition
def dem_winning_dists(partition):

      num_wins = 0

      for p_dem in partition["PRES20"].percents("Democratic"):
            if p_dem > 0.5:
                  num_wins += 1

      return num_wins

#Determine number of majority latino districts for a given partition
def get_num_maj_latino_dists(partition):
         
        num_dists = 0
    
        for dist in partition.parts:
                if partition["dist_pop"][dist] > 0 and partition["dist_latino_pop"][dist]/partition["dist_pop"][dist] > 0.5:
                    num_dists += 1
    
        return num_dists

#Start timer
start_time = time.time()


"""
Elections from json file:
            "G20PRED": 753,
            "G20PRER": 62,
            "G20USSD": 684,
            "G20USSR": 51,
"""

# elections = [
#     Election("PRES20", {"Democratic": "G20PRED", "Republican": "G20PRER"}),
#     Election("USS20", {"Democratic": "G20USSD", "Republican": "G20USSR"}),
# ]

# Declare updaters to be used per partition in the random walk
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population"), 
               "cut_edges": cut_edges, 
               #"dem_winning_dists": dem_winning_dists,
               "dist_pop": Tally("TOTPOP", alias="dist_pop"),
               "dist_latino_pop": Tally("HISP", alias="dist_latino_pop")}

# election_updaters = {election.name: election for election in elections}
# my_updaters.update(election_updaters)

# Set up the initial partition object
initial_partition = Partition(
    ga_graph,
    assignment="CD",
    updaters=my_updaters
)

# Create the partial to iterate over in the random walk
random_walk_partial = partial(recom,
                              pop_col = "TOTPOP",
                              epsilon = POPULATION_TOLERANCE,
                              pop_target=TOT_POP/NUM_IL_DISTS,
                              node_repeats=2,
                            )

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, POPULATION_TOLERANCE, pop_key="dist_pop")


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
