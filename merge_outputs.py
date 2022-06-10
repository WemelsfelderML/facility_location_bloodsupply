import numpy as np
import pandas as pd
import pickle
import csv
import time
import datetime as dt
import os
import itertools
from itertools import permutations

from utils import *

# By executing this file, all results obtained by running the optimizer, for all different settings,
# will be merged into one file (outputs.csv), adding the corresponding settings as values in separate columns.

def main():

	HOME_DIR = ""				# Path to the directory where this file is stored (ending with a "/")
	COUNTRIES = ["NL", "FI"]
	SOURCES = ["E", "F"]

	df = pd.DataFrame(columns=["P_CAT", "ORDER_DATA", "TRANSPORTS", "OBJ", "COUNTRY", "source", "cost", "avg_cost", "dr", "n", "m", "A1_max_time (H)", "A1_max_time (UH)", "p_cost", "p_time", "obj_best_cost", "obj_bound_cost", "obj_gap_cost", "update_time_cost", "obj_best_time", "obj_bound_time", "obj_gap_time", "update_time_time", "calc_time", "nvars", "model_status", "open_locs"])

	for P_CAT in [True, False]:	# [True, False]
		for ORDER_DATA in [True, False]:	# [True, False]
			for TRANSPORTS in [["routed", "direct"], ["direct"], ["routed"]]:	# [["routed", "direct"], ["direct"], ["routed"]]
				for OBJ in [["cost", "time"], ["cost"], ["time"]]:	# [["cost", "time"], ["cost"], ["time"]]
					for src in SOURCES:
						for country in COUNTRIES:
							df = merge_outputs(HOME_DIR, df, P_CAT, ORDER_DATA, TRANSPORTS, OBJ, country, src)

def merge_outputs(HOME_DIR, df, P_CAT, ORDER_DATA, TRANSPORTS, OBJ, country, src):
	
	output_file = HOME_DIR + "results/" + settings(P_CAT, ORDER_DATA, TRANSPORTS, OBJ) + country + "_" + src + "/outputs.csv"

	if os.path.exists(output_file) == True:
		
		output = pd.read_csv(output_file, sep=",", encoding = "1250")
		output["P_CAT"] = [P_CAT for i in output.index]
		output["ORDER_DATA"] = [ORDER_DATA for i in output.index]
		output["TRANSPORTS"] = ["+".join(TRANSPORTS) for i in output.index]
		output["OBJ"] = ["+".join(OBJ) for i in output.index]
		output["COUNTRY"] = [country for i in output.index]
		output["source"] = [src for i in output.index]
		for obj in [obj for obj in ["time", "cost"] if obj not in OBJ]:
			output.loc[:, ["obj_best_"+obj, "obj_bound_"+obj, "obj_gap_"+obj, "update_time_"+obj]] = [["-"]*4 for i in output.index]
		output = output[df.columns]

		df = df.append(output)
		df.to_csv(HOME_DIR + "results/outputs.csv", sep=',', encoding = "1250", index=False)
		
	return df


if __name__ == "__main__":
	main()
