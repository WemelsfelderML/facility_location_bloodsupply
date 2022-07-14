import numpy as np
import pandas as pd
import pickle
import csv
import time
import datetime as dt
import os
import itertools
from itertools import permutations

from optimize import *
from utils import *
from robustness_testing import *

def main():

	##############
	## SETTINGS ##
	##############

	COUNTRY = "NL"

	HOME_DIR = ""											# Path to the directory where this file is stored (ending with a "/")
	INPUT_DIR = HOME_DIR + "/data_"+COUNTRY+"/"		# Path were the data is stored.

	# Determine the set of locations that distribution centers can be placed.
	# E (current distr. centers), F (E + all hospitals), Q (F + greenfield points)
	# The optimizer is run separately for each element in this list.
	SOURCES = ["F"] 	# "E": current DCs, "F": currently existing facilities, "R": some grid

	# Number of distribution centers that is allowed to exist.
	# The optimizer is run separately for each element in this list.
	N = [4, 5, 6, 7]

	# Number of distribution centers that slow-moving products are placed.
	# The optimizer is run separately for each element in this list.
	# Note: only used if P_CAT = True
	M = [1, 2, 3]

	# Set time limit (hour,minutes) for delivery reliability.
	ROD_TIME = {							
		"UH": (1,0),		# time limit for Academic Hospitals
		"H": (1,0)			# time limit for other hospitals
	}

	# Choose whether the following additions will be included in the model.
	OBJ = ["cost"]					# "cost", "time", or both
	TRANSPORTS = ["direct", "routed"]		# "routed", "direct", or both
	ORDER_DATA = True						# Include hospital ordering data.
	P_CAT = False							# Differentiate between slow-moving and fast-moving products.

	# If "R" in SOURCES, a name should be chosen here, corresponding to equally lonlat, Dt and C_direct files.
	# Use "" if no special name is desired.
	SCENARIO_NAME = ""

	# Names of current DC locations that are desired to keep open ([] if none).
	MANUAL_LOC = []

	# Minimum desired reliability of delivery (fraction of 1), set to 0 if there is no minimum. 
	# Only used if "time" is not in OBJ.
	# How to find the current reliability of delivery (ROD): 
	# Run with settings SOURCES=["E"], N=[len(E)], ROD_TIME=current, TIME_OBJ=True, DIRECT=True, P_CAT=False.
	# Check the value in "rod" column of the outputs.csv file and divide by 100.
	if COUNTRY == "NL":
		if ORDER_DATA == True:
			ROD_MIN = 0.9805		# 0.9805
		else:
			ROD_MIN = 0.9636		# 0.9636
	elif COUNTRY == "FI":
		ROD_MIN = 1.0

	# Number of minutes after which the optimizer is terminated if it is still running.
	TIMEOUT = 60*60

	# Number of processor threads the optimizer should use.
	THREADS = 14

	# Insert a mapbox access token. This is needed to visualise the output on a map.
	# If you do not wish to visualize results on a map, comment the "make_map" function in optimize.py.
	MAPBOX_TOKEN = ""

	# Whether or not to show gurobi process output in the terminal.
	SHOW_GUROBI_OUTPUT = False

	# Whether to run the facility location model, or the robustness testing.
	# Note: the latter only works if results already exist for the chosen settings.
	GOAL = "locate"		# "locate" for finding locations, "robust" for testing robustness
	

	####################
	## CHECK SETTINGS ##
	####################

	if (("cost" in OBJ) or ("time" in OBJ)) == False:
		print("EXECUTION ABORTED\nPlease define the 'OBJ' parameter to contain at least one of the following objectives: 'cost', 'time'.")
		return
	if (("routed" in TRANSPORTS) or ("direct" in TRANSPORTS)) == False:
		print("EXECUTION ABORTED\nPlease define the 'TRANSPORTS' parameter to contain at least one of the following transportation types: 'routed', 'direct'.")
		return
	if ("direct" not in TRANSPORTS) and ("time" in OBJ):
		print("EXECUTION ABORTED\nInfeasible combination: 'time' in OBJ, but 'direct' not in TRANSPORTS\nTo use the time objective, please allow direct transports.")
		return
	if (GOAL == "robust") and (ORDER_DATA == False):
		print("EXECUTION ABORTED\nInfeasible combination: GOAL = 'robust' and ORDER_DATA = False\nTo test how robust your results are to changes in ordering data, please set ORDER_DATA = True.")
		return


	##########
	## DATA ##
	##########

	# If len(OBJ) > 1, create a list of all orders of these objectives.
	obj_combs = list(permutations(range(len(OBJ)), len(OBJ)))

	# Product categories: fast-movers and slow-movers
	P = {0:"fast", 1:"slow"}

	start = time.perf_counter()

	# Dataframe containing all locations
	# column   type  	description
	# ------ --------  ---------------
	#  name   object 	location name
	#  type   object 	"E", "H", "R"
	#  lat    float64	latitude
	#  lon    float64	longitude
	if len(SCENARIO_NAME) > 0:
		lonlat = pd.read_csv(INPUT_DIR+"F_"+SCENARIO_NAME+".csv", index_col="name", sep=",", encoding = "1250")
	else:
		lonlat = pd.read_csv(INPUT_DIR+"F.csv", index_col="name", sep=",", encoding = "1250")

	# Dictionary:
		# values: hospital names.
		# keys: their indices in the order and cost data matrices below.
	with open(INPUT_DIR+'H.pickle', 'rb') as handle:
		H = pickle.load(handle)

	hospitals = list(lonlat[lonlat["type"]=="H"].index)
	dcs = list(lonlat[lonlat["type"]=="E"].index)

	E = dict()
	F = dict()
	for j in range(len(hospitals)):
		F[j] = hospitals[j]
	for i in range(len(dcs)):
		E[i] = dcs[i]
		F[i+len(hospitals)] = dcs[i]

	if "R" in SOURCES:
		R = dict()	# grid locations + MANUAL_LOC
		scenario_locs = list(lonlat[lonlat["type"].isin(["R"])].index)
		for i in range(len(scenario_locs)):
			R[i] = scenario_locs[i]
		for i in [i for i in MANUAL_LOC if i not in R.values()]:
			R[len(R)] = i
	
	# Matrices of shape H×P, with all hospitals on the vertical axis and the product types (fast/slow moving) on the horizontal axis.
	if ORDER_DATA == True:
		if "direct" in TRANSPORTS:
			# Matrix cells contain the total number of products in direct transports .
			with open(INPUT_DIR+'OPd.pickle', 'rb') as handle:
				OPd = pickle.load(handle)
			# Matrix cells contain the total number of direct transports.
			with open(INPUT_DIR+'Od.pickle', 'rb') as handle:
				Od = pickle.load(handle)
			
			if P_CAT == False:
				OPd = sum(OPd[:,p] for p in P.keys())
				Od = sum(Od[:,p] for p in P.keys())
		
		if "routed" in TRANSPORTS:	
			# Matrix cells contain the total number of routed transports.
			with open(INPUT_DIR+'Or.pickle', 'rb') as handle:
				Or = pickle.load(handle)
			# Matrix cells contain the total number of routed transports to cluster centers.
			with open(INPUT_DIR+'Orc.pickle', 'rb') as handle:
				Orc = pickle.load(handle)

			if P_CAT == False:
				Or = sum(Or[:,p] for p in P.keys())
				Orc = sum(Orc[:,p] for p in P.keys())

	# Dictionary:
		# key: index of some hospital (corresponding to keys of H).
		# value: index of the largest hospital of the cluster of hospital [key].
	if "routed" in TRANSPORTS:
		with open(INPUT_DIR+'Hc.pickle', 'rb') as handle:
			Hc = pickle.load(handle)

	# Dt: R×R matrix of estimated transport times (format = H:MM:SS).
	# C: R×R matrix of estimated transport costs (in euros) from i∈S to j∈H and back to i.
		# C_direct: cost of one direct transport.
		# C_routed_km: cost per km of a routed transport.
		# C_routed_stop: cost per delivery stop of a routed transport.
	# Both indices and columns are location names, equal to lonlat column "name".
	if len(SCENARIO_NAME) > 0:
		Dt = pd.read_csv(INPUT_DIR + 'Dt_'+SCENARIO_NAME+'.csv', sep=',', encoding = "1250", index_col="name", low_memory=False)
		if "direct" in TRANSPORTS:
			C_direct = pd.read_csv(INPUT_DIR + 'Cd_'+SCENARIO_NAME+'.csv', encoding = "1250", sep=',', index_col="name")
		if "routed" in TRANSPORTS:
			C_routed_km = pd.read_csv(INPUT_DIR + 'Cr_km_'+SCENARIO_NAME+'.csv', encoding = "1250", sep=',', index_col="name")
			C_routed_stop = pd.read_csv(INPUT_DIR + 'Cr_stop_'+SCENARIO_NAME+'.csv', encoding = "1250", sep=',', index_col="name")
	else:
		Dt = pd.read_csv(INPUT_DIR + 'Dt.csv', sep=',', encoding = "1250", index_col="name", low_memory=False)
		if "direct" in TRANSPORTS:
			C_direct = pd.read_csv(INPUT_DIR + 'Cd.csv', sep=',', encoding = "1250", index_col="name")
		if "routed" in TRANSPORTS:
			C_routed_km = pd.read_csv(INPUT_DIR + 'Cr_km.csv', sep=',', encoding = "1250", index_col="name")
			C_routed_stop = pd.read_csv(INPUT_DIR + 'Cr_stop.csv', sep=',', encoding = "1250", index_col="name")

	for col in Dt.columns:
		Dt[col] = Dt[col].apply(lambda x: pd.to_timedelta(x))

	stop = time.perf_counter()
	print(f"\n\nData loading time: {stop - start:0.4f} seconds")


	##################
	# RUN OPTIMIZER ##
	##################

	# Convert the chosen settings to a path to store results.
	SETTINGS = settings(P_CAT, ORDER_DATA, TRANSPORTS, OBJ)

	# If a directory to store the results does not yet exist, make one.
	path = HOME_DIR+"results"
	if os.path.exists(path) == False:
		os.mkdir(path)

	# If a directory for results with the chosen settings does not yet exist, make one.
	path = HOME_DIR + "results/" + SETTINGS
	if os.path.exists(path) == False:
		os.mkdir(path)

	# Run the model separately for all sources.
	for src in SOURCES:
		if src == "E":
			S = E
		elif src == "F":
			S = F
		elif src == "R":
			S = R

		if "direct" in TRANSPORTS:
			# Cd_ij = transportation cost in euros from i∈S to j∈H and back to i
			Cd = np.zeros([len(S),len(H)])
			for i in S.keys():
				for j in H.keys():
					Cd[i,j] = C_direct.loc[S[i],H[j]]
		if "routed" in TRANSPORTS:
			# CR_km_ij = km cost in euros from i∈S to j∈H and back to i
			Cr_km = np.zeros([len(S),len(H)])
			for i in S.keys():
				for j in H.keys():
					Cr_km[i,j] = C_routed_km.loc[S[i],H[j]]
			# CR_stop_i = cost for stopping at any destination when starting from i∈S
			Cr_stop = np.zeros([len(S)])
			for i in S.keys():
				Cr_stop[i] = C_routed_stop.loc[S[i],"B_stop"]

		if len(SCENARIO_NAME) > 0:
			src += "_" + SCENARIO_NAME

		# If a directory for results with the chosen source does not yet exist, make one.
		path = HOME_DIR + "results/" + SETTINGS + COUNTRY + "_" + src
		if os.path.exists(path) == False:
			os.mkdir(path)

		# if lonlat does not contain a "subtype" column (containing whether a hospital is H or UH), create a dummy
		if "subtype" not in lonlat.columns:
			lonlat["subtype"] = [np.nan] * len(lonlat)

		# U_ij = 1 if distance i∈S to j∈H can be traveled within the time limit for emergency orders
		U = np.zeros([len(S), len(H)])
		for i in S.keys(): 
			for j in H.keys():
				if lonlat.loc[H[j],"subtype"] == "UH":
					if Dt.loc[S[i],H[j]] <= pd.to_timedelta("0 days "+str(ROD_TIME["UH"][0])+":"+str(ROD_TIME["UH"][1])+":00"):
						U[i,j] = 1
				else:
					if Dt.loc[S[i],H[j]] <= pd.to_timedelta("0 days "+str(ROD_TIME["H"][0])+":"+str(ROD_TIME["H"][1])+":00"):
						U[i,j] = 1

		# If a directory for results with the chosen ROD_TIME does not yet exist, make one.
		path = HOME_DIR + "results/" + SETTINGS + COUNTRY + "_" + src + "/H" + str(ROD_TIME["H"][0]).zfill(2) + "." + str(ROD_TIME["H"][1]).zfill(2) + "_UH" + str(ROD_TIME["UH"][0]).zfill(2) + "." + str(ROD_TIME["UH"][1]).zfill(2)
		if os.path.exists(path) == False:
			os.mkdir(path)

		# Run optimizer.
		for obj_comb in obj_combs:
			prio = {OBJ[i] : obj_comb[i] for i in range(len(OBJ))}
			start = time.perf_counter()	

			kwargs = dict()
			if "routed" in TRANSPORTS:
				kwargs["Hc"] = Hc
				kwargs["Cr_km"] = Cr_km
				kwargs["Cr_stop"] = Cr_stop
				if ORDER_DATA == True:
					kwargs["Or"] = Or
					kwargs["Orc"] = Orc
			if "direct" in TRANSPORTS:
				kwargs["Cd"] = Cd
				if ORDER_DATA == True:
					kwargs["OPd"] = OPd
					kwargs["Od"] = Od

			# Run facility location model.
			if GOAL == "locate":
				find_optimal_facility_locations(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, N, M, P, lonlat, H, S, Dt, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, MANUAL_LOC, ROD_MIN, TIMEOUT, THREADS, MAPBOX_TOKEN, SHOW_GUROBI_OUTPUT, **kwargs)
			
			# Run robustness testing (only possible if results for chosen settings already exist)
			elif GOAL == "robust":
				find_worst_case_objvals(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, N, M, P, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, TIMEOUT, THREADS, SHOW_GUROBI_OUTPUT, **kwargs)

			stop = time.perf_counter()
			print(f"\n\nOptimizer running time: {stop - start:0.4f} seconds")



if __name__ == "__main__":

	main()