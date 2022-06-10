from gurobipy import *
import numpy as np
import pandas as pd
import pickle
import csv
import time
import math
import os

from visualize import *
from utils import *


def find_optimal_facility_locations(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, N, M, P, lonlat, H, S, Dt, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, MANUAL_LOC, ROD_MIN, TIMEOUT, THREADS, MAPBOX_TOKEN, SHOW_GUROBI_OUTPUT, **kwargs):

	for n in N:
		# If we differentiate between product categories:
		if P_CAT == True:
			for m in M:
				kwargs["P"] = P
				kwargs["m"] = m
				model, calc_time = build_and_run_model(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, n, lonlat, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, MANUAL_LOC, ROD_MIN, TIMEOUT, THREADS, MAPBOX_TOKEN, SHOW_GUROBI_OUTPUT, **kwargs)
				process_model_outputs_pcat(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, model, n, lonlat, H, S, Dt, U, ROD_TIME, COUNTRY, src, prio, calc_time, **kwargs)
		# If we treat all products the same:
		else:
			model, calc_time = build_and_run_model(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, n, lonlat, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, MANUAL_LOC, ROD_MIN, TIMEOUT, THREADS, MAPBOX_TOKEN, SHOW_GUROBI_OUTPUT)
			process_model_outputs(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, model, n, lonlat, H, S, Dt, U, ROD_TIME, COUNTRY, src, prio, calc_time)

def build_and_run_model(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, n, lonlat, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, MANUAL_LOC, ROD_MIN, TIMEOUT, THREADS, MAPBOX_TOKEN, SHOW_GUROBI_OUTPUT, **kwargs):

	print_iteration_in_terminal(OBJ, ORDER_DATA, TRANSPORTS, P_CAT, src, ROD_TIME, prio, n, **kwargs)

	start = time.perf_counter()

	if P_CAT == True:
		P = kwargs["P"]
		m = kwargs["m"]


	################
	## PARAMETERS ##
	################

	model = Model(name="model")
	if SHOW_GUROBI_OUTPUT == False:
		model.Params.LogToConsole = 0
	model.Params.LogFile = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "log", **kwargs)+".csv"
	model.setParam('Threads', THREADS)
	model.setParam('TimeLimit', TIMEOUT)

	# Only load these matrices if routed transports are included.
	if "routed" in TRANSPORTS:
		Hc = kwargs["Hc"]				# Mapping from each hospital to the largest hospital of its cluster.
		Cr_km = kwargs["Cr_km"]			# Cost per km in euros from i∈S to j∈H and back to i.
		Cr_stop = kwargs["Cr_stop"]		# Cost for stopping at any destination in a cluster when starting from i∈S.
		if ORDER_DATA == True:
			Or = kwargs["Or"]			# Total number of routed transports to each hospital.
			Orc = kwargs["Orc"]			# Total number of routed transports to each hospital that is a cluster center.

	# Only load these matrices if direct transports are included.
	if "direct" in TRANSPORTS:
		Cd = kwargs["Cd"]				# Transportation cost in euros for a direct transport from a DC to a hospital and back.
		if ORDER_DATA == True:
			OPd = kwargs["OPd"]			# Total number of prodcts received by direct transport for each hospital.
			Od = kwargs["Od"]			# Total number of direct transports to each hospital.
			

	###############
	## VARIABLES ##
	###############

	# x: For each candidate location i∈S: x[i] = 1 if a DC is located there, x[i] = 0 otherwise.
	# y: For each pair of candidate location i∈S and hospital j∈H: y[i,j] = 1 if hospital j is allocated to DC i, y[i,j] = 0 otherwise.
	# z: For each candidate location i∈S: z[i] = 1 if slow-moving products will be stored at this location, z[i] = 0 otherwise.

	x = model.addVars(len(S), name='x', vtype=GRB.BINARY, lb=0, ub=1)
	if P_CAT == True:
		y = model.addVars(len(S), len(H), len(P), name='y', vtype=GRB.BINARY, lb=0, ub=1)
		z = model.addVars(len(S), name='z', vtype=GRB.BINARY, lb=0, ub=1)
	else:
		y = model.addVars(len(S), len(H), name='y', vtype=GRB.BINARY, lb=0, ub=1)

	model.update()
	print("# variables: " + str(len(model.getVars())))


	#################
	## CONSTRAINTS ##
	#################

	# A maximum of n DCs is allowed to be opened in total.
	model.addConstr(quicksum(x[i] for i in S.keys()) <= n)

	# If some of the candidate locations have been manually selected in the settings (main.py), force the model to choose these locations.
	if len(MANUAL_LOC) > 0:	
		model.addConstrs(x[i] == 1 for i in [i for i in S.keys() if S[i] in MANUAL_LOC])

	# A maximum of m DCs is allowed to store slow-moving products.
	if P_CAT == True:
		model.addConstr(quicksum(z[i] for i in S.keys()) <= m)
		model.addConstrs(x[i] >= z[i] for i in S.keys())

	# If i∈S is not in use as a DC, do not allow any hospitals to be allocated to this location.
	# Make sure that each hospital is allocated to exactly one DC.
	if P_CAT == True:
		model.addConstrs(quicksum(y[i,j,0] for j in H.keys()) <= x[i] * len(H) for i in S.keys())
		model.addConstrs(quicksum(y[i,j,1] for j in H.keys()) <= z[i] * len(H) for i in S.keys())
		model.addConstrs(quicksum(y[i,j,p] for i in S.keys()) == 1 for j in H.keys() for p in P.keys())
	else:
		model.addConstrs(quicksum(y[i,j] for j in H.keys()) <= x[i] * len(H) for i in S.keys())
		model.addConstrs(quicksum(y[i,j] for i in S.keys()) == 1 for j in H.keys())

	# Do not allow the reliability of delivery to be below the value for ROD_MIN given in the settings (main.py).
	# Note that this constraint is only active if maximizing reliability is not an objective function.
	if ("time" not in OBJ) and (ROD_MIN > 0):
		if P_CAT == True:
			if (ORDER_DATA == True) and ("direct" in TRANSPORTS):
				model.addConstr(quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) / sum(OPd[j,p] for j in H.keys() for p in P.keys()) >= ROD_MIN)
			else:
				model.addConstr(quicksum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) / (len(P)*len(H)) >= ROD_MIN)
		else:
			if (ORDER_DATA == True) and ("direct" in TRANSPORTS):
				model.addConstr(quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) / sum(OPd[j] for j in H.keys()) >= ROD_MIN)
			else:
				model.addConstr(quicksum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) / len(H) >= ROD_MIN)

	
	################
	## OBJECTIVES ##
	################

	if P_CAT == False:
		if ORDER_DATA == False:
			if "direct" in TRANSPORTS:
				if "routed" in TRANSPORTS:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, for both routed and direct transports.
							model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j]) + (Cr_km[i,j] * y[i,j]) + (Cr_stop[i] * y[i,Hc[j]]) for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, for both routed and direct transports.
							model.setObjective(expr = quicksum((Cd[i,j] * y[i,j]) + (Cr_km[i,j] * y[i,j]) + (Cr_stop[i] * y[i,Hc[j]]) for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, for both routed and direct transports.
						model.setObjective(expr = -1 * quicksum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
				else:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, only for direct transports.
							model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j] for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, only for direct transports.
							model.setObjective(expr = quicksum(Cd[i,j] * y[i,j] for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, only for direct transports.
						model.setObjective(expr = -1 * quicksum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
			else:								# Minimize transportation cost, only for routed transports.
				model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j]) + (Cr_stop[i] * y[i,Hc[j]]) for i in S.keys() for j in H.keys()))

		else:
			if "direct" in TRANSPORTS:
				if "routed" in TRANSPORTS:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, for both routed and direct transports, using hospital ordering data.
							model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j] * Od[j]) + (Cr_km[i,j] * y[i,j] * Orc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, for both routed and direct transports, using hospital ordering data.
							model.setObjective(expr = quicksum((Cd[i,j] * y[i,j] * Od[j]) + (Cr_km[i,j] * y[i,j] * Orc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, for both routed and direct transports, using hospital ordering data.
						model.setObjective(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
				else:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, only for direct transports, using hospital ordering data.
							model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j] * Od[j] for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, only for direct transports, using hospital ordering data.
							model.setObjective(expr = quicksum(Cd[i,j] * y[i,j] * Od[j] for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, only for direct transports, using hospital ordering data.
						model.setObjective(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
			else:								# Minimize transportation cost, only for routed transports, using hospital ordering data.
				model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j] * Orc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()))

	else:
		if ORDER_DATA == False:
			if "direct" in TRANSPORTS:
				if "routed" in TRANSPORTS:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, for both routed and direct transports, differentiating between product types.	
							model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j,p]) + (Cr_km[i,j] * y[i,j,p]) + (Cr_stop[i] * y[i,Hc[j],p]) for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, for both routed and direct transports, differentiating between product types.
							model.setObjective(expr = quicksum((Cd[i,j] * y[i,j,p]) + (Cr_km[i,j] * y[i,j,p]) + (Cr_stop[i] * y[i,Hc[j],p]) for p in P.keys() for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, for both routed and direct transports, differentiating between product types.
						model.setObjective(expr = -1 * quicksum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
				else:
					if "cost" in OBJ:	
						if "time" in OBJ:		# Both minimize cost and maximize reliability, only for direct transports, differentiating between product types.
							model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j,p] for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, only for direct transports, differentiating between product types.
							model.setObjective(expr = quicksum(Cd[i,j] * y[i,j,p] for p in P.keys() for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, only for direct transports, differentiating between product types.
						model.setObjective(expr = -1 * quicksum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
			else:								# Minimize transportation cost, only for routed transports, differentiating between product types.
				model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j,p]) + (Cr_stop[i] * y[i,Hc[j],p]) for p in P.keys() for i in S.keys() for j in H.keys()))
		
		else:
			if "direct" in TRANSPORTS:
				if "routed" in TRANSPORTS:
					if "cost" in OBJ:
						if "time" in OBJ:		# Both minimize cost and maximize reliability, for both routed and direct transports, using hospital ordering data, and differentiating between product types.
							model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j,p] * Od[j,p]) + (Cr_km[i,j] * y[i,j,p] * Orc[j,p]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, for both routed and direct transports, using hospital ordering data, and differentiating between product types.
							model.setObjective(expr = quicksum((Cd[i,j] * y[i,j,p] * Od[j,p]) + (Cr_km[i,j] * y[i,j,p] * Orc[j,p]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, for both routed and direct transports, using hospital ordering data, and differentiating between product types.
						model.setObjective(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
				else:		
					if "cost" in OBJ:			
						if "time" in OBJ:		# Both minimize cost and maximize reliability, only for direct transports, using hospital ordering data, and differentiating between product types.
							model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j,p] * Od[j,p] for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
							model.setObjectiveN(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
						else:					# Minimize transportation cost, only for direct transports, using hospital ordering data, and differentiating between product types.
							model.setObjective(expr = quicksum(Cd[i,j] * y[i,j,p] * Od[j,p] for p in P.keys() for i in S.keys() for j in H.keys()))
					else:						# Maximize reliability of delivery, only for direct transports, using hospital ordering data, and differentiating between product types.
						model.setObjective(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
			else:								# Minimize transportation cost, only for routed transports, using hospital ordering data, and differentiating between product types.
				model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j,p] * Orc[j,p]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()))

	# Minimize the objective functions.
	# Note that the reliability is multiplied by -1.
	model.ModelSense = GRB.MINIMIZE

	stop = time.perf_counter()
	calc_time = stop - start
	print(f"\nmodel initialization: {calc_time:0.4f} seconds")


	##############
	## OPTIMIZE ##
	##############

	model._best = None
	model._bound = None
	model._data = []
	model._start = time.time()
	start = time.perf_counter()
	model.optimize(callback=mycallback)
	stop = time.perf_counter()
	calc_time = stop - start
	print(f"\noptimize: {calc_time:0.4f} seconds")

	return model, calc_time


def process_model_outputs(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, model, n, lonlat, H, S, Dt, U, ROD_TIME, COUNTRY, src, prio, calc_time, **kwargs):

	print(status_code_to_message(model.status))

	# If an optimal solution was found.
	if model.status == 2:

		# Load parameters again.
		if "routed" in TRANSPORTS:
			Hc = kwargs["Hc"]
			Cr_km = kwargs["Cr_km"]
			Cr_stop = kwargs["Cr_stop"]
			if ORDER_DATA == True:
				Or = kwargs["Or"]
				Orc = kwargs["Orc"]
		if "direct" in TRANSPORTS:
			Cd = kwargs["Cd"]
			if ORDER_DATA == True:
				OPd = kwargs["OPd"]
				Od = kwargs["Od"]

		# Get the values of the model variable as found for the optimal solution.
		x = np.zeros([len(S)])
		y = np.zeros([len(S), len(H)])
		for var in model.getVars():
			name = re.split(r'\W+', var.varName)[0]
			if name == "x":
				index0 = int(re.split(r'\W+', var.varName)[1])
				x[index0] = var.X
			if name == "y":
				index0 = int(re.split(r'\W+', var.varName)[1])
				index1 = int(re.split(r'\W+', var.varName)[2])
				y[index0, index1] = var.X

		# Write the values found to local files.
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "x")+".pickle",'wb') as f:
			pickle.dump(x, f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y")+".pickle",'wb') as f:
			pickle.dump(y, f)
		nvars = len(x) + (len(y) * len(y[0]))

		# Create a dataframe to store objective values for all DC and hospital locations separately.
		obj_values_split = pd.DataFrame(index = pd.Series(name="name"), columns = ["type", "allocation", "transp. time", "r.o.d.", "avg cost per ride", "avg cost direct", "avg cost routed", "# direct transports", "# routed transports", "total cost direct", "total cost routed"])

		if ORDER_DATA == True:

			# Store objective values of all candidate locations that were chosen open a DC.
			for i in [i for i in S.keys() if x[i] == 1]:
				
				# Reliability of delivery for transports departing from location i∈S.
				rod_i = round(sum(OPd[j] * y[i,j] * U[i,j] for j in H.keys()) * 100 / max(sum(OPd[j] * y[i,j] for j in H.keys()),0.0001), 2)
				obj_values_split.loc[S[i],["type", "allocation", "transp. time", "r.o.d."]] = ["S", "-", "-", rod_i]
				
				# Transportation cost, both in total and on average, for all direct transports departing from location i∈S.
				if "direct" in TRANSPORTS:
					orders_direct = sum(y[i,j]*Od[j] for j in H.keys())
					cost_i_direct = round(sum(Cd[i,j]*y[i,j]*Od[j] for j in H.keys()), 2)
					avg_cost_i_direct = round(cost_i_direct / max(orders_direct,0.0001), 2)
					obj_values_split.loc[S[i],["avg cost direct", "# direct transports", "total cost direct"]] = [avg_cost_i_direct, orders_direct, cost_i_direct]
					if "routed" not in TRANSPORTS:
						obj_values_split.loc[S[i],"avg cost per ride"] = avg_cost_i_direct
				
				# Transportation cost, both in total and on average, for all routed transports departing from location i∈S.
				if "routed" in TRANSPORTS:
					orders_routed = sum(y[i,j]*Orc[j] for j in H.keys())
					cost_i_routed = round(sum((Cr_km[i,j]*y[i,j]*Orc[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j]) for j in H.keys()), 2)
					avg_cost_i_routed = round(cost_i_routed / max(orders_routed,0.0001), 2)
					obj_values_split.loc[S[i],["avg cost routed", "# routed transports", "total cost routed"]] = [avg_cost_i_routed, orders_routed, cost_i_routed]
					if "direct" not in TRANSPORTS:
						obj_values_split.loc[S[i],"avg cost per ride"] = avg_cost_i_routed
					else:
						obj_values_split.loc[S[i],"avg cost per ride"] = round((cost_i_direct + cost_i_routed) / max(orders_direct + orders_routed, 0.0001),2)

			# Store objective values of all hospital locations.
			for j in H.keys():
				# Find to which DC location the hospital is allocated, and how much time this transport is estimated to take.
				allocation = ",".join([S[i] for i in S.keys() if y[i,j] == 1])
				transp_time = ",".join([str(math.floor(Dt.loc[S[i],H[j]] / np.timedelta64(1,'h'))).zfill(2)+":"+str(round((Dt.loc[S[i],H[j]]-np.timedelta64(math.floor(Dt.loc[S[i],H[j]] / np.timedelta64(1,'h')),'h')) / np.timedelta64(1,'m'))).zfill(2) for i in S.keys() if y[i,j] == 1])
				
				# Reliability of delivery for deliveries to hospital j∈H.
				rod_j = round(sum(OPd[j] * y[i,j] * U[i,j] for i in S.keys()) * 100 / max(OPd[j],0.0001), 2)
				obj_values_split.loc[H[j],["type", "allocation", "transp. time", "r.o.d."]] = ["H", allocation, transp_time, rod_j]
				
				# Transportation cost, both in total and on average, for all direct deliveries to hospital j∈H.
				if "direct" in TRANSPORTS:
					cost_j_direct = round(sum(Cd[i,j]*y[i,j]*Od[j] for i in S.keys()), 2)
					avg_cost_j_direct = round(cost_j_direct / max(Od[j], 0.0001), 2)
					obj_values_split.loc[H[j],["avg cost direct", "# direct transports", "total cost direct"]] = [avg_cost_j_direct, Od[j], cost_j_direct]
					if "routed" not in TRANSPORTS:
						obj_values_split.loc[H[j],"avg cost per ride"] = avg_cost_j_direct
				
				# Transportation cost, both in total and on average, for all routed deliveries to hospital j∈H.
				if "routed" in TRANSPORTS:
					cost_j_routed = round(sum((Cr_km[i,Hc[j]] + Cr_stop[i]) * y[i,Hc[j]] * Or[j] for i in S.keys()), 2)
					avg_cost_j_routed = round(cost_j_routed / max(Or[j], 0.0001), 2)
					obj_values_split.loc[H[j],["avg cost routed", "# routed transports", "total cost routed"]] = [avg_cost_j_routed, Or[j], cost_j_routed]
					if "direct" not in TRANSPORTS:
						obj_values_split.loc[H[j],"avg cost per ride"] = avg_cost_j_routed
					else:
						obj_values_split.loc[H[j],"avg cost per ride"] = round((cost_j_direct + cost_j_routed) / max((Od[j]+Or[j]), 0.0001),2)

			# Reliability of delivery, to be exported by the write_output function.
			rod = round(sum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) * 100 / sum(OPd[j] for j in H.keys()), 2)
			
			# Transportation cost, both in total and on average per hospital, to be exported by the write_output function.
			if "routed" in TRANSPORTS:
				if "direct" in TRANSPORTS:
					cost = sum((Cd[i,j]*y[i,j]*Od[j]) + (Cr_km[i,j]*y[i,j]*Orc[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j]) for i in S.keys() for j in H.keys())
					avg_cost = cost / sum((Od[j]+Orc[j]) for j in H.keys())
				else:
					cost = sum((Cr_km[i,j]*y[i,j]*Orc[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j]) for i in S.keys() for j in H.keys())
					avg_cost = cost / sum(Orc[j] for j in H.keys())
			else:
				cost = sum((Cd[i,j]*y[i,j]*Od[j]) for i in S.keys() for j in H.keys())
				avg_cost = cost / sum(Od[j] for j in H.keys())
		else:
			# Reliability of delivery and transportation cost, both in total and on average per hospital, to be exported by the write_output function.
			rod = round(sum(y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) * 100 / len(H), 2)
			cost = sum(Cd[i,j] * y[i,j] for i in S.keys() for j in H.keys())
			avg_cost = cost / len(H.keys())

		# Create a local file that stores the dataframe with objective values for all locations separately.
		chosen_R = [S[i] for i in S.keys() if x[i] == 1 and S[i] in list(lonlat[lonlat["type"]=="R"].index)]
		if len(chosen_R) > 0:
			file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "outputsplit_"+"".join(chosen_R[0].split("_")))+".csv"
		else:
			file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "outputsplit")+".csv"
		obj_values_split.to_csv(file, sep=',', encoding = "1250", index=True)

		kwargs = dict()
		if ORDER_DATA == True:
			if "direct" in TRANSPORTS:
				kwargs["OPd"] = OPd
				kwargs["Od"] = Od
			if "routed" in TRANSPORTS:
				kwargs["Or"] = Or
				kwargs["Hc"] = Hc

		# Write all relevant output to a local file, and visualize the optimal solution on a map.
		write_output(HOME_DIR, SETTINGS, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, round(cost,2), round(avg_cost,2), rod, model._data, calc_time, nvars, status_code_to_message(model.status))	# model._best, model._bound
		# make_map(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, ROD_TIME, n, lonlat, S, H, U, COUNTRY, src, prio, **kwargs)

	# If an optimal solution has not been found due to some kind of error, or because the time limit was exceeded, write only the settings to the output file.
	elif model.status != 11:
		write_output(HOME_DIR, SETTINGS, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, np.zeros([len(S)]), "-", "-","-", model._data, calc_time, len(model.getVars()), status_code_to_message(model.status))


def process_model_outputs_pcat(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, model, n, lonlat, H, S, Dt, U, ROD_TIME, COUNTRY, src, prio, calc_time, **kwargs):

	print(status_code_to_message(model.status))

	P = kwargs["P"]
	m = kwargs["m"]

	# If an optimal solution was found.
	if model.status == 2:
	
		# Load parameters again.
		if "routed" in TRANSPORTS:
			Hc = kwargs["Hc"]
			Cr_km = kwargs["Cr_km"]
			Cr_stop = kwargs["Cr_stop"]
			if ORDER_DATA == True:
				Or = kwargs["Or"]
				Orc = kwargs["Orc"]
		if "direct" in TRANSPORTS:
			Cd = kwargs["Cd"]
			if ORDER_DATA == True:
				OPd = kwargs["OPd"]
				Od = kwargs["Od"]
		
		# Get the values of the model variable as found for the optimal solution.
		x = np.zeros([len(S)])
		y = np.zeros([len(S), len(H), len(P)])
		z = np.zeros([len(S)])

		# Get values of solved model variables.
		for var in model.getVars():
			name = re.split(r'\W+', var.varName)[0]
			if name == "x":
				index0 = int(re.split(r'\W+', var.varName)[1])
				x[index0] = var.X

			if name == "z":
				index0 = int(re.split(r'\W+', var.varName)[1])
				z[index0] = var.X

			if name == "y":
				index0 = int(re.split(r'\W+', var.varName)[1])
				index1 = int(re.split(r'\W+', var.varName)[2])
				index2 = int(re.split(r'\W+', var.varName)[3])
				y[index0, index1, index2] = var.X

		# Write the values found to local files.
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "x", m=m)+".pickle",'wb') as f:
			pickle.dump(x, f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "z", m=m)+".pickle",'wb') as f:
			pickle.dump(z, f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y", m=m)+".pickle",'wb') as f:
			pickle.dump(y, f)
		nvars = len(x) + len(z) + (len(y) * len(y[0]) * len(y[0][0]))

		# Create a dataframe to store objective values for all DC and hospital locations separately.
		obj_values_split = pd.DataFrame(index = pd.Series(name="name"), columns = ["type", "allocation", "transp. time", "r.o.d.", "avg cost per ride", "avg cost direct", "avg cost routed", "# direct transports", "# routed transports", "total cost direct", "total cost routed"])


		if ORDER_DATA == True:

			# Store objective values of all candidate locations that were chosen open a DC.
			for i in [i for i in S.keys() if x[i] == 1]:
				
				# Reliability of delivery for transports departing from location i∈S.
				rod_i = round(sum(OPd[j,p] * y[i,j,p] * U[i,j] for j in H.keys() for p in P.keys()) * 100 / max(sum(OPd[j,p] * y[i,j,p] for j in H.keys() for p in P.keys()),0.0001), 2)
				obj_values_split.loc[S[i],["type", "allocation", "transp. time", "r.o.d."]] = ["S", "-", "-", rod_i]

				# Transportation cost, both in total and on average, for all direct transports departing from location i∈S.
				if "direct" in TRANSPORTS:
					orders_direct = sum(y[i,j,p]*Od[j,p] for j in H.keys() for p in P.keys())
					cost_i_direct = round(sum(Cd[i,j]*y[i,j,p]*Od[j,p] for j in H.keys() for p in P.keys()), 2)
					avg_cost_i_direct = round(cost_i_direct / max(orders_direct,0.0001), 2)
					obj_values_split.loc[S[i],["avg cost direct", "# direct transports", "total cost direct"]] = [avg_cost_i_direct, orders_direct, cost_i_direct]
					if "routed" not in TRANSPORTS:
						obj_values_split.loc[S[i],"avg cost per ride"] = avg_cost_i_direct

				# Transportation cost, both in total and on average, for all routed transports departing from location i∈S.
				if "routed" in TRANSPORTS:
					orders_routed = sum(y[i,j,p]*Orc[j,p] for j in H.keys() for p in P.keys())
					cost_i_routed = round(sum((Cr_km[i,j]*y[i,j,p]*Orc[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or[j,p]) for j in H.keys() for p in P.keys()), 2)
					avg_cost_i_routed = round(cost_i_routed / max(orders_routed,0.0001), 2)
					avg_cost_i = round((cost_i_direct + cost_i_routed) / max(orders_direct + orders_routed, 0.0001),2)
					if "direct" not in TRANSPORTS:
						obj_values_split.loc[S[i],"avg cost per ride"] = avg_cost_i_routed
					else:
						obj_values_split.loc[S[i],"avg cost per ride"] = round((cost_i_direct + cost_i_routed) / max(orders_direct + orders_routed, 0.0001),2)
			
			# Store objective values of all hospital locations.
			for j in H.keys():
				# Find to which DC location the hospital is allocated, and how much time this transport is estimated to take.
				allocation = ",".join([S[i] for i in S.keys() if sum(y[i,j,p] for p in P.keys()) >= 1])
				transp_time = ",".join([str(math.floor(Dt.loc[S[i],H[j]] / np.timedelta64(1,'h'))).zfill(2)+":"+str(round((Dt.loc[S[i],H[j]]-np.timedelta64(math.floor(Dt.loc[S[i],H[j]] / np.timedelta64(1,'h')),'h')) / np.timedelta64(1,'m'))).zfill(2) for i in S.keys() if sum(y[i,j,p] for p in P.keys()) == 1])

				# Reliability of delivery for deliveries to hospital j∈H.
				rod_j = round(sum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for p in P.keys()) * 100 / max(sum(OPd[j,:]),0.0001), 2)
				obj_values_split.loc[H[j],["type", "allocation", "transp. time", "r.o.d."]] = ["H", allocation, transp_time, rod_j]

				# Transportation cost, both in total and on average, for all direct deliveries to hospital j∈H.
				if "direct" in TRANSPORTS:
					cost_j_direct = round(sum(Cd[i,j]*y[i,j,p]*Od[j,p] for i in S.keys() for p in P.keys()), 2)
					avg_cost_j_direct = round(cost_j_direct / max(sum(Od[j,:]), 0.0001), 2)
					obj_values_split.loc[H[j],["avg cost direct", "# direct transports", "total cost direct"]] = [avg_cost_j_direct, sum(Od[j,:]), cost_j_direct]
					if "routed" not in TRANSPORTS:
						obj_values_split.loc[H[j],"avg cost per ride"] = avg_cost_j_direct

				# Transportation cost, both in total and on average, for all routed deliveries to hospital j∈H.
				if "routed" in TRANSPORTS:
					cost_j_routed = round(sum((Cr_km[i,Hc[j]] + Cr_stop[i]) * y[i,Hc[j],p] * Or[j,p] for i in S.keys() for p in P.keys()), 2)
					avg_cost_j_routed = round(cost_j_routed / max(sum(Or[j,:]), 0.0001), 2)
					avg_cost_j = round((cost_j_direct + cost_j_routed) / max(sum(Od[j,:])+sum(Or[j,:]), 0.0001),2)
					if "direct" not in TRANSPORTS:
						obj_values_split.loc[H[j],"avg cost per ride"] = avg_cost_j_routed
					else:
						obj_values_split.loc[H[j],"avg cost per ride"] = round((cost_j_direct + cost_j_routed) / max(sum(Od[j,:])+sum(Or[j,:]), 0.0001),2)


			# Reliability of delivery, to be exported by the write_output function.
			rod = round(sum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) * 100 / sum(sum(OPd[j,:]) for j in H.keys()), 2)
			
			# Transportation cost, both in total and on average per hospital, to be exported by the write_output function.
			if "routed" in TRANSPORTS:
				if "direct" in TRANSPORTS:
					cost = sum((Cd[i,j]*y[i,j,p]*Od[j,p]) + (Cr_km[i,j]*y[i,j,p]*Orc[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
					avg_cost = cost / sum((Od[j,p]+Orc[j,p]) for j in H.keys() for p in P.keys())
				else:
					cost = sum((Cr_km[i,j]*y[i,j,p]*Orc[j,p]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
					avg_cost = cost / sum(Orc[j,p] for j in H.keys() for p in P.keys())
			else:
				cost = sum((Cd[i,j]*y[i,j,p]*Od[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
				avg_cost = cost / sum(Od[j,p] for j in H.keys() for p in P.keys())
		else:
			# Reliability of delivery and transportation cost, both in total and on average per hospital, to be exported by the write_output function.			
			rod = round(sum(y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) * 100 / (len(H)*len(P)), 2)
			cost = sum((Cd[i,j]*y[i,j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
			avg_cost = cost / (len(H)*len(P))
		
		# Create a local file that stores the dataframe with objective values for all locations separately.
		chosen_R = [S[i] for i in S.keys() if x[i] == 1 and S[i] in list(lonlat[lonlat["type"]=="R"].index)]
		if len(chosen_R) > 0:
			file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "outputsplit_"+"".join(chosen_R[0].split("_")), m=m)+".csv"
		else:
			file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "outputsplit", m=m)+".csv"
		obj_values_split.to_csv(file, sep=',', encoding = "1250", index=True)
				
		# Write all relevant output to a local file, and visualize the optimal solution on a map.
		write_output(HOME_DIR, SETTINGS, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, round(cost,2), round(avg_cost,2), rod, model._data, calc_time, nvars, status_code_to_message(model.status), m=m)	# model._best, model._bound
		# make_map(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, ROD_TIME, n, lonlat, S, H, U, COUNTRY, src, prio, **kwargs)

	# If an optimal solution has not been found due to some kind of error, or because the time limit was exceeded, write only the settings to the output file.
	elif model.status != 11:
		write_output(HOME_DIR, SETTINGS, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, np.zeros([len(S)]), "-", "-", "-", model._data, calc_time, len(model.getVars()), status_code_to_message(model.status), m=m)

# Callback that stores, during the the optimization process, the best and worst values that are still considered possible.
def mycallback(model, where):
	if where == GRB.Callback.MIP:
		cur_best = round(model.cbGet(GRB.Callback.MIP_OBJBST),3)
		cur_bound = round(model.cbGet(GRB.Callback.MIP_OBJBND),3)

		if (model._best != cur_best) or (model._bound != cur_bound):
			model._best = cur_best
			model._bound = cur_bound

			if model.ModelSense == 1:
				if model._best < 0:
					model._data.append(["time", time.time() - model._start, cur_best, cur_bound])
				else:
					model._data.append(["cost", time.time() - model._start, cur_best, cur_bound])
			elif model.ModelSense == -1:
				if model._best > 0:
					model._data.append(["time", time.time() - model._start, cur_best, cur_bound])
				else:
					model._data.append(["cost", time.time() - model._start, cur_best, cur_bound])