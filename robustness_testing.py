from gurobipy import *
import numpy as np
import pandas as pd
import pickle
import time

from utils import *

def find_worst_case_objvals(HOME_DIR, SETTINGS, ROD_TIME, COUNTRY, src, N, M, P, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, TIMEOUT, THREADS, SHOW_GUROBI_OUTPUT, **kwargs):

	# Set order deviation d as the percentage that hospital orders are allowed to deviate from their true value.
	for d in range(1,50,2):
		ORDER_DEVIATION = d/100

		for n in N:
		
			# If we differentiate between product categories:
			if P_CAT == True:
				for m in M:
					kwargs["P"] = P
					kwargs["m"] = m

					# Check if results already exist for the given scenario.
					if os.path.exists(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y", **kwargs)+".pickle") == True:
						# The two functions below can be found in "robustness_testing.py".
						model = build_and_run_worst_case(HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, TIMEOUT, THREADS, SHOW_GUROBI_OUTPUT, **kwargs)
						process_robustness_outputs_pcat(model, HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, **kwargs)

			# If we treat all products the same:
			else:
				# Check if results already exist for the given scenario.
				if os.path.exists(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y")+".pickle") == True:
					# The two functions below can be found in "robustness_testing.py".
					model = build_and_run_worst_case(HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, TIMEOUT, THREADS, SHOW_GUROBI_OUTPUT)
					process_robustness_outputs(model, HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME)

def build_and_run_worst_case(HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, TIMEOUT, THREADS, SHOW_GUROBI_OUTPUT, **kwargs):

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
	model.setParam('Threads', THREADS)
	model.setParam('TimeLimit', TIMEOUT)

	# Only load these matrices if routed transports are included.
	if "routed" in TRANSPORTS:
		Hc = kwargs["Hc"]				# Mapping from each hospital to the largest hospital of its cluster.
		Cr_km = kwargs["Cr_km"]			# Cost per km in euros from i∈S to j∈H and back to i.
		Cr_stop = kwargs["Cr_stop"]		# Cost for stopping at any destination in a cluster when starting from i∈S.
		if ORDER_DATA == True:
			Or_real = kwargs["Or"]		# Total number of routed transports to each hospital.
			Cc = np.zeros(len(H))		
			for j in Hc.values():		# For each hospital whether it is the center of a cluster (1) or not (0).
			    Cc[j] = 1				# Note that we use Cc instead of Orc since Orc is dependent on Or and the latter is now a variable.

	# Only load these matrices if direct transports are included.
	if "direct" in TRANSPORTS:
		Cd = kwargs["Cd"]				# Transportation cost in euros for a direct transport from a DC to a hospital and back.
		if ORDER_DATA == True:
			OPd_real = kwargs["OPd"]	# Total number of prodcts received by direct transport for each hospital.
			Od_real = kwargs["Od"]		# Total number of direct transports to each hospital.

	# Retrieve the allocations of the scenario to be tested.
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "y", **kwargs)+".pickle",'rb') as f:
		y = pickle.load(f)


	###############
	## VARIABLES ##
	###############

	if P_CAT == False:
		OPd = model.addVars(len(H), name='OPd', vtype=GRB.INTEGER, lb=0)
		Od = model.addVars(len(H), name='Od', vtype=GRB.INTEGER, lb=0)
		Or = model.addVars(len(H), name='Or', vtype=GRB.INTEGER, lb=0)
	else:
		OPd = model.addVars(len(H), len(P), name='OPd', vtype=GRB.INTEGER, lb=0)
		Od = model.addVars(len(H), len(P), name='Od', vtype=GRB.INTEGER, lb=0)
		Or = model.addVars(len(H), len(P), name='Or', vtype=GRB.INTEGER, lb=0)

	model.update()


	#################
	## CONSTRAINTS ##
	#################

	if P_CAT == False:
		# Make sure that the orders do not deviate more than the allowed percentage from their true value.
		if "direct" in TRANSPORTS:
			# The total number of products received by direct transport for each hospital
			# may not deviate more than the allowed percentage from its true value.
			model.addConstrs(OPd[j] >= (1-ORDER_DEVIATION)*OPd_real[j] for j in H.keys())
			model.addConstrs(OPd[j] <= (1+ORDER_DEVIATION)*OPd_real[j] for j in H.keys())
			
			# The total number of direct transports for each hospital may not
			# deviate more than the allowed percentage from its true value.	
			model.addConstrs(Od[j] >= (1-ORDER_DEVIATION)*Od_real[j] for j in H.keys())
			model.addConstrs(Od[j] <= (1+ORDER_DEVIATION)*Od_real[j] for j in H.keys())
			
			# The total number of direct transports, as well as the total number of products
			# received via direct transports should stay its true value.
			model.addConstr(quicksum(OPd_real[j] for j in H.keys()) == quicksum(OPd[j] for j in H.keys()))
			model.addConstr(quicksum(Od_real[j] for j in H.keys()) == quicksum(Od[j] for j in H.keys()))
		
		if "routed" in TRANSPORTS:
			# The total number of routed transports for each hospital may not
			# deviate more than the allowed percentage from its true value.	
			model.addConstrs(Or[j] >= (1-ORDER_DEVIATION)*Or_real[j] for j in H.keys())
			model.addConstrs(Or[j] <= (1+ORDER_DEVIATION)*Or_real[j] for j in H.keys())

			# The total number of routed transports should stay its true value.
			model.addConstr(quicksum(Or_real[j] for j in H.keys()) == quicksum(Or[j] for j in H.keys()))

	else:
		# Make sure that the orders do not deviate more than the allowed percentage from their true value.
		if "direct" in TRANSPORTS:
			# The total number of products received by direct transport for each hospital
			# may not deviate more than the allowed percentage from its true value.
			model.addConstrs(OPd[j,p] >= (1-ORDER_DEVIATION)*OPd_real[j,p] for j in H.keys() for p in P.keys())
			model.addConstrs(OPd[j,p] <= (1+ORDER_DEVIATION)*OPd_real[j,p] for j in H.keys() for p in P.keys())
			
			# The total number of direct transports for each hospital may not
			# deviate more than the allowed percentage from its true value.	
			model.addConstrs(Od[j,p] >= (1-ORDER_DEVIATION)*Od_real[j,p] for j in H.keys() for p in P.keys())
			model.addConstrs(Od[j,p] <= (1+ORDER_DEVIATION)*Od_real[j,p] for j in H.keys() for p in P.keys())
			
			# The total number of direct transports, as well as the total number of products
			# received via direct transports should stay its true value.
			model.addConstr(quicksum(OPd_real[j,p] for j in H.keys() for p in P.keys()) == quicksum(OPd[j,p] for j in H.keys() for p in P.keys()))
			model.addConstr(quicksum(Od_real[j,p] for j in H.keys() for p in P.keys()) == quicksum(Od[j,p] for j in H.keys() for p in P.keys()))
		
		if "routed" in TRANSPORTS:
			# The total number of routed transports for each hospital may not
			# deviate more than the allowed percentage from its true value.	
			model.addConstrs(Or[j,p] >= (1-ORDER_DEVIATION)*Or_real[j,p] for j in H.keys() for p in P.keys())
			model.addConstrs(Or[j,p] <= (1+ORDER_DEVIATION)*Or_real[j,p] for j in H.keys() for p in P.keys())

			# The total number of routed transports should stay its true value.
			model.addConstr(quicksum(Or_real[j,p] for j in H.keys() for p in P.keys()) == quicksum(Or[j,p] for j in H.keys() for p in P.keys()))


	################
	## OBJECTIVES ##
	################

	if P_CAT == False:
		if "direct" in TRANSPORTS:
			if "routed" in TRANSPORTS:
				if "cost" in OBJ:
					if "time" in OBJ:		# Both maximize cost and minimize reliability, for both routed and direct transports.
						model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j] * Od[j]) + (Cr_km[i,j] * y[i,j] * Or[j] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
						model.setObjectiveN(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
					else:					# Maximize transportation cost, for both routed and direct transports.
						model.setObjective(expr = quicksum((Cd[i,j] * y[i,j] * Od[j]) + (Cr_km[i,j] * y[i,j] * Or[j] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()))
				else:						# Minimize reliability of delivery, for both routed and direct transports.
					model.setObjective(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
			else:
				if "cost" in OBJ:
					if "time" in OBJ:		# Both maximize cost and minimize reliability, only for direct transports.
						model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j] * Od[j] for i in S.keys() for j in H.keys()) , index=0, priority=prio["cost"], name="cost")
						model.setObjectiveN(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()), index=1, priority=prio["time"], name="time")
					else:					# Maximize transportation cost, only for direct transports.
						model.setObjective(expr = quicksum(Cd[i,j] * y[i,j] * Od[j] for i in S.keys() for j in H.keys()))
				else:						# Minimize reliability of delivery, only for direct transports.
					model.setObjective(expr = -1 * quicksum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()))
		else:								# Maximize transportation cost, only for routed transports.
			model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j] * Or[j] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j]] * Or[j]) for i in S.keys() for j in H.keys()))

	else:
		if "direct" in TRANSPORTS:
			if "routed" in TRANSPORTS:
				if "cost" in OBJ:
					if "time" in OBJ:		# Both maximize cost and minimize reliability, for both routed and direct transports, and differentiating between product types.
						model.setObjectiveN(expr = quicksum((Cd[i,j] * y[i,j,p] * Od[j,p]) + (Cr_km[i,j] * y[i,j,p] * Or[j,p] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
						model.setObjectiveN(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
					else:					# Maximize transportation cost, for both routed and direct transports, and differentiating between product types.
						model.setObjective(expr = quicksum((Cd[i,j] * y[i,j,p] * Od[j,p]) + (Cr_km[i,j] * y[i,j,p] * Or[j,p] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()))
				else:						# Minimize reliability of delivery, for both routed and direct transports, and differentiating between product types.
					model.setObjective(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
			else:		
				if "cost" in OBJ:			
					if "time" in OBJ:		# Both maximize cost and minimize reliability, only for direct transports, and differentiating between product types.
						model.setObjectiveN(expr = quicksum(Cd[i,j] * y[i,j,p] * Od[j,p] for p in P.keys() for i in S.keys() for j in H.keys()), index=0, priority=prio["cost"], name="cost")
						model.setObjectiveN(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()), index=1, priority=prio["time"], name="time")
					else:					# Maximize transportation cost, only for direct transports, and differentiating between product types.
						model.setObjective(expr = quicksum(Cd[i,j] * y[i,j,p] * Od[j,p] for p in P.keys() for i in S.keys() for j in H.keys()))
				else:						# Minimize reliability of delivery, only for direct transports, and differentiating between product types.
					model.setObjective(expr = -1 * quicksum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()))
		else:								# Maximize transportation cost, only for routed transports, and differentiating between product types.
			model.setObjective(expr = quicksum((Cr_km[i,j] * y[i,j,p] * Or[j,p] * Cc[j]) + (Cr_stop[i] * y[i,Hc[j],p] * Or[j,p]) for p in P.keys() for i in S.keys() for j in H.keys()))

	# Minimize the objective functions.
	# Note that this is reversed w.r.t. optimize.py
	model.ModelSense = GRB.MAXIMIZE

	stop = time.perf_counter()
	calc_time = stop - start
	print(f"\nmodel initialization: {calc_time:0.4f} seconds")


	##############
	## OPTIMIZE ##
	##############

	start = time.perf_counter()
	model.optimize()
	stop = time.perf_counter()
	calc_time = stop - start
	print(f"\noptimize: {calc_time:0.4f} seconds")

	return model


def process_robustness_outputs(model, HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, **kwargs):

	# Get the values of the model variable as found for the optimal solution.
	OPd = np.zeros([len(H)])
	Od = np.zeros([len(H)])
	Or = np.zeros([len(H)])
	for var in model.getVars():
		name = re.split(r'\W+', var.varName)[0]
		if name == "OPd":
			index0 = int(re.split(r'\W+', var.varName)[1])
			OPd[index0] = var.X
		if name == "Od":
			index0 = int(re.split(r'\W+', var.varName)[1])
			Od[index0] = var.X
		if name == "Or":
			index0 = int(re.split(r'\W+', var.varName)[1])
			Or[index0] = var.X

	# Load parameters again.
	Orc = np.zeros([len(H)])
	Orc_real = np.zeros([len(H)])
	if "routed" in TRANSPORTS:
		Hc = kwargs["Hc"]
		Cr_km = kwargs["Cr_km"]
		Cr_stop = kwargs["Cr_stop"]
		if ORDER_DATA == True:
			Or_real = kwargs["Or"]
			for j in Hc.values():
				Orc[j] = Or[j]
				Orc_real[j] = Or_real[j]
	if "direct" in TRANSPORTS:
		Cd = kwargs["Cd"]
		if ORDER_DATA == True:
			OPd_real = kwargs["OPd"]
			Od_real = kwargs["Od"]

	# Retrieve open locations and allocations of the scenario.	
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "x")+".pickle",'rb') as f:
		x = pickle.load(f)
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "y")+".pickle",'rb') as f:
		y = pickle.load(f)

	# Reliability of delivery, both the worst case and real value, to be exported by the write_robustness_outputs function.
	rod = sum(OPd[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) * 100 / sum(OPd[j] for j in H.keys())
	rod_real = sum(OPd_real[j] * y[i,j] * U[i,j] for i in S.keys() for j in H.keys()) * 100 / sum(OPd_real[j] for j in H.keys())
	
	# Transportation cost, both the worst case and real value, and both in total and on average per hospital, 
	# to be exported by the write_robustness_outputs function.
	if "routed" in TRANSPORTS:
		if "direct" in TRANSPORTS:
			cost = sum((Cd[i,j]*y[i,j]*Od[j]) + (Cr_km[i,j]*y[i,j]*Orc[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j]) for i in S.keys() for j in H.keys())
			cost_real = sum((Cd[i,j]*y[i,j]*Od_real[j]) + (Cr_km[i,j]*y[i,j]*Orc_real[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or_real[j]) for i in S.keys() for j in H.keys())
			avg_cost = cost / sum((Od[j]+Orc[j]) for j in H.keys())
			avg_cost_real = cost_real / sum((Od_real[j] + Orc_real[j]) for j in H.keys())
		else:
			cost = sum((Cr_km[i,j]*y[i,j]*Orc[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or[j]) for i in S.keys() for j in H.keys())
			cost_real = sum((Cr_km[i,j]*y[i,j]*Orc_real[j]) + (Cr_stop[i]*y[i,Hc[j]]*Or_real[j]) for i in S.keys() for j in H.keys())
			avg_cost = cost / sum(Orc[j] for j in H.keys())
			avg_cost_real = cost_real / sum(Orc_real[j] for j in H.keys())
	else:
		cost = sum((Cd[i,j]*y[i,j]*Od[j]) for i in S.keys() for j in H.keys())
		cost_real = sum((Cd[i,j]*y[i,j]*Od_real[j]) for i in S.keys() for j in H.keys())
		avg_cost = cost / sum(Od_real[j] for j in H.keys())
		avg_cost_real = cost_real / sum(Od_real[j] for j in H.keys())

	# Difference between the true value and the worst case value of the cost and reliability.
	rod_diff = rod - rod_real
	cost_diff = ((cost - cost_real) / cost_real) * 100
	avg_cost_diff = (avg_cost - avg_cost_real) / avg_cost_real

	# Write all relevant output to a local file.
	write_robustness_outputs(HOME_DIR, SETTINGS, ORDER_DEVIATION, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, cost, avg_cost, rod, cost_real, avg_cost_real, rod_real, rod_diff, cost_diff, avg_cost_diff)	# model._best, model._bound


def process_robustness_outputs_pcat(model, HOME_DIR, SETTINGS, ORDER_DEVIATION, ROD_TIME, COUNTRY, src, n, H, S, U, prio, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, SCENARIO_NAME, **kwargs):

	P = kwargs["P"]
	m = kwargs["m"]

	# Get the values of the model variable as found for the optimal solution.
	OPd = np.zeros([len(H), len(P)])
	Od = np.zeros([len(H), len(P)])
	Or = np.zeros([len(H), len(P)])
	for var in model.getVars():
		name = re.split(r'\W+', var.varName)[0]
		if name == "OPd":
			index0 = int(re.split(r'\W+', var.varName)[1])
			index1 = int(re.split(r'\W+', var.varName)[2])
			OPd[index0, index1] = var.X
		if name == "Od":
			index0 = int(re.split(r'\W+', var.varName)[1])
			index1 = int(re.split(r'\W+', var.varName)[2])
			Od[index0, index1] = var.X
		if name == "Or":
			index0 = int(re.split(r'\W+', var.varName)[1])
			index1 = int(re.split(r'\W+', var.varName)[2])
			Or[index0, index1] = var.X

	# Load parameters again.
	Orc = np.zeros([len(H), len(P)])
	Orc_real = np.zeros([len(H), len(P)])
	if "routed" in TRANSPORTS:
		Hc = kwargs["Hc"]
		Cr_km = kwargs["Cr_km"]
		Cr_stop = kwargs["Cr_stop"]
		if ORDER_DATA == True:
			Or_real = kwargs["Or"]
			for j in Hc.values():
				Orc[j] = Or[j]
				Orc_real[j] = Or_real[j]
	if "direct" in TRANSPORTS:
		Cd = kwargs["Cd"]
		if ORDER_DATA == True:
			OPd_real = kwargs["OPd"]
			Od_real = kwargs["Od"]

	# Retrieve open locations and allocations of the scenario.	
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "x", **kwargs)+".pickle",'rb') as f:
		x = pickle.load(f)
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "y", **kwargs)+".pickle",'rb') as f:
		y = pickle.load(f)
	with open(generate_filename(HOME_DIR, settings(P_CAT, ORDER_DATA, TRANSPORTS, ["cost"]), ["cost"], P_CAT, n, ROD_TIME, COUNTRY, src, {"cost":0}, "z", **kwargs)+".pickle",'rb') as f:
		z = pickle.load(f)

	# Reliability of delivery, both the worst case and real value, to be exported by the write_robustness_outputs function.
	rod = sum(OPd[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) * 100 / sum(sum(OPd[j,:]) for j in H.keys())
	rod_real = sum(OPd_real[j,p] * y[i,j,p] * U[i,j] for i in S.keys() for j in H.keys() for p in P.keys()) * 100 / sum(sum(OPd_real[j,:]) for j in H.keys())
	
	# Transportation cost, both the worst case and real value, and both in total and on average per hospital, 
	# to be exported by the write_robustness_outputs function.
	if "routed" in TRANSPORTS:
		if "direct" in TRANSPORTS:
			cost = sum((Cd[i,j]*y[i,j,p]*Od[j,p]) + (Cr_km[i,j]*y[i,j,p]*Orc[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
			cost_real = sum((Cd[i,j]*y[i,j,p]*Od_real[j,p]) + (Cr_km[i,j]*y[i,j,p]*Orc_real[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or_real[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
			avg_cost = cost / sum((Od[j,p]+Orc[j,p]) for j in H.keys() for p in P.keys())
			avg_cost_real = cost_real / sum((Od_real[j,p] + Orc_real[j,p]) for j in H.keys() for p in P.keys())
		else:
			cost = sum((Cr_km[i,j]*y[i,j,p]*Orc[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
			cost_real = sum((Cr_km[i,j]*y[i,j,p]*Orc_real[j,p]) + (Cr_stop[i]*y[i,Hc[j],p]*Or_real[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
			avg_cost = cost / sum(Orc[j,p] for j in H.keys() for p in P.keys())
			avg_cost_real = cost_real / sum(Orc_real[j,p] for j in H.keys() for p in P.keys())
	else:
		cost = sum((Cd[i,j]*y[i,j,p]*Od[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
		cost_real = sum((Cd[i,j]*y[i,j,p]*Od_real[j,p]) for i in S.keys() for j in H.keys() for p in P.keys())
		avg_cost = cost / sum(Od_real[j,p] for j in H.keys() for p in P.keys())
		avg_cost_real = cost_real / sum(Od_real[j,p] for j in H.keys() for p in P.keys())

	# Difference between the true value and the worst case value of the cost and reliability.
	rod_diff = rod - rod_real
	cost_diff = ((cost - cost_real) / cost_real) * 100
	avg_cost_diff = (avg_cost - avg_cost_real) / avg_cost_real

	# Write all relevant output to a local file.
	write_robustness_outputs(HOME_DIR, SETTINGS, ORDER_DEVIATION, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, cost, avg_cost, rod, cost_real, avg_cost_real, rod_real, rod_diff, cost_diff, avg_cost_diff, m=m)	# model._best, model._bound
