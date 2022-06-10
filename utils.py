import pandas as pd
import numpy as np
import math
import os
import numbers


# Generate the file path and name of any to-be-exported file (e.g. variables, outputs, visualization)
def generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, file_type, **kwargs):

	base = HOME_DIR + "results/" + SETTINGS + COUNTRY + "_" + src + "/H" + str(ROD_TIME["H"][0]).zfill(2) + "." + str(ROD_TIME["H"][1]).zfill(2) + "_UH" + str(ROD_TIME["UH"][0]).zfill(2) + "." + str(ROD_TIME["UH"][1]).zfill(2) + "/" 

	if file_type in ["outputs", "robustness"]:
		return HOME_DIR + "results/" + SETTINGS + COUNTRY + "_" + src + "/" + file_type
	else:
		if P_CAT == True:
			return base + file_type + "_n" + str(n) + "-m" + str(kwargs["m"]) + "_" + "-".join([p[0] + str(prio[p]) for p in OBJ])
		else:
			return base + file_type + "_n" + str(n) + "_" + "-".join([p[0] + str(prio[p]) for p in prio.keys()])

# Write the outputs of optimizing DC locations to a local file.
def write_output(HOME_DIR, SETTINGS, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, cost, avg_cost, rod, model_data, calc_time, nvars, model_status, **kwargs):

	file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "outputs") + ".csv"

	# Determine which candidate locations are chosen to open a DC.
	open_locs = []
	for i in S.keys():
		if x[i] == 1:
			open_locs.append(S[i])

	# Determine which hospitals can not be served within the time limit for emergency orders.
	unreachable = []
	for j in H.keys():
		if sum(U[i,j]*x[i] for i in S.keys()) == 0:
			unreachable.append(H[j])

	# Retreive the lower and upper bound values that were stored in the callback function during the optimization.
	mipgap_columns = []
	mipgap_values = []
	for obj in OBJ:
		mipgap_columns += ["obj_best_"+obj, "obj_bound_"+obj, "obj_gap_"+obj, "update_time_"+obj]

		log = pd.DataFrame(model_data, columns=["obj", "calc_time", "best", "bound"])
		log = log.loc[log["obj"]==obj].sort_values(by="calc_time", ascending=False).reset_index()
		if len(log) > 0:
			last = dict(log.loc[0])
			mipgap_values += [last["best"], last["bound"], abs(last["bound"]-last["best"])/abs(last["best"]), last["calc_time"]]
		else:
			mipgap_values += ["-"] * 4

	# Either create a new file or append outputs to an existing file.
	if os.path.exists(file):
		outputs = pd.read_csv(file, sep=",", encoding = "1250")
	else:
		outputs = pd.DataFrame(columns=["cost", "avg_cost", "rod", "n", "m", "p_cost", "p_time"] + mipgap_columns + ["calc_time", "nvars", "model_status", "open_locs", "unreachable"])

	# Add all objective values and other relevant information to the output dataframe.
	r = len(outputs.index)
	outputs.loc[r,:] = ["-" for i in range(len(outputs.columns))]
	outputs.loc[r, ["cost", "avg_cost", "rod"]] = [cost, avg_cost, rod]
	outputs.loc[r, mipgap_columns] = mipgap_values
	outputs.loc[r, ["n", "A1_max_time (H)", "A1_max_time (UH)"]] = [n, str(ROD_TIME["H"][0]).zfill(2)+":"+str(ROD_TIME["H"][1]).zfill(2), str(ROD_TIME["UH"][0]).zfill(2)+":"+str(ROD_TIME["UH"][1]).zfill(2)]
	outputs.loc[r, ["calc_time", "nvars", "model_status", "open_locs", "unreachable"]] = [f"{calc_time:0.4f} sec", round(nvars,0), model_status, ",".join(open_locs), ",".join(unreachable)]
	for p in OBJ:
		outputs.loc[r, "p_"+p] = prio[p]
	if P_CAT == True:
		outputs.loc[r, "m"] = kwargs["m"]

	# Write the dataframe with outputs to a local file.
	outputs.to_csv(file, sep=',', encoding="1250", index=False)


# Write the outputs of robustness testing to a local file.
def write_robustness_outputs(HOME_DIR, SETTINGS, ORDER_DEVIATION, OBJ, P_CAT, ROD_TIME, COUNTRY, src, n, H, S, U, prio, x, cost, avg_cost, rod, cost_real, avg_cost_real, rod_real, rod_diff, cost_diff, avg_cost_diff, **kwargs):

	file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "robustness") + ".csv"

	# Either create a new file or append outputs to an existing file.
	if os.path.exists(file):
		outputs = pd.read_csv(file, sep=",")
	else:
		outputs = pd.DataFrame(columns=["cost", "avg_cost", "rod", "cost_real", "avg_cost_real", "rod_real", "cost_diff", "avg_cost_diff", "rod_diff", "deviation", "n", "m", "A1_max_time", "p_cost", "p_time""calc_time"])

	# Add all test results to the outputs dataframe.
	r = len(outputs.index)
	outputs.loc[r,:] = ["-" for i in range(len(outputs.columns))]
	outputs.loc[r, ["deviation"]] = [ORDER_DEVIATION * 100]
	outputs.loc[r, ["cost", "avg_cost", "rod"]] = [cost, avg_cost, rod]
	outputs.loc[r, ["cost_real", "avg_cost_real", "rod_real"]] = [cost_real, avg_cost_real, rod_real]
	outputs.loc[r, ["cost_diff", "avg_cost_diff", "rod_diff"]] = [cost_diff, avg_cost_diff, rod_diff]
	outputs.loc[r, ["n", "A1_max_time (H)", "A1_max_time (UH)"]] = [n, str(ROD_TIME["H"][0]).zfill(2)+":"+str(ROD_TIME["H"][1]).zfill(2), str(ROD_TIME["UH"][0]).zfill(2)+":"+str(ROD_TIME["UH"][1]).zfill(2)]
	for p in OBJ:
		outputs.loc[r, "p_"+p] = prio[p]
	if P_CAT == True:
		outputs.loc[r, "m"] = kwargs["m"]

	# Write the data frame to a local file.
	outputs.to_csv(file, sep=',', encoding="1250", index=False)


# At the start of an optimization episode, print the chosen settings in the terminal.
def print_iteration_in_terminal(OBJ, ORDER_DATA, TRANSPORTS, P_CAT, src, ROD_TIME, prio, n, **kwargs):

	additions = {
		"order data": ORDER_DATA,
		"product categories": P_CAT
	}

	print("\nSTART OPTIMIZER")

	add = [s for s in additions.keys() if additions[s] == True]
	if len(add) == 0:
		print("Model additions: none")
	else:
		print("Model additions: " + ", ".join(add))
	print("Transports: " + ", ".join(TRANSPORTS))
	if len(OBJ) > 1:
		print("Objectives: " + ", ".join(OBJ))
	else:
		print("Objective: " + OBJ[0])
	print("Source: " + src)
	print("DR max time (H): " + str(ROD_TIME["H"][0]).zfill(2) + ":" + str(ROD_TIME["H"][1]).zfill(2))
	print("DR max time (UH): " + str(ROD_TIME["UH"][0]).zfill(2) + ":" + str(ROD_TIME["UH"][1]).zfill(2))
	if P_CAT == True:
		print("n: " + str(n) + ", m: " + str(kwargs["m"]))
	else:
		print("n: " + str(n))
	if len(OBJ) > 1:
		[print("Obj. priority: " + i) for i in prio.keys() if prio[i] == 1]


# After finishing an optimization episode, print the model status in the terminal.
def status_code_to_message(code):

	status_code = {
		1 : "MODEL IS LOADED, BUT NO SOLUTION IS AVAILABLE",
		2 : "MODEL IS SOLVED OPTIMALLY",
		3 : "MODEL IS INFEASIBLE",
		4 : "MODEL IS EITHER INFEASIBLE OR UNBOUNDED\nTo obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.",
		5 : "MODEL IS UNBOUNDED\nAn unbounded ray allows the objective to improve without limit.",
		6 : "NO SOLUTION AVAILABLE\nThe optimal objective was worse than the Cutoff parameter.",
		7 : "OPTIMIZATION TERMINATED\nIterationLimit or BarIterLimit parameter was exceeded.",
		8 : "OPTIMIZATION TERMINATED\nNodeLimit parameter was exceeded.",
		9 : "OPTIMIZATION TERMINATED\nTimeLimit parameter was exceeded.",
		10 : "OPTIMIZATION TERMINATED\nSolutionLimit parameter was exceeded.",
		11 : "OPTIMIZATION TERMINATED BY USER\nObtained results are not saved.",
		12 : "OPTIMIZATION TERMINATED\nUnrecoverable numerical difficulties.",
		13 : "UNABLE TO SATISFY OPTIMALITY\nA sub-optimal solution is available.",
		14 : "ASYNCHRONOUS CALL WAS MADE, ASSOCIATED OPTIMIZATION NOT YET COMPLETE",
		15 : "LIMIT SET BY USER WAS EXCEEDED\nThis is either a bound on the best objective or the best bound."
	}

	return status_code[code]

# Translate settings that were chosen in main.py to text, in order to make directories for storing the results.
def settings(P_CAT, ORDER_DATA, TRANSPORTS, OBJ):
	
	s_names = []
	if P_CAT == True:
		s_names.append("PCAT")
	if ORDER_DATA == True:
		s_names.append("DATA")
	if "routed" in TRANSPORTS:
		if "direct" in TRANSPORTS:
			s_names.append("RD")
		else:
			s_names.append("R")
	else:
		s_names.append("D")
	if "cost" in OBJ:
		if "time" in OBJ:
			s_names.append("CT")
		else:
			s_names.append("C")
	else:
		s_names.append("T")

	return "_".join(s_names) + "/"

