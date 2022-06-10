import numpy as np
import pandas as pd
import re
import codecs
import json
import pickle
import kaleido
from plotly.graph_objs import *
from plotly.graph_objs import scattermapbox as smb
from plotly.graph_objs import scatter
from plotly.offline import plot
import time
import os
from mapboxgl.utils import *
from mapboxgl.viz import *

from utils import *

# Visualize the optimal solution found on e map.
def make_map(HOME_DIR, SETTINGS, MAPBOX_TOKEN, OBJ, ORDER_DATA, TRANSPORTS, P_CAT, ROD_TIME, n, lonlat, S, H, U, COUNTRY, src, prio, **kwargs):


	#################
	## GATHER DATA ##
	#################

	# Retrieve parameters.

	if P_CAT == True:
		P = kwargs["P"]
		m = kwargs["m"]

	if "routed" in TRANSPORTS:
		Hc = kwargs["Hc"]

	if ORDER_DATA == True:
		if P_CAT == True:
			O = np.zeros([len(H),len(P)])
		else:
			O = np.zeros([len(H)])

		if "routed" in TRANSPORTS:
			Or = kwargs["Or"]
			O += Or
		if "direct" in TRANSPORTS:
			OPd = kwargs["OPd"]
			Od = kwargs["Od"]
			O += Od


	# Retrieve stored optimal values for model variables.	
	if P_CAT == True:
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "z", m=m)+".pickle",'rb') as f:
			z = pickle.load(f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "x", m=m)+".pickle",'rb') as f:
			x = pickle.load(f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y", m=m)+".pickle",'rb') as f:
			y = pickle.load(f)
	else:
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "x")+".pickle",'rb') as f:
			x = pickle.load(f)
		with open(generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "y")+".pickle",'rb') as f:
			y = pickle.load(f)

	# Determine for each hospital whether it is served within the given time limit for emergency orders.
	if P_CAT == True:
		e = np.zeros([len(H), len(P)])
		for j in H.keys():
			e[j,0] = sum(y[i,j,0] * U[i,j] for i in S.keys())
			e[j,1] = sum(y[i,j,1] * U[i,j] for i in S.keys())
	else:
		e = np.zeros([len(H)])
		for j in H.keys():
			e[j] = sum(y[i,j] * U[i,j] for i in S.keys())

	# Get all hospital locations, and sort such that for each hospital the index is the same as its value for j in H.
	hospitals = lonlat[lonlat["type"]=="H"]
	hospitals = hospitals.loc[[H[j] for j in range(len(H))],].reset_index()
	
	# For each hospital, determine its demand, and let the dot size of this hospital on the map depend on this demand.
	if ORDER_DATA == True:
		if P_CAT == True:
			hospitals["demand"] = [sum(list(O[j])) for j in hospitals.index]
			hospitals["demand_emergency"] = [sum(list(OPd[j])) for j in hospitals.index]
			hospitals["demand_emergency_slowmov"] = [OPd[j,1] for j in hospitals.index]
		else:
			hospitals["demand"] = [O[j] for j in hospitals.index]
			hospitals["demand_emergency"] = [OPd[j] for j in hospitals.index]
		hospitals["size"] = (((hospitals["demand"] / hospitals["demand"].max()) * 35) + 10).apply(lambda x: float(x))
	else:
		hospitals["size"] = [15 for i in range(len(hospitals))]

	# Distinguish hospitals with and without orders, these will be shown in a different color on the map.
	if ORDER_DATA == True:
		hospitals_no_orders = hospitals[hospitals["demand"]==0]
		hospitals = hospitals[hospitals["demand"]>0]

	# Determine which hospitals are not within the time limit for emergency orders from any DC, these will be shown in a different color on the map.
	if P_CAT == True:
		hospitals_too_far_allprod = hospitals.loc[[j for j in hospitals.index if sum(e[j,:]) == 0],]
		hospitals_too_far_slowmov = hospitals.loc[[j for j in hospitals.index if e[j,1] < e[j,0]],]
		if ORDER_DATA == True:
			hospitals_too_far_allprod = hospitals_too_far_allprod[hospitals_too_far_allprod["demand_emergency"]>0]
			hospitals_too_far_slowmov = hospitals_too_far_slowmov[hospitals_too_far_slowmov["demand_emergency_slowmov"]>0]
	else:
		hospitals_too_far = hospitals.loc[[j for j in hospitals.index if e[j] == 0],]
		if ORDER_DATA == True:
			hospitals_too_far = hospitals_too_far[hospitals_too_far["demand_emergency"]>0]

	# For each DC, determine how many orders are supplied from this location, and let the dot size of this DC on the map depend on this supply.
	DCs = pd.DataFrame.from_dict(S, orient ='index').rename(columns={0:"name"})
	DCs = DCs.merge(lonlat.reset_index(), on="name", how="left")
	DCs = DCs.loc[[i for i in S.keys() if x[i] == 1],]
	if ORDER_DATA == True:
		if P_CAT == True:
			DCs["flow"] = [sum(y[i,j,p]*O[j,p] for j in H.keys() for p in P.keys()) for i in DCs.index]
		else:
			DCs["flow"] = [sum(y[i,j]*O[j] for j in H.keys()) for i in DCs.index]
	else:
		if P_CAT == True:
			DCs["flow"] = [sum(sum(y[i,:,:])) for i in DCs.index]
		else:
			DCs["flow"] = [sum(y[i,:]) for i in DCs.index]
	DCs["size"] = (((DCs["flow"] / DCs["flow"].max()) * 40) + 15).apply(lambda x: float(x))

	# If we distinguish between product categories, find which DCs store slow-moving products, and show those in a different color on the map.
	if P_CAT == True:
		DCs_allprod = DCs.loc[[i for i in DCs.index if z[i] == 1],]
		DCs_fastmov = DCs.loc[[i for i in DCs.index if z[i] == 0],]

	# Create a dataframe to store all paths.
	paths = pd.DataFrame(columns=["from", "to", "start_lat", "start_lon", "end_lat", "end_lon"])

	# Function that adds a path to the data frame when given a DC location i∈S and a hospital location j∈H.
	def add_path(paths, lonlat, S, H, i, j):
		paths.loc[len(paths)] = [S[i], H[j], float(lonlat.loc[S[i],"lat"]), float(lonlat.loc[S[i],"lon"]), float(lonlat.loc[H[j],"lat"]), float(lonlat.loc[H[j],"lon"])]
		return paths

	# Check for all combinations of i∈S and j∈H whether hospital j is allocated to DC i. If so, add this path to the dataframe.
	if P_CAT == True:
		if ORDER_DATA == True:
			for i in S.keys():
				for j in H.keys():
					if ("routed" in TRANSPORTS and sum(y[i,Hc[j],:])>0 and sum(Or[j,:]) > 0) or ("direct" in TRANSPORTS and sum(y[i,j,:])>0 and sum(Od[j,:])>0):
						paths = add_path(paths, lonlat, S, H, i, j)
		else:
			for i in S.keys():
				for j in H.keys():
					if (("routed" in TRANSPORTS) and (sum(y[i,Hc[j],:])>0)) or (("direct" in TRANSPORTS) and (sum(y[i,j,:])>0)):
						paths = add_path(paths, lonlat, S, H, i, j)
	else:
		if ORDER_DATA == True:
			for i in S.keys():
				for j in H.keys():
					if ("routed" in TRANSPORTS and y[i,Hc[j]]>0 and Or[j] > 0) or ("direct" in TRANSPORTS and y[i,j]>0 and Od[j]>0):
						paths = add_path(paths, lonlat, S, H, i, j)
		else:
			for i in S.keys():
				for j in H.keys():
					if (("routed" in TRANSPORTS) and (y[i,Hc[j]]>0)) or (("direct" in TRANSPORTS) and (y[i,j]>0)):
						paths = add_path(paths, lonlat, S, H, i, j)


	################
	## CREATE MAP ##
	################


	# Initialize the figure.
	fig = Figure()

	# Draw all allocation paths (grey).
	for i in range(len(paths)):
		fig.add_trace(
			Scattermapbox(
				lon = [paths['start_lon'][i], paths['end_lon'][i]],
				lat = [paths['start_lat'][i], paths['end_lat'][i]],
				mode = 'lines',
				line = dict(width = 1.2,color = '#06043D'),
				opacity = 0.3,
				showlegend = False,
				hoverinfo = "skip",
			)
		)

	# Draw all hospitals that can be served within the time limit for emergency orders (dark orange).
	fig.add_trace(
		Scattermapbox(
			lat=hospitals["lat"],
			lon=hospitals["lon"],
			mode='markers', 
			name="hospital · served ≤ max time", 
			legendgroup="hospitals",
			legendgrouptitle={"text":"hospitals"},
			marker=smb.Marker(
				size=hospitals["size"],
				color='#FF812C',
				opacity=0.95
			), 
			text=hospitals["name"],
			hoverinfo="text",
		),
	)
	

	if ORDER_DATA == True:

		# Draw all hospitals that did not place any orders (light orange).
		fig.add_trace(
			Scattermapbox(
				lat=hospitals_no_orders["lat"],
				lon=hospitals_no_orders["lon"],
				mode='markers', 
				name='hospital · no orders', 
				legendgroup="hospitals",
				legendgrouptitle={"text":"hospitals"},
				marker=smb.Marker(
					size=10, 
					color='#FFB27D',
					opacity=0.7
				), 
				text=hospitals_no_orders["name"],
				hoverinfo="text",
			),
		)

	if P_CAT == True:

		fig.add_traces([
			# Draw all hospitals that can not be served within the time limit for emergency orders (dark green).
			Scattermapbox(
				lat=hospitals_too_far_allprod["lat"],
				lon=hospitals_too_far_allprod["lon"],
				mode='markers', 
				name="hospital · served > max time for all products", 
				legendgroup="hospitals",
				legendgrouptitle={"text":"hospitals"},
				marker=smb.Marker(
					size=hospitals_too_far_allprod["size"], 
					color='#038E77',
					opacity=0.95
				), 
				text=hospitals_too_far_allprod["name"],
				hoverinfo="text",
			),

			# Draw all hospitals of which only the slow-moving products can not be served within the time limit (light green).
			Scattermapbox(
				lat=hospitals_too_far_slowmov["lat"],
				lon=hospitals_too_far_slowmov["lon"],
				mode='markers', 
				name="hospital · served > max time for slow-movers, ≤ max time for fast-movers", 
				legendgroup="hospitals",
				legendgrouptitle={"text":"hospitals"},
				marker=smb.Marker(
					size=hospitals_too_far_slowmov["size"], 
					color='#62C9B8',
					opacity=0.95
				), 
				text=hospitals_too_far_slowmov["name"],
				hoverinfo="text",
			),

			# Draw all DC locations that store only fast-moving products (light purple).
			Scattermapbox(
				lat=DCs_fastmov["lat"],
				lon=DCs_fastmov["lon"],
				mode='markers', 
				name='DC · only fast-movers in stock',
				legendgroup="DCs",
				legendgrouptitle={"text":"DCs"},
				marker=smb.Marker(
					size=DCs_fastmov["size"], 
					color='#AB60C0',
					opacity=0.95
				), 
				text=DCs_fastmov["name"],
				hoverinfo="text",
			),

			# Draw all DC locations that store both fast-moving and slow-moving products (dark purple).
			Scattermapbox(
				lat=DCs_allprod["lat"],
				lon=DCs_allprod["lon"],
				mode='markers', 
				name='DC · all products in stock', 
				legendgroup="DCs",
				legendgrouptitle={"text":"DCs"},
				marker=smb.Marker(
					size=DCs_allprod["size"], 
					color='#510267',
					opacity=0.95
				), 
				text=DCs_allprod["name"],
				hoverinfo="text",
			)
		])

	else:
		fig.add_traces([
			# Draw all hospitals that can not be served within the time limit for emergency orders (dark green).
			Scattermapbox(
				lat=hospitals_too_far["lat"],
				lon=hospitals_too_far["lon"],
				mode='markers', 
				name="hospital · served > max time", 
				legendgroup="hospitals",
				legendgrouptitle={"text":"hospitals"},
				marker=smb.Marker(
					size=hospitals_too_far["size"], 
					color='#038E77',
					opacity=0.95
				), 
				text=hospitals_too_far["name"],
				hoverinfo="text",
			),

			# Draw DC locations (dark purple).
			Scattermapbox(
				lat=DCs["lat"],
				lon=DCs["lon"],
				mode='markers', 
				name='DC',
				legendgroup="DCs",
				legendgrouptitle={"text":"DCs"},
				marker=smb.Marker(
					size=DCs["size"], # *1.5
					color='#510267',
					opacity=0.95
				), 
				text=DCs["name"],
				hoverinfo="text",
			),

			# # Draw the grid of candidate locations (pink).
			# Scattermapbox(
		 #        lat=R[R["type"]=="R"]["lat"],
		 #        lon=R[R["type"]=="R"]["lon"],
		 #        mode='markers', 
		 #        marker=smb.Marker(
		 #        	size=9,
		 #            color='#DF3787',
		 #            opacity=0.9),
		 #        showlegend = False,
		 #        text=[R[R["type"]=="R"]["name"][i] + ": " + str(R[R["type"]=="R"]["lat"][i]) + ", " + str(R[R["type"]=="R"]["lon"][i]) for i in R[R["type"]=="R"].index],
		 #        hoverinfo="text",
		 #    ),
		])

	

	fig.update(layout_showlegend=False)

	if COUNTRY == "NL":
		fig.update_layout(Layout(
			height = 900, width = 750, 
			mapbox = dict(
			    accesstoken = MAPBOX_TOKEN,
			    center = {"lat":52.110851, "lon":5.303027},
			    zoom = 6.7,
			    style = 'streets')
			))
	if COUNTRY == "FI":
		fig.update_layout(Layout(
			height = 900, width = 750, 
			mapbox=dict(
			    accesstoken = MAPBOX_TOKEN,
			    center = {"lat":64.4, "lon":26},
			    zoom=4.6, 
			    style='streets')
			))
	else:
		fig.update_layout(Layout(
			height = 900, width = 750, 
			mapbox=dict(
			    accesstoken = MAPBOX_TOKEN,
			    center = {"lat":hospitals["lat"].mean(), "lon":hospitals["lon"].mean()},
			    zoom=5, 
			    style='streets')
			))

	# Write the visualization to an interactive html file.
	if P_CAT == True:
		file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "map", m=m)
	else:
		file = generate_filename(HOME_DIR, SETTINGS, OBJ, P_CAT, n, ROD_TIME, COUNTRY, src, prio, "map")
	fig.write_html(file+".html")