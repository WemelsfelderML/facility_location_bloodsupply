# Determining optimal locations for blood distribution centers
In this repository you can find the code corresponding to the paper "Determining optimal locations for blood distribution centers". 


## Preparations

Clone the github code to your machine.
Get the path to the directory where you have the code, and change the HOME_DIR parameter in **main.py** to this path.
Create a directory for all of your data, and change the INPUT_DIR parameter to this path.

### Locations
Create a dataframe of the following shape, containing all locations to be used, call it **F.csv**, and store it in your data directory.
The "type" column should contain an "E" for current DC locations, an "H" for hospital locations, and an "R" for self-generated candidate DC locations.
 column   type  	description
 ------ --------  ---------------
  name   object 	location name
  type   object 	"E", "H", "R"	
  lat    float64	latitude
  lon    float64	longitude

Create a dictionary and store it as **H.pickle**:
- values: all the hospital names contained in **F.csv** 
- keys: the indices of these hospitals in the _order_ and _cost_ matrices.

### Ordering data
If you have access to hospital ordering data and wish to use this in the model, create the following files. Also make sure to set the ORDER_DATA parameter to _True_.
These are matrices (numpy arrays) of shape H×P, with all hospitals (H) on the vertical axis and all product types (P) on the horizontal axis.
If you do _not_ with to distinguish product types, simply put all data in the left column.

Create these files only when including direct transports (make sure to include "direct" in the TRANSPORTS parameter).
**OPd.pickle**: OPd[j,p] contains the total number of products p∈P in direct transports to hospital j∈H.
**Od.pickle**: Od[j,p] contains the total number of direct transports of product p∈P to hospital j∈H.

Create these files only when including routed transports (make sure to include "routed" in the TRANSPORTS parameter).
**Or.pickle**: Or[j,p] contains the total number of routed transports of product p∈P to hospital j∈H.
**Orc.pickle**: Orc[j,p] contains the total number of routed transports of product p∈P to hospital j∈H if j is a cluster center.
**Hc.pickle**: Dictionary with _key_: same as the keys of **H.pickle**, _value_: index of the largest hospital (cluster center) of its cluster.

### Estimations of transportation cost and time

Create several dataframes, with on both axes all hospital locations and candidate DC locations.
All rows and columns should be indexed by the location's name as denoted in the "name" column of **F.csv**.
**Dt.csv**: transportation time from the DC location (row) to the hospital location (column), format = H:MM:SS.
**Cd.csv**: transportation cost of a direct transport from the DC location (row) to the hospital location (column) and back.
**Cr_km.csv**: transportation cost per km of a routed transport.
**Cr_stop.csv**: cost per delivery stop of a routed transport.

## Settings

In **main.py** several settings should be modified to make sure you are obtaining the desired results.

SOURCES determines which locations will be used as candidate DC locations. It is a list containing one or more of the characters "E", "F" and "R", refering to the locations in **F.csv**. With "E" only the current DC locations will be considered, with "F" both current DC locations and all hospital locations will be considered, and with "R" a manually created set of locations will be used as candidates. The optimizer will be executed separately for all elements in the SOURCES list. In case "R" is in the list, the value of parameter SCENARIO_NAME should be set to a name for this set, and separate files should be created for locations, transportation time and cost, also containing this name.

N is a list containing the number of DCs that should be located. The optimizer will be executed for each element of this list separately.

ROD_TIME contains the maximum delivery time that is allowed for emergency deliveries, in order to calculate the reliability of delivery. As the maximum time for academic hospitals and regular hospitals may differ, these are separated.

OBJ is a list containing the objectives to be used. When "cost" is in the list, the transportation cost will be minimized. When "time" is in the list, the reliability of delivery will be maximized. If both elements are in the list, both objectives will be optimized. Then the optimizer will be executed twice, each time with a different objective as its priority. If "time" is not in the OBJ parameter, then a constraint is added to the model to obtain a minimal reliability of delivery. State the minimal required value in parameter ROD_MIN.

TRANSPORTS contains the type of transports that is considered, being "direct", "routed" or both.

P_CAT states whether different product types (slow-movers and fast-movers) are distinguished, being either _True_ or _False_.

If you wish to visualize the results on a map, you should create a Mapbox account and generate a token. Set the value of parameter MAPBOX_TOKEN to this token.