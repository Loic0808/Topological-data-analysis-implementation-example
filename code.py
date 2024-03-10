import numpy as np
import pandas as pd
import gudhi as gd
from sympy import *
from ripser import Rips
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

%md
#### Create dataset
When done test the algorithm with a dataset with 4 classes where the 4th class is given by just two points in a 4th radius and only one of them is marked

# Function to generate points for concentric rings
def generate_concentric_ring_points(num_points, center, min_radius, max_radius):
    r = np.sqrt(np.random.uniform(min_radius**2, max_radius**2, num_points))
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

# Generate points for three concentric rings
center = (0, 0)
min_radius = 1
max_radius = 2
num_points_per_ring = 100
 
ring1_points = generate_concentric_ring_points(num_points_per_ring, center, min_radius, max_radius)
ring2_points = generate_concentric_ring_points(num_points_per_ring, center, max_radius + 0.5, max_radius + 1.5)
ring3_points = generate_concentric_ring_points(num_points_per_ring, center, max_radius + 2, max_radius + 3)
 
# Generate 3 more points
num_points = 3
x_coords = np.random.uniform(-3, -2.5, num_points)
y_coords = np.random.uniform(4.5, 4.8, num_points)
 

# Create dataframes
data1 = pd.DataFrame(ring1_points, columns=['x', 'y'])
data2 = pd.DataFrame(ring2_points, columns=['x', 'y'])
data3 = pd.DataFrame(ring3_points, columns=['x', 'y'])
data4 = pd.DataFrame({'x': x_coords, 'y': y_coords})


# Assign labels
data1['label'] = 0
data2['label'] = 0
data3['label'] = 0
data4['label'] = 0
 
# Randomly select points and assign labels
data1.loc[np.random.choice(num_points_per_ring, 20, replace=False), 'label'] = 1
data2.loc[np.random.choice(num_points_per_ring, 20, replace=False), 'label'] = 2
data3.loc[np.random.choice(num_points_per_ring, 20, replace=False), 'label'] = 3
 
random_index = np.random.randint(0, len(data4))
random_row = data4.iloc[random_index]
random_row['label'] = 4
data4.iloc[random_index] = random_row
 
# Combine dataframes
data = pd.concat([data1, data2, data3, data4], ignore_index = True)
 
# Define colors based on labels
colors = {1: 'green', 2: 'red', 3: 'black', 4: 'orange', 0: 'blue'}

# Plotting
fig, ax = plt.subplots()
for label, group in data.groupby('label'):
    ax.scatter(group['x'], group['y'], c=colors[label], label=label)
ax.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot with colored labels')
plt.show()
 
%md
###### The points with Label = 0 are the ones we want to classifiy (unlabeled). We need to pay attention that the dimension of the data (in our case 2) is much smaller than the number of points of our data (300). The set P is given by all points, the set S is given by the points which have a label 1, 2 or 3, and the set X is given by the points with Label 0.
 
 
# Compute distances between each point
# Extract x and y coordinates as a numpy array
points = data[['x', 'y']].values
 
# Compute pairwise Euclidean distances
#dist_matrix = cdist(points, points)
 
#print(dist_matrix)
 
%md
#### Construct a filtered simplicial complex K for the data
####Some theory:

When constructing simplicial Rips complexes using Gudhi, it is not yet a filtered simplicial complex, since it's only a complex for a specific epsilon. We need to iterate over the epsilon to get a filtered structure.
 
Algorithm 2 in the paper
 
epsilon = 0.2 # epsilon is between 0 and 1. Remark, when taking a large epsilon, we obtain the same result as on the ripser diagramm above.
epsilon_i = 0.001
max_filtration_level = 10
i = 0
# K represents the filtered simplicial complex. We initialise it with an empty complex
K = gd.SimplexTree()
 
#The following list of tuppels will give the filtration value (see definition) for each simplex
Filtration_value = []


 
while i <= max_filtration_level:
    # Build simplicial Rips complex
    skeleton = gd.RipsComplex(
    #distance_matrix = dist_matrix,
    points = points,
    max_edge_length = epsilon_i
    )
    K_i = skeleton.create_simplex_tree(max_dimension = 2)
    print(K_i != K)
    print(i)
    if K_i != K:
        # Join K_i to K
        rips_filtration = K_i.get_filtration()
        rips_list = list(rips_filtration)
        for splx in rips_list :
            K.insert(splx[0], splx[1])
            ### This section has been modified from the original code
            Filtration_value.append((splx[0], epsilon_i))
            ###
        i += 1
        epsilon_i = epsilon_i + epsilon
    else:
        break

# Dictionary to store the smallest x value for each unique list
min_x_values = {}
 
# Iterate through the list of tuples
for sublist, x in Filtration_value:
    sublist_tuple = tuple(sublist)  # Convert the list to a tuple
    if sublist_tuple in min_x_values:
        # If sublist already exists in min_x_values, update the minimum x value if necessary
        if x < min_x_values[sublist_tuple]:
            min_x_values[sublist_tuple] = x
    else:
        # If sublist is not in min_x_values, add it with its corresponding x value
        min_x_values[sublist_tuple] = x
 
# Filter the original list based on the minimum x values
Xi_K = [(sublist, x) for sublist, x in Filtration_value if x == min_x_values[tuple(sublist)]]
 
K.dimension()
K.num_vertices()
K.num_simplices()

# Other way to get persistence intervals
BarCodes_Rips = K.persistence()
#for i in range(20):
#    print(BarCodes_Rips[i])
 
# Access directly specific dimension
#K.persistence_intervals_in_dimension(2)
 
%md
We don't use the whole complex K, we will choose a good approximation K_i for the computations. Also algorithm 3 is directly given in the Gudhi library by .persistence_intervals_in_dimension()
 
# Function so that life(d) is well defined even if d(death) = infinity
def theta(d): # d represents a persistence interval
    # If death is at infinity return maximal value from filtration value collection
    if d[1] == np.inf:
        return [d[0], epsilon_i]
    else:
        return d


# Returns birth, death, life span
def life(d):
    return [d[0], d[1], d[1] - d[0]]
 
# Well defined life span
def interval(d):
    return life(theta(d))
   
# Returns (only) the lifespan of persistance intervals given in a list of intervals
def persistence_interval(arr):
    result = [interval(d)[2] for d in arr]
    return result
 
max_dimension = 2
# List containing all persistence intervals
maximum_persistence_interval = []
# Corresponding death or bith of the interval. This choice has to be made by the user
corresponding_output = []

for i in range(0, max_dimension+1):
    # Algorithm 3
    arr = K.persistence_intervals_in_dimension(i)
    if len(arr) == 0:
        continue
    else:
        # Instead of taking the maximum we can also take the smallest value to the average (argmin to avrg) or a random interval
        max_value = max(persistence_interval(arr))
        maximum_persistence_interval.append(max_value)
 
        # Find the corresponding first output
        for lst, output in zip(arr, persistence_interval(arr)):
            if output == max_value:
                # For birth take index 0, for death take index 1
                corresponding_output.append(interval(lst)[1])
                break
 
# Returns max persistence interval for each dimension 0, 1, ...
print(maximum_persistence_interval)
# Returns corresponding time of death (our choice to use death time)
print(corresponding_output)

%md
#### Define the Labeling function
%md
The information contained in the simplex is stored in the format
<br>
([198], 0.0)
<br>
([199], 0.0)
<br>
([105, 165], 0.0024191509315663914)
<br>
([124, 179], 0.0091416205516085)
<br>
([145, 183], 0.013039522339199805)
<br>
The point [198] corresponds to the 198th point (x,y) in the dataframe constructed above. The first simplex to appear is [105, 165] and the number next to it represents its diameter.


  
rips_filtration = K.get_filtration() # The get_filtration() function allows to compute the list of simplices found in the Rips complex. The filtration value (float next to point) indicates the diameter of the simplex
rips_list = list(rips_filtration)
len(rips_list)


for splx in rips_list[280:320] :
   print(splx)

 

def check_list_in_tuples(list_of_tuples, L):
    for tup in list_of_tuples:
        if L == tup[0]:
            return True
    return False


"""
In the following we will work with sympy variables which are the easiest way to have the structure of an R-module
"""
#for each non zero label we define a variable (a generator). Here x represents the label 1 and y represents the label 2
x, y, z, w = symbols('x y z w') # For dataset with squares where there where only two classes, take only x and y


def Phi(sigma, K_list):
    if not check_list_in_tuples(K_list, sigma):
        return 0
    else:
        # sigma is a simplex given as a list as found in a simplicial complex when using the .get_filtration() method
        output = 0
        for S in sigma:
            point = points[S]
            # Find the row where the values match
            result = data[(data['x'] == point[0]) & (data['y'] == point[1])]
            label = result['label']
            if not label.empty and label.iloc[0] == 0:
                output += 0
            elif not label.empty and label.iloc[0] == 1:
                output += x
            elif not label.empty and label.iloc[0] == 2:
                output += y
            ###
            elif not label.empty and label.iloc[0] == 3:
                output += z
            else:
                output += w
            ###
    return output


# First we need to compute St_K(sigma)
def St_K(sigma, K_list):
    """
    sigma is again a simplex given as a list
    K_list is the simplicial complex in list format after using .get_filtration
    result is a list of simplexes which are co-faces to sigma
    """
    result = []
    for sublist, number in K_list:
        if all(num in sublist for num in sigma):
            result.append(sublist)
    return result


# We compute Lk_K(sigma) using Lemma 1
def Lk_K(sigma, K_list):
    """
    Same input as above
    result is a list of simplices which have no edge in common with sigma
    """
    simplex_list = St_K(sigma, K_list)
    result = []
    for sublist in simplex_list:
        filtered_sublist = [num for num in sublist if num not in sigma]
        result.append(filtered_sublist)
    result = [sublist for sublist in result if sublist]
    return result


# Funtion to access the epsilon_i of a specific simplex
def find_number_by_list(list_of_tuples, given_list):
    for sublist, number in list_of_tuples:
        if sublist == given_list:
            #print("The epsilon_i of " + str(sublist) + " is " + str(number))
            return number
    return None  # Return None if the given list is not found in any tuple


# From here note that v is a simplex with only one vertex, i.e. of dimension 0
def Psi(v, K_list):
    Lk_K_v = Lk_K(v, K_list)
    #print(Lk_K_v)
    result = 0
    for sigma in Lk_K_v:
        #print(sigma)
        # Access Xi_K(sigma) using the set Xi_K. This is the smallest radius epsilon_i such that the sigma appears in K_i
        Xi_K_sigma = find_number_by_list(Xi_K, sigma)
        #print("Xi_K_sigma = " + str(Xi_K_sigma))
        if Xi_K_sigma is not None:  # Check if Xi_K_sigma is not None
            #print("Phi applied to a neighbour is " + str(Phi(sigma, K_list)))
            result += (1/(Xi_K_sigma)**2)*Phi(sigma, K_list)
    return result


%md
#### Label the test set X

data['Simplex'] = data.index

# Keep rows with -1 in the 'label' column
data_filtered = data[data['label'] == 0]

# Drop the 'label' column
X = data_filtered.drop(columns=['label'])


%md
Algorithm 4
#max_element = max(maximum_persistence_interval)
#max_index = maximum_persistence_interval.index(max_element)
#epsilon_i = corresponding_output[max_index]
# Remark: one can find the right value for epsilon using the persistence diagram, but this is difficult. It is probably easier to try different values (for example in a matrix where each row contains a specific order of epsilon for one algorithm run).

epsilon_i = 0.3

# Since epsilon is contsant for each i, the function psi is easy to write down

def psi_F(epsilon_i):
    skeleton = gd.RipsComplex(
    #distance_matrix = dist_matrix,
    points = points,
    max_edge_length = epsilon_i
    )
    K_i = skeleton.create_simplex_tree(max_dimension = 2)
    return K_i

K_i = psi_F(epsilon_i)

rips_filtration_K_i = K_i.get_filtration() # The get_filtration() function allows to compute the list of simplices found in the Rips complex. The filtration value (float next to point) indicates the diameter of the simplex
rips_list_K_i = list(rips_filtration_K_i)

#len(rips_list_K_i)
#for splx in rips_list_K_i[0:400] :
#   print(splx)


def labelling(expr):
    if expr == 0:
        return 0
    # Extract symbols from the expression
    symbols_list = list(expr.free_symbols)
  
    # Initialize variables to store maximum coefficient and corresponding symbol
    max_coefficient = 0
    symbol_with_max_coefficient = None

    # Iterate through symbols and find the one with the largest coefficient
    for symbol in symbols_list:
        coefficient = expr.coeff(symbol)
        if coefficient > max_coefficient:
            max_coefficient = coefficient
            symbol_with_max_coefficient = symbol

    return symbol_with_max_coefficient

random_rows = X #X.sample(n=50)
# Create a dictionary to store results
results_dict = {}

# Iterate through selected rows
for index, row in random_rows.iterrows():
    simplex_number = row['Simplex']
    simplex = [simplex_number]
    result = Psi(simplex, rips_list_K_i)
    result = labelling(result)
    results_dict[simplex_number] = result


for simplex, label in results_dict.items():
    if label == x:
        data["label"].loc[simplex] = 10
    elif label == y:
        data["label"].loc[simplex] = 11
    elif label == z:
        data["label"].loc[simplex] = 12
    elif label == w:
        data["label"].loc[simplex] = 13
    else:
        data["label"].loc[simplex] = 0


# Define colors based on labels
colors2 = {1: 'green', 2: 'red', 3: 'black', 4: 'orange', 0: 'blue', 10: 'lightgreen', 11: 'pink', 12: 'grey', 13: 'yellow'}
 
# Plotting
fig2, ax2 = plt.subplots()

for label, group in data.groupby('label'):
    ax2.scatter(group['x'], group['y'], c=colors2[label], label=label)
ax2.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot with colored labels')
plt.show()




%md

#### Apply the algorithm a multiple times

epsilon_list = [0.2, 0.3, 0.3, 0.3, 0.4, 0.3, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1]

# Assume that the valued found by the algorithm are correct
for i in range(0,13):
    # Dictionary of replacements
    replacements = {10: 1, 11: 2, 12: 3, 13: 4}
    data['label'] = data['label'].replace(replacements)
    # Keep rows with -1 in the 'label' column
    data_filtered2 = data[data['label'] == 0]
      # Drop the 'label' column

    X2 = data_filtered2.drop(columns=['label'])
    epsilon_j = epsilon_list[i]
    K_j = psi_F(epsilon_j)
    rips_filtration_K_j = K_j.get_filtration()
    rips_list_K_j = list(rips_filtration_K_j)

    random_rows2 = X2 #X.sample(n=50)
    # Create a dictionary to store results
    results_dict2 = {}

    # Iterate through selected rows
    for index, row in random_rows2.iterrows():
        simplex_number = row['Simplex']
        simplex = [simplex_number]
        result = Psi(simplex, rips_list_K_j)
        result = labelling(result)
        results_dict2[simplex_number] = result


    for simplex, label in results_dict2.items():
      if label == x:
          data["label"].loc[simplex] = 10
      elif label == y:
          data["label"].loc[simplex] = 11
      elif label == z:
          data["label"].loc[simplex] = 12
      elif label == w:
          data["label"].loc[simplex] = 13
      else:
          data["label"].loc[simplex] = 0

# Dictionary of replacements

replacements = {10: 1, 11: 2, 12: 3, 13: 4}
data['label'] = data['label'].replace(replacements)

# Plotting
fig4, ax4 = plt.subplots()
for label, group in data.groupby('label'):
    ax4.scatter(group['x'], group['y'], c=colors2[label], label=label)
ax4.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot with colored labels')
plt.show()

# Count occurrences of each integer in 'label'
counts = data['label'].value_counts()
print(counts)
