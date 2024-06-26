import gudhi as gd
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

"""### Create dataset"""

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

# Create pandas dataframes
data1 = pd.DataFrame(ring1_points, columns=['x', 'y'])
data2 = pd.DataFrame(ring2_points, columns=['x', 'y'])
data3 = pd.DataFrame(ring3_points, columns=['x', 'y'])
data4 = pd.DataFrame({'x': x_coords, 'y': y_coords})

"""We have 5 labels in this example: 0 is for the unknown class and then we have labels 1-4 which the algorithm will have to predict."""

# Assign labels
data1['label'] = 0
data2['label'] = 0
data3['label'] = 0
data4['label'] = 0

# Randomly select x points and assign labels
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

# Extract x and y coordinates as a numpy array
points = data[['x', 'y']].values

num_of_it = 3

label_df = pd.concat([data["label"]] * num_of_it, axis=1)
iteration_df = pd.concat([data, label_df], axis=1)

num_label_columns = label_df.shape[1]
label_column_names = [f"label_{i+1}" for i in range(num_label_columns)]
iteration_df.columns = list(data.columns) + label_column_names

display(iteration_df)

"""### Create class for the TDA algorithm

Remark: We need to pay attention that the dimension of the data (in our case 2) is much smaller than the number of points of our dataset (303). The set P is given by all points, the set S is given by the points which have a label 1, 2, or 4, and the set X is given by the points with Label 0.
"""

"""
The following class allows to construct a filtered simplicial complex K, with initial epsilon given by epsilon_0 and increment size epsilon_i.
"""

class FilteredSimplicialComplex:
    def __init__(self, epsilon_0, epsilon_increment, max_filtration_level, max_dimension_of_complex, points):
        self.epsilon_i = epsilon_0
        self.epsilon_increment = epsilon_increment
        self.epsilon_list = [] #[self.epsilon_i]
        self.max_filtration_level = max_filtration_level
        self.max_dimension_of_complex = max_dimension_of_complex
        self.points = points # type numpy.ndarray

        # K represents the filtered simplicial complex. We initialise it with an empty complex
        self.K = gd.SimplexTree()

        # This dictionnary will contain the individual simplicial complexes that will compose K
        self.Simplicial_complexes = {}

        #The following list of tuppels will give the filtration value (see definition) for each simplex
        self.Filtration_value = []

        # For each epsilon_i = epsilon_0 + i*epsilon we create a simplicial complex K_i and add it to the filtered simplicial complex K
        for i in range(self.max_filtration_level):
            # Build simplicial Rips complex
            skeleton = gd.RipsComplex(
                points = self.points,
                max_edge_length = self.epsilon_i
            )
            K_i = skeleton.create_simplex_tree(self.max_dimension_of_complex)
            #print("Iteration: ", i, ",", "epsilon_i = ", self.epsilon_i, ",", "K_i != K: ", K_i != self.K)

            if K_i != self.K:
                # Join K_i to K
                rips_filtration = K_i.get_filtration()
                rips_list = list(rips_filtration)
                # Add the Komplex to the dictionnary
                Simplicial_complex = f"K_{i}"
                self.Simplicial_complexes[Simplicial_complex] = rips_list

                for splx in rips_list :
                    # splx[0] represent the coordinates of the simplex and splx[1] is the biggest distance between two points
                    self.K.insert(splx[0], splx[1])
                    self.Filtration_value.append((splx[0], self.epsilon_i))

            else:
                # Add the complex at step i and use the one calculated at step i-1
                j = i-1
                Simplicial_complex = f"K_{i}"
                self.Simplicial_complexes[Simplicial_complex] = self.Simplicial_complexes[f"K_{j}"]

            self.epsilon_list.append(self.epsilon_i)
            self.epsilon_i = self.epsilon_i + self.epsilon_increment

        # Now we assign to each simplex the smallest value epsilon such that the simplex is present in the simplicial complex K_i
        self.min_x_values = {}

        # Iterate through the list of tuples
        for sublist, x in self.Filtration_value:
            sublist_tuple = tuple(sublist)  # Convert the list to a tuple
            if sublist_tuple in self.min_x_values:
                # If sublist already exists in self.min_x_values, update the minimum x value if necessary
                if x < self.min_x_values[sublist_tuple]:
                    self.min_x_values[sublist_tuple] = x
            else:
                # If sublist is not in self.min_x_values, add it with its corresponding x value
                self.min_x_values[sublist_tuple] = x

        # Filter the original list based on the minimum x values
        self.Xi_K = [(sublist, x) for sublist, x in self.Filtration_value if x == self.min_x_values[tuple(sublist)]]

    def get_dimension(self):
        return self.K.dimension()

    def get_num_vertices(self):
        return self.K.num_vertices()

    def get_num_simplices(self):
        return self.K.num_simplices()

    def psi_F(self, epsilon_i):
        # We assume epsilon_i is contained in epsilon_list. We then access directly the simplicial complex K_i in the dictionnary above instead of creating
        # a new one.
        i = self.epsilon_list.index(epsilon_i)
        simplicial_complex = f"K_{i}"
        K_i_list = self.Simplicial_complexes[simplicial_complex]
        return K_i_list

epsilon = 0.1
epsilon_0 = 0.00001
max_filtration_level = 10 #11
max_dimension_of_complex = 100

K_class = FilteredSimplicialComplex(epsilon_0, epsilon, max_filtration_level, max_dimension_of_complex, points)

# Checks if simplex is in simplicial complex
def check_list_in_tuples(list_of_tuples, L):
    for tup in list_of_tuples:
        if L == tup[0]:
            return True
    return False

# Check if array is in dataframe
def find_row_with_array(np_array, df):
    # Convert the NumPy array to a Pandas Series for comparison
    np_series = pd.Series(np_array)
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if all elements of the NumPy array are present in the row
        if np_series.isin(row).all():
            # If found, return the row as a DataFrame object
            return pd.DataFrame(row).transpose()

    # If the array is not found in any row, return None
    return None

# All labels except the unknown one
label_list = data[data['label'] != 0]['label'].unique().tolist()

def create_symbols(n):
    """
    Generate n sympy symbols with names s_i.

    Parameters:
    n (int): Number of symbols to generate.

    Returns:
    symbols (list): List of generated sympy symbols.
    """
    symbols = [sp.symbols('s{}'.format(i)) for i in range(1, n + 1)]
    return symbols


def Phi(sigma, K_list, df, j, num_symbols):
    # sigma is a simplex given as a list as found in a simplicial complex when using the .get_filtration() method
    # K_list is the list of simplices contained in the simplicial complex K_i
    # df is the datarframe with the labels and the vectors

    if not check_list_in_tuples(K_list, sigma):
        return 0
    else:
        output = 0
        symbols = create_symbols(num_symbols)  # Create symbols dynamically
        label_symbol_map = {}  # Dictionary to map labels to symbols

        # Map labels to symbols dynamically
        for i in range(num_symbols):
            label = label_list[i]
            label_symbol_map[label] = symbols[i]

        # In the following S represents the vertices of sigma
        for S in sigma:
            # Extract coordinates of simplex S
            point = points[S]
            # Find the row where the values match
            result = find_row_with_array(point, df)
            # Get label of Simplex S
            label = result[f'label_{j}']

            # Important remark: The class 0 is represented by the string '0' and the other classes are represented by integers
            if not label.empty and label.iloc[0] == 0:
                output += 0
            elif not label.empty and label.iloc[0] in label_symbol_map:
                output += label_symbol_map[label.iloc[0]]
            else:
                output += symbols[-1]  # Default symbol u

    return output


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

def Lk_K_approx(sigma, K_list):
    """
    Same input as above

    result an approximation of Lk_K for faster computations
    """
    simplex_list = St_K(sigma, K_list)
    result = []
    for sublist in simplex_list:
        filtered_sublist = [num for num in sublist if num not in sigma]
        result.append(filtered_sublist)
        limit = 40
        if len(result) > limit:
            break

    result = [sublist for sublist in result if sublist]
    return result

# Funtion to access the epsilon_i of a specific simplex using the list Xi_K defined above
def find_number_by_list(list_of_tuples, given_list):
    for sublist, number in list_of_tuples:
        if sublist == given_list:
            return number
    return None  # Return None if the given list is not found in any tuple


def Psi(v, K_list, df, j, num_symbols):
    # From here note that v is a simplex with only one vertex, i.e. of dimension 0
    Lk_K_v = Lk_K_approx(v, K_list) # Lk_K
    result = 0
    #print(len(Lk_K_v))
    for sigma in Lk_K_v:
        # Access Xi_K(sigma) using the set Xi_K and function find_number_by_list. This is the smallest radius epsilon_i such that the sigma appears in K_i
        Xi_K_sigma = find_number_by_list(K_class.Xi_K, sigma)
        if Xi_K_sigma is not None:  # Check if Xi_K_sigma is not None
            result += (1/(Xi_K_sigma)**2)*Phi(sigma, K_list, df, j, num_symbols) # Take the square to put more weight on vectors near
    return result

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

# Get the number of labels
num_symbols = len(label_list)
generators = create_symbols(num_symbols)

length = 10
epsilons = []
for i in range(num_of_it):
    epsilons.append(random.choices(K_class.epsilon_list, k=length))

print(epsilons)

for j in range(num_of_it):
    print("#######################################################################")
    for i in range(0, length):
        # Keep rows with the unknown labelled points
        data_filtered = iteration_df[iteration_df[f"label_{j+1}"] == 0]

        # Drop the "label_{i+1}" column
        X = data_filtered[['x', 'y']] #data_filtered.drop(columns=[f"label_{i+1}"])

        if data_filtered.empty:
            break

        sub_list = epsilons[j]

        # psi_F already outputs lists
        rips_list_K_i = K_class.psi_F(sub_list[i])

        # Create a dictionary to store results
        results_dict = {}

        # Iterate through selected rows and apply the algorithm
        for index, row in X.iterrows():
            # simplex_number = row['Simplex']
            simplex = [index]
            result = Psi(simplex, rips_list_K_i, iteration_df, j+1, num_symbols)
            #print(result)
            result = labelling(result)
            results_dict[index] = result


        # Add results to the dataframe
        for simplex, label in results_dict.items():
            if label in generators:
                symbol_index = generators.index(label)
                iteration_df[f"label_{j+1}"].loc[simplex] = label_list[symbol_index]  # labels_list contains labels corresponding to symbols
            else:
                iteration_df[f"label_{j+1}"].loc[simplex] = 0

display(iteration_df)

# Define a function to find the most common value in a row
def most_common(row):
    counts = row.value_counts()
    max_count = counts.max()
    most_common_values = counts[counts == max_count].index.tolist()
    return np.random.choice(most_common_values)

# Apply the function row-wise to fill the missing values
iteration_df['label'] = iteration_df.apply(lambda row: most_common(row.dropna()), axis=1)

final_df = iteration_df[['x', 'y', 'label']]

# Plotting
fig4, ax4 = plt.subplots()
for label, group in final_df.groupby('label'):
    ax4.scatter(group['x'], group['y'], c=colors[label], label=label)
ax4.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot with colored labels')
plt.show()

# Count occurrences of each integer in 'label'
counts = final_df['label'].value_counts()
print(counts)
