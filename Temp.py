Lets begin defining  some material props

import numpy as np

# Define material properties
E = 200e9  # Young's modulus in Pa
nu = 0.3   # Poisson's ratio

# Define element properties
num_nodes_element = 4  # Number of nodes per element
element_length = 1.0   # Length of the element

# Define shape functions and their derivatives for 2D quadrilateral element
def shape_functions(xi, eta):
    N = np.array([(1-xi)*(1-eta)/4,
                  (1+xi)*(1-eta)/4,
                  (1+xi)*(1+eta)/4,
                  (1-xi)*(1+eta)/4])
    dN_dxi = np.array([-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4])
    dN_deta = np.array([-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4])
    return N, dN_dxi, dN_deta

# Define material matrix for plane stress
def material_matrix():
    C = E / (1 - nu**2) * np.array([[1, nu, 0],
                                     [nu, 1, 0],
                                     [0, 0, (1 - nu) / 2]])
    return C

# Gauss quadrature points and weights
gauss_points = np.array([[-0.577350269, -0.577350269],
                         [ 0.577350269, -0.577350269],
                         [ 0.577350269,  0.577350269],
                         [-0.577350269,  0.577350269]])
gauss_weights = np.array([1, 1, 1, 1])

# Finite element strain rate calculation
def calculate_strain_rate(velocities):
    strain_rate = np.zeros((3, 3))
    for gp in range(len(gauss_points)):
        xi, eta = gauss_points[gp]
        weight = gauss_weights[gp]
        N, dN_dxi, dN_deta = shape_functions(xi, eta)
        J = np.dot(np.array([dN_dxi, dN_deta]).T, velocities)
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        B = np.zeros((3, 8))
        for i in range(num_nodes_element):
            B[0, 2*i] = dN_dxi[i]
            B[1, 2*i+1] = dN_deta[i]
            B[2, 2*i] = dN_deta[i]
            B[2, 2*i+1] = dN_dxi[i]
        strain_rate += np.dot(B.T, np.dot(invJ.T, np.dot(B, velocities))) * detJ * weight
    strain_rate *= 0.5
    return strain_rate

# Example usage
velocities = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])  # Example velocities at nodes
strain_rate = calculate_strain_rate(velocities)
print("Strain rate:")
print(strain_rate)
