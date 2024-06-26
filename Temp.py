
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


#LETS DEFINE AN INTEGRATOR
#ver la que al ppio aumentaba 0.5

# Leapfrog explicit integration to calculate strain from strain rates and stresses
def leapfrog_integration(strain_rate, stresses, dt):
    strain = np.zeros_like(strain_rate)
    strain_prev = np.zeros_like(strain_rate)
    strain_next = np.zeros_like(strain_rate)
    num_steps = 10  # Number of integration steps
    for i in range(num_steps):
        strain_next = strain_prev + 0.5 * dt * (3 * strain_rate - strain_prev)
        stresses_next = np.dot(material_matrix(), strain_next)
        strain = strain_prev + 0.5 * dt * (stresses + stresses_next)
        strain_prev = strain_next
        stresses = stresses_next
    return strain


# 2 elements
# asssembly Final code



# import numpy as np

# # Define material properties
# E = 200e9  # Young's modulus in Pa
# nu = 0.3   # Poisson's ratio
# yield_stress = 250e6  # Yield stress in Pa
# H = 5e9  # Hardening modulus in Pa

# # Define element properties
# num_nodes_element = 4  # Number of nodes per element
# element_length = 1.0   # Length of the element

# # Define shape functions and their derivatives for 2D quadrilateral element
# def shape_functions(xi, eta):
    # N = np.array([(1-xi)*(1-eta)/4,
                  # (1+xi)*(1-eta)/4,
                  # (1+xi)*(1+eta)/4,
                  # (1-xi)*(1+eta)/4])
    # dN_dxi = np.array([-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4])
    # dN_deta = np.array([-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4])
    # return N, dN_dxi, dN_deta

# # Define material matrix for plane stress
# def material_matrix():
    # C = E / (1 - nu**2) * np.array([[1, nu, 0],
                                     # [nu, 1, 0],
                                     # [0, 0, (1 - nu) / 2]])
    # return C

# # Gauss quadrature points and weights
# gauss_points = np.array([[-0.577350269, -0.577350269],
                         # [ 0.577350269, -0.577350269],
                         # [ 0.577350269,  0.577350269],
                         # [-0.577350269,  0.577350269]])
# gauss_weights = np.array([1, 1, 1, 1])

# # J2 plasticity function
# def J2_plasticity(sigma, delta_gamma, dt):
    # # Von Mises stress
    # s_eq = np.sqrt(3 / 2 * np.sum(sigma ** 2))
    
    # # Check if yielding occurs
    # if s_eq > yield_stress:
        # # Plastic multiplier
        # dlambda = (s_eq - yield_stress) / (3 * H + 0.0001)
        # # Plastic strain increment
        # delta_epsilon_p = dlambda * np.sign(sigma) / (1 + 3 * H * dt)
        # # Update plastic strain
        # epsilon_p = epsilon_p_prev + delta_epsilon_p
        # # Update stress
        # sigma -= 2 * H * dlambda * np.sign(sigma)
    # else:
        # # No plastic strain increment
        # delta_epsilon_p = np.zeros_like(sigma)
    
    # return sigma, delta_epsilon_p

# # Finite element strain rate calculation
# def calculate_strain_rate(velocities):
    # strain_rate = np.zeros((3, 3))
    # for gp in range(len(gauss_points)):
        # xi, eta = gauss_points[gp]
        # weight = gauss_weights[gp]
        # N, dN_dxi, dN_deta = shape_functions(xi, eta)
        # J = np.dot(np.array([dN_dxi, dN_deta]).T, velocities)
        # detJ = np.linalg.det(J)
        # invJ = np.linalg.inv(J)
        # B = np.zeros((3, 8))
        # for i in range(num_nodes_element):
            # B[0, 2*i] = dN_dxi[i]
            # B[1, 2*i+1] = dN_deta[i]
            # B[2, 2*i] = dN_deta[i]
            # B[2, 2*i+1] = dN_dxi[i]
        # strain_rate += np.dot(B.T, np.dot(invJ.T, np.dot(B, velocities))) * detJ * weight
    # strain_rate *= 0.5
    # return strain_rate

# # Leapfrog explicit integration to calculate strain from strain rates and stresses
# def leapfrog_integration(strain_rate, stresses, dt):
    # strain = np.zeros_like(strain_rate)
    # strain_prev = np.zeros_like(strain_rate)
    # strain_next = np.zeros_like(strain_rate)
    # num_steps = 10  # Number of integration steps
    # for i in range(num_steps):
        # strain_next = strain_prev + 0.5 * dt * (3 * strain_rate - strain_prev)
        # stresses_next = np.dot(material_matrix(), strain_next)
        # stresses_next, _ = J2_plasticity(stresses_next, dt)
        # strain = strain_prev + 0.5 * dt * (stresses + stresses_next)
        # strain_prev = strain_next
        # stresses = stresses_next
    # return strain

# # Calculate element forces from stresses
# def calculate_element_forces(stresses):
    # element_forces = np.dot(np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]), stresses)
    # return element_forces

# # Calculate nodal forces from element forces
# def assemble_nodal_forces(element_forces, element_nodes):
    # num_nodes = np.max(element_nodes) + 1
    # nodal_forces = np.zeros((num_nodes, 2))
    # for i, nodes in enumerate(element_nodes):
        # nodal_forces[nodes] += element_forces[i]
    # return nodal_forces

# # Calculate acceleration from nodal forces
# def calculate_acceleration(nodal_forces):
    # # Assuming constant mass for simplicity
    # mass = 1.0
    # acceleration = nodal_forces / mass
    # return acceleration

# # Calculate velocity from acceleration
# def calculate_velocity(acceleration, dt):
    # velocity = acceleration * dt
    # return velocity

# # Example usage with 2-element case
# element_nodes = [[0, 1, 2, 3], [2, 3, 4, 5]]  # Element connectivity
# velocities = np.array([[1, 0], [2, 0], [2, 1], [1, 1], [3, 1], [4, 1]])  # Example velocities at nodes
# strain_rate = calculate_strain_rate(velocities)
# stresses = np.array([[100e6, 50e6, 0], [50e6, 150e6, 0], [0, 0, 75e6], [100e6, 50e6, 0], [50e6, 150e6, 0], [0, 0, 75e6]])  # Example stresses
# dt = 0.01  # Time step
# epsilon_p_prev = np.zeros_like(stresses)  # Initial plastic strain
# strain = leapfrog_integration(strain_rate, stresses, dt)
# element_forces = calculate_element_forces(stresses)
# nodal_forces = assemble_nodal_forces(element_forces, element_nodes)
# acceleration = calculate_acceleration


