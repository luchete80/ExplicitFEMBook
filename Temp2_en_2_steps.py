import numpy as np

def quad_strain_calc(strain_rate, dt, stress, elastic_modulus, poissons_ratio, velocity, acceleration):
    # Define Gauss points and weights for quadrilateral element
    gauss_points = np.array([[-0.577350269189626, -0.577350269189626],
                             [0.577350269189626, -0.577350269189626],
                             [0.577350269189626, 0.577350269189626],
                             [-0.577350269189626, 0.577350269189626]])
    gauss_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Compute the Jacobian matrix for a quadrilateral element
    J = np.array([[-0.25, 0.25, 0.25, -0.25],
                  [-0.25, -0.25, 0.25, 0.25]])

    # Compute the strain tensor using the strain rate tensor and the time step
    strain = np.zeros((2, 2))
    for i in range(4):
        dNi_dx = J[0, i]
        dNi_dy = J[1, i]
        strain += np.outer(shape_functions[:, i], [dNi_dx, dNi_dy])

    strain *= dt

    # Compute the stress deviator (J2)
    dev_strain = strain - (np.trace(strain) / 2) * np.eye(2)
    J2 = np.sqrt(0.5 * np.sum(dev_strain ** 2))

    # Compute the Jaumann rate of stress
    stress += 2 * elastic_modulus * (1 + poissons_ratio) * strain
    stress_dev = stress - (np.trace(stress) / 2) * np.eye(2)
    dev_strain_rate = strain_rate - (np.trace(strain_rate) / 2) * np.eye(2)
    stress += dt * (2 * elastic_modulus * (1 + poissons_ratio) * dev_strain_rate - 2 * elastic_modulus * poissons_ratio * J2 * dev_strain)

    # Update velocity and acceleration
    velocity += acceleration * dt
    acceleration += (stress / density) * dt

    # Calculate element forces using Gauss quadrature
    element_force = np.zeros(2)
    for i in range(4):
        dNi_dx = J[0, i]
        dNi_dy = J[1, i]
        Ni = shape_functions[:, i]
        for j in range(4):
            x_gauss, y_gauss = gauss_points[j]
            weight = gauss_weights[j]
            detJ = dNi_dx * dNi_dy
            Nx = Ni[0] + x_gauss * dNi_dx
            Ny = Ni[1] + y_gauss * dNi_dy
            element_force += -np.dot(Nx, stress) * dNi_dx * detJ * weight
            element_force += -np.dot(Ny, stress) * dNi_dy * detJ * weight

    return strain, stress, velocity, acceleration, element_force

# Example usage:
strain_rate = np.array([[1.0, 0.5], [0.5, 2.0]])  # Example strain rate tensor
dt = 0.1  # Example time step
stress = np.array([[100.0, 50.0], [50.0, 80.0]])  # Initial stress tensor
elastic_modulus = 200.0  # Example elastic modulus
poissons_ratio = 0.3  # Example Poisson's ratio
density = 10.0  # Example material density
velocity = np.array([0.0, 0.0])  # Initial velocity
acceleration = np.array([0.0, 0.0])  # Initial acceleration

# Leapfrog integration
# Initial half step
strain_half, stress_half, velocity_half, acceleration_half, element_force_half = quad_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

# Full step
strain, stress, velocity, acceleration, element_force = quad_strain_calc(strain_rate, dt, stress_half, elastic_modulus, poissons_ratio, velocity_half, acceleration_half)

# Final half step
strain_final, stress_final, velocity_final, acceleration_final, element_force_final = quad_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

print("Strain tensor:")
print(strain)
print("Stress tensor:")
print(stress)
print("Velocity:")
print(velocity_final)
print("Acceleration:")
print(acceleration_final)
print("Element Force:")
print(element_force_final)




#If we want to change yo axisymm

def axisymmetric_strain_calc(strain_rate, dt, stress, elastic_modulus, poissons_ratio, velocity, acceleration):
    # Define Gauss points and weights for axisymmetric element
    gauss_points = np.array([[0.112701665379258, 0.500000000000000, 0.887298334620742],
                             [0.500000000000000, 0.887298334620742, 0.112701665379258],
                             [0.887298334620742, 0.112701665379258, 0.500000000000000],
                             [0.112701665379258, 0.887298334620742, 0.500000000000000]])
    gauss_weights = np.array([5/18, 5/18, 5/18, 5/18])

    # Compute the strain tensor using the strain rate tensor and the time step
    strain = strain_rate * dt
    
    # Compute the stress deviator (J2)
    dev_strain = strain - (np.trace(strain) / 3) * np.eye(2)
    J2 = np.sqrt(2 * np.sum(dev_strain ** 2))

    # Compute the Jaumann rate of stress
    stress += 2 * elastic_modulus * (1 + poissons_ratio) * strain
    
    # Update velocity and acceleration
    velocity += acceleration * dt
    acceleration += (stress / density) * dt

    # Calculate element forces using Gauss quadrature
    element_force = np.zeros(2)
    for i in range(4):
        r = gauss_points[i]
        weight = gauss_weights[i]
        element_force += np.dot(shape_function(r), stress) * weight

    return strain, stress, velocity, acceleration, element_force

def shape_function(r):
    # Define shape function for axisymmetric element
    return np.array([1 - r**2, 1 - r**2])

# Example usage:
strain_rate = np.array([[1.0, 0.5], [0.5, 2.0]])  # Example strain rate tensor
dt = 0.1  # Example time step
stress = np.array([[100.0, 50.0], [50.0, 80.0]])  # Initial stress tensor
elastic_modulus = 200.0  # Example elastic modulus
poissons_ratio = 0.3  # Example Poisson's ratio
density = 10.0  # Example material density
velocity = np.array([0.0, 0.0])  # Initial velocity
acceleration = np.array([0.0, 0.0])  # Initial acceleration

# Leapfrog integration
# Initial half step
strain_half, stress_half, velocity_half, acceleration_half, element_force_half = axisymmetric_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

# Full step
strain, stress, velocity, acceleration, element_force = axisymmetric_strain_calc(strain_rate, dt, stress_half, elastic_modulus, poissons_ratio, velocity_half, acceleration_half)

# Final half step
strain_final, stress_final, velocity_final, acceleration_final, element_force_final = axisymmetric_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

print("Strain tensor:")
print(strain)
print("Stress tensor:")
print(stress)
print("Velocity:")
print(velocity_final)
print("Acceleration:")
print(acceleration_final)
print("Element Force:")
print(element_force_final)
This code now computes the element forces using Gauss quadrature for axisymmetric elements. Replace the example values with your specific values for strain_rate, dt, stress, elastic_modulus, poissons_ratio, density, velocity, and acceleration.






Message ChatGPT…

ChatGPT can make mistakes. Consider checking important information.

    

def shape_function(r):
    # Define shape function for axisymmetric element
    return np.array([1 - r**2, 1 - r**2])

# Example usage:
strain_rate = np.array([[1.0, 0.5], [0.5, 2.0]])  # Example strain rate tensor
dt = 0.1  # Example time step
stress = np.array([[100.0, 50.0], [50.0, 80.0]])  # Initial stress tensor
elastic_modulus = 200.0  # Example elastic modulus
poissons_ratio = 0.3  # Example Poisson's ratio
density = 10.0  # Example material density
velocity = np.array([0.0, 0.0])  # Initial velocity
acceleration = np.array([0.0, 0.0])  # Initial acceleration

# Leapfrog integration
# Initial half step
strain_half, stress_half, velocity_half, acceleration_half, element_force_half = axisymmetric_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

# Full step
strain, stress, velocity, acceleration, element_force = axisymmetric_strain_calc(strain_rate, dt, stress_half, elastic_modulus, poissons_ratio, velocity_half, acceleration_half)

# Final half step
strain_final, stress_final, velocity_final, acceleration_final, element_force_final = axisymmetric_strain_calc(strain_rate, dt/2, stress, elastic_modulus, poissons_ratio, velocity, acceleration)

print("Strain tensor:")
print(strain)
print("Stress tensor:")
print(stress)
print("Velocity:")
print(velocity_final)
print("Acceleration:")
print(acceleration_final)
print("Element Force:")
print(element_force_final)


#This code now computes the element 
forces using Gauss quadrature for 
                           
axisymmetric elements. Replace the example values with your specific values 
for strain_rate, dt, stress, elastic_modulus, poissons_ratio, density, 
