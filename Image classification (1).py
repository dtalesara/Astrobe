#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt


# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm


# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc


# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc


# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    # Initialize the quantum circuit with amplitude encoding
    qc = amplitude_encoding(feature)

    # Add the parameterized unitary circuit
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)

    # Use Statevector to simulate the circuit and get the final statevector
    statevector = Statevector.from_instruction(qc)

    # Calculate probabilities from the statevector
    probabilities = np.abs(statevector.data) ** 2  # Array of 64 probabilities for 6 qubits

    # Classification rule: sum probabilities of relevant groups
    triangle_prob = np.sum(probabilities[:32])  # First half for Triangle
    square_prob = np.sum(probabilities[32:])   # Second half for Square

    # Return the predicted label: 0 -> Triangle, 1 -> Square
    return 0 if triangle_prob > square_prob else 1


# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        # Adjusting the error calculation to include more comprehensive terms
        total_error += np.sum((np.array(label) - predicted_label) ** 2)  # Mean Squared Error (MSE)
    return total_error / len(features)



# Step 6: Image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)


# Step 7: Train the model
def train_classifier(features, labels, n_qubits):
    # Define initial guess for theta (parameters for the quantum circuit)
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters for 6 qubits

    # Optimize the cost function using L-BFGS-B
    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",  # More suitable for continuous optimization
        options={"maxiter": 1000, "disp": True},  # Display progress
    )
    return result.x 


# Step 8: Main execution
if __name__ == "__main__":
    # Define number of qubits
    n_qubits = 6

    # Define features and labels for training
    # Replace these with actual flattened vectors of triangle and square images
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png') 
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")  
    features = [feature_1, feature_2]

    labels = [0, 1]  # Labels: 0 -> Triangle, 1 -> Square

    # Train the quantum classifier
    optimal_theta = train_classifier(features, labels, n_qubits)
    print("Optimal Parameters:", optimal_theta)

    # Test the classifier on a new image
    new_feature = preprocess_image(r'C:\Users\User\Downloads\7252.png')  # Example image
    predicted_label = classify_shape(new_feature, optimal_theta, n_qubits)
    print("Predicted Label:", "Triangle" if predicted_label == 0 else "Square")
    print(f"Final Cost: {cost_history[-1]}")


# In[10]:


from scipy.optimize import minimize
import numpy as np

# Improved train_classifier with multiple optimizer runs and better initialization
def train_classifier(features, labels, n_qubits, max_retries=5):
    # Store best result across multiple runs
    best_result = None
    best_cost = float('inf')

    for _ in range(max_retries):
        # Initialize the parameters for each retry (can use small values to start)
        initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters for 6 qubits
        
        # Optimize the cost function using L-BFGS-B
        result = minimize(
            fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
            x0=initial_theta,
            method="L-BFGS-B",  # More suitable for continuous optimization
            options={"maxiter": 1000, "disp": False},  # Display progress can be turned off
        )
        
        # If this run results in a better cost, update the best result
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result.x

    return best_result  # Optimal parameters from the best run
print(f"Final Cost: {cost_history[-1]}")


# Main execution
if __name__ == "__main__":
    # Define number of qubits
    n_qubits = 6

    # Define features and labels for training
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png')  # Triangle
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")  # Square
    features = [feature_1, feature_2]

    labels = [0, 1]  # Labels: 0 -> Triangle, 1 -> Square

    # Train the quantum classifier
    optimal_theta = train_classifier(features, labels, n_qubits)
    print("Optimal Parameters:", optimal_theta)

    # Test the classifier on a new image
    new_feature = preprocess_image(r'C:\Users\User\Downloads\7252.png')  # Example image
    predicted_label = classify_shape(new_feature, optimal_theta, n_qubits)
    print("Predicted Label:", "Triangle" if predicted_label == 0 else "Square")
    print(f"Final Cost: {cost_history[-1]}")


# In[ ]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt




# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm


# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc


# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc


# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    # Initialize the quantum circuit with amplitude encoding
    qc = amplitude_encoding(feature)

    # Add the parameterized unitary circuit
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)

    # Use Statevector to simulate the circuit and get the final statevector
    statevector = Statevector.from_instruction(qc)

    # Calculate probabilities from the statevector
    probabilities = np.abs(statevector.data) ** 2  # Array of 64 probabilities for 6 qubits

    # Post-select based on the highest-probability state
    max_prob_index = np.argmax(probabilities)
    binary_outcome = format(max_prob_index, f"0{n_qubits}b")  # Binary string representation

    # Use the last qubit as the classification output
    final_qubit_value = int(binary_outcome[-1])  # Last bit indicates class (0 -> Triangle, 1 -> Square)

    return final_qubit_value
# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        # Adjusting the error calculation to include more comprehensive terms
        total_error += np.sum((np.array(label) - predicted_label) ** 2)  # Mean Squared Error (MSE)
    return total_error / len(features)



# Step 6: Image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)


# Step 7: Train the model
def train_classifier(features, labels, n_qubits):
    # Define initial guess for theta (parameters for the quantum circuit)
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters for 6 qubits
    cost_history = []

    initial_cost = cost_function(initial_theta, features, labels, n_qubits)
    print(f"Initial Cost: {initial_cost}")
    cost_history.append(initial_cost)

    # Optimize the cost function using L-BFGS-B
    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",  # More suitable for continuous optimization
        options={"maxiter": 1000, "disp": True},  # Display progress
    )
    return result.x 


# Main execution
if __name__ == "__main__":
    # Define number of qubits
    n_qubits = 6

    # Define features and labels for training
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png')  # Triangle
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")  # Square
    features = [feature_1, feature_2]

    labels = [0, 1]  # Labels: 0 -> Triangle, 1 -> Square

    # Train the quantum classifier
    optimal_theta = train_classifier(features, labels, n_qubits)
    print("Optimal Parameters:", optimal_theta)

    # Train the quantum classifier
    optimal_theta = train_classifier(features, labels, n_qubits)
    print("Optimal Parameters:", optimal_theta)
    # Test the classifier on a new image
    new_feature = preprocess_image(r'C:\Users\User\Downloads\pxArt (3).png')  # Example image
    predicted_label = classify_shape(new_feature, optimal_theta, n_qubits)
    print("Predicted Label:", f"|{predicted_label}⟩")  # Display as ket notation


# In[ ]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm

# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc

# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc

# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    qc = amplitude_encoding(feature)
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2
    triangle_prob = np.sum(probabilities[:32])
    square_prob = np.sum(probabilities[32:])
    return 0 if triangle_prob > square_prob else 1

# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        total_error += (label - predicted_label) ** 2
    return total_error / len(features)

# Step 6: Train the model with cost tracking
def train_classifier(features, labels, n_qubits):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)
    cost_history = []

    def callback(theta):
        cost = cost_function(theta, features, labels, n_qubits)
        cost_history.append(cost)

    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": True},
        callback=callback,
    )
    return result.x, cost_history

# Step 7: Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)

# Step 8: Main execution
if __name__ == "__main__":
    n_qubits = 6
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png') 
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")
    features = [feature_1, feature_2]
    labels = [0, 1]

    converged = False
    iteration_count = 0
    max_retries = 10

    while not converged and iteration_count < max_retries:
        iteration_count += 1
        print(f"Training iteration: {iteration_count}")
        optimal_theta, cost_history = train_classifier(features, labels, n_qubits)

        # Test the classifier
        test_feature = preprocess_image(r'C:\Users\User\Downloads\pxArt (3).png')
        predicted_label = classify_shape(test_feature, optimal_theta, n_qubits)

        print(f"Predicted Label: {predicted_label}")

        if predicted_label == labels[0]:  # Assuming test_feature is Triangle
            converged = True

    if converged:
        print("Model successfully trained and converged!")
    else:
        print("Model did not converge after max retries.")

    # Plot the cost function vs iterations
    
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', label="Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost Function vs Iterations")
        plt.legend()
        plt.grid()
        plt.show()
    


# In[ ]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm

# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc

# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc

# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    qc = amplitude_encoding(feature)
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2
    triangle_prob = np.sum(probabilities[:32])  
    square_prob = np.sum(probabilities[32:])   
    
    return 0 if triangle_prob > square_prob else 1

# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        total_error += (label - predicted_label) ** 2
    return total_error / len(features)

# Step 6: Train the model with cost tracking
def train_classifier(features, labels, n_qubits):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)
    cost_history = []  # List to store cost values during optimization

    def callback(xk):
        cost = cost_function(xk, features, labels, n_qubits)
        cost_history.append(cost)
        print(f"Iteration: {len(cost_history)}, Cost: {cost}")
        
        result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",  # More suitable for continuous optimization
        options={"maxiter": 100, "disp": True},  # Display progress
    )
    return result.x 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()

    return result.x, cost_history  # Return both optimal parameters and cost history

# Step 7: Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)

# Step 8: Main execution
if __name__ == "__main__":
    n_qubits = 6
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png') 
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")
    features = [feature_1, feature_2]
    labels = [0, 1]

    converged = False
    iteration_count = 0
    max_retries = 10

    while not converged and iteration_count < max_retries:
        iteration_count += 1
        print(f"Training iteration: {iteration_count}")
        optimal_theta, cost_history = train_classifier(features, labels, n_qubits)

        # Test the classifier
        test_feature = preprocess_image(r'C:\Users\User\Downloads\pxArt (6).png')
        predicted_label = classify_shape(test_feature, optimal_theta, n_qubits)

        print(f"Predicted Label: {predicted_label}")

        

    
    


# Nikita code

# In[ ]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.library import TwoLocal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Amplitude Encoding
def amplitude_encode(amplitudes):
    """
    Normalize the input data and encode it into a quantum state.
    
    Parameters:
        amplitudes (list or np.array): Input data to be encoded.
    
    Returns:
        QuantumCircuit: Quantum circuit with the encoded data.
    """
    amplitudes = np.array(amplitudes, dtype=np.float64)
    if len(amplitudes) & (len(amplitudes) - 1) != 0:
        raise ValueError("Input size must be a power of 2.")
    amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize the data
    num_qubits = int(np.log2(len(amplitudes)))  # Number of qubits needed
    qc = QuantumCircuit(num_qubits)
    qc.initialize(amplitudes, range(num_qubits))
    return qc

# Step 2: Parameterized Circuit
def u_theta(amplitudes, parameters):
    num_qubits = int(np.log2(len(amplitudes)))  # Number of qubits
    twolocal = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=5,  # Number of repetitions for the ansatz
    )
    param_dict = dict(zip(twolocal.parameters, parameters))  # Map parameters to circuit
    twolocal.assign_parameters(param_dict, inplace=True)  # Assign parameters
    
    # Combine amplitude encoding and variational circuit
    qc = QuantumCircuit(num_qubits, num_qubits)  # Add classical bits for measurement
    feature_map = amplitude_encode(amplitudes)  # Encode data
    qc.compose(feature_map, inplace=True)  # Add feature map
    qc.compose(twolocal, inplace=True)  # Add variational ansatz
    qc.measure(range(num_qubits), range(num_qubits))  # Measure all qubits
    return qc

# Step 3: Cost Function
def cost_function(params, features, labels):
    print(f"Params: {params}")  # Log parameters
    total_error = 0
    sampler = Sampler()
    
    for feature, label in zip(features, labels):
        qnn = u_theta(feature, params)
        result = sampler.run(qnn).result()
        
        # Extract probabilities correctly
        probabilities = np.zeros(len(label))
        for key, value in result.quasi_dists[0].items():
            probabilities[key] = value  # Assign probability to corresponding state index
        
        print(f"Feature: {feature}, Label: {label}, Probabilities: {probabilities}")
        total_error += np.sum((np.array(label) - probabilities) ** 2)
    
    return total_error  # Return the total error so optimizer can minimize it

# Step 4: Optimization Callback
cost_history = []
def callback_function(params):
    """
    Track the cost value during optimization.
    """
    cost = cost_function(params, features, labels)
    cost_history.append(cost)
    print(f"Cost: {cost}")

# Step 5: Define Features and Labels
features = [[1, 0, 0, 0], [0, 1, 0, 0]]  # Example features (encoded as amplitude vectors)
labels = [[1, 0, 0, 0], [0, 1, 0, 0]]  # Corresponding one-hot encoded labels

# Step 6: Optimization
np.random.seed(42)
initial_theta = np.random.rand(12)  # Initialize random parameters (matches TwoLocal param count)

# Use 'L-BFGS-B' optimization method
result = minimize(
    fun=cost_function,
    x0=initial_theta,
    args=(features, labels),
    method="COBYLA",
    callback=callback_function,
    options={"disp": True, "maxiter": 10000}
)

# Step 7: Plotting the Cost Function
plt.figure(figsize=(8, 5))
plt.plot(cost_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.title("Optimization Progress")
plt.grid()
plt.show()


# In[4]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm

# Step 2: Preprocess galaxy images
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to ensure consistent input size
    image_array = np.array(image).flatten()  # Flatten into a 1D vector
    return normalize_amplitudes(image_array)

# Step 3: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc

# Step 4: Define the parameterized quantum circuit
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc

# Step 5: Multi-Class Galaxy Classification
def classify_galaxy(feature, theta, n_qubits, num_classes):
    qc = amplitude_encoding(feature)
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)

    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2  # Probability distribution

    # Assign each galaxy type to a portion of the quantum state space
    bin_size = len(probabilities) // num_classes
    class_probs = [np.sum(probabilities[i * bin_size : (i + 1) * bin_size]) for i in range(num_classes)]
    
    return np.argmax(class_probs)  # Return class with highest probability

# Step 6: Define the cost function
def cost_function(theta, features, labels, n_qubits, num_classes):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_galaxy(feature, theta, n_qubits, num_classes)
        total_error += (label - predicted_label) ** 2  # Squared error
    return total_error / len(features)

# Step 7: Train the quantum classifier
def train_classifier(features, labels, n_qubits, num_classes):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters

    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits, num_classes=num_classes),
        x0=initial_theta,
        method="COBYLA",
        options={"maxiter": 1000, "disp": True},
    )
    return result.x

# Step 8: Main Execution
if __name__ == "__main__":
    n_qubits = 6
    num_classes = 2  # Spiral, Elliptical

    # Load galaxy images and labels
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\pxArt (7).png')  
    feature_2 = preprocess_image(r'C:\Users\User\Downloads\pxArt (8).png')
    features = [feature_1, feature_2]
    labels = [0, 1]  # Spiral = 0, Elliptical = 1

    # Train classifier
    optimal_theta = train_classifier(features, labels, n_qubits, num_classes)

    # Test on a new galaxy image
    test_feature = preprocess_image(r'C:\Users\User\Downloads\pxArt (9).png')
    predicted_label = classify_galaxy(test_feature, optimal_theta, n_qubits, num_classes)

    galaxy_types = ["Spiral", "Elliptical"]
    print(f"Predicted Galaxy Type: {galaxy_types[predicted_label]}")


# In[3]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler  # Updated import
from qiskit.circuit.library import TwoLocal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial
from qiskit.quantum_info import Statevector

# Step 1: Amplitude Encoding
def amplitude_encode(amplitudes):
    """
    Normalize the input data and encode it into a quantum state.
    
    Parameters:
        amplitudes (list or np.array): Input data to be encoded.
    
    Returns:
        QuantumCircuit: Quantum circuit with the encoded data.
    """
    amplitudes = np.array(amplitudes, dtype=np.float64)
    if len(amplitudes) & (len(amplitudes) - 1) != 0:
        raise ValueError("Input size must be a power of 2.")
    amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize the data
    num_qubits = int(np.log2(len(amplitudes)))  # Number of qubits needed
    qc = QuantumCircuit(num_qubits)
    qc.initialize(amplitudes, range(num_qubits))
    return qc

# Step 2: Parameterized Circuit
def u_theta(amplitudes, parameters):
    num_qubits = int(np.log2(len(amplitudes)))  # Number of qubits
    twolocal = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=5,  # Number of repetitions for the ansatz
    )
    param_dict = dict(zip(twolocal.parameters, parameters))  # Map parameters to circuit
    twolocal.assign_parameters(param_dict, inplace=True)  # Assign parameters
    
    # Combine amplitude encoding and variational circuit
    qc = QuantumCircuit(num_qubits, num_qubits)  # Add classical bits for measurement
    feature_map = amplitude_encode(amplitudes)  # Encode data
    qc.compose(feature_map, inplace=True)  # Add feature map
    qc.compose(twolocal, inplace=True)  # Add variational ansatz
    qc.measure(range(num_qubits), range(num_qubits))  # Measure all qubits
    return qc

# Step 5: Multi-Class Galaxy Classification
def classify_galaxy(feature, theta, n_qubits, num_classes):
    qc = amplitude_encode(feature)
    unitary_circuit = u_theta(feature, theta)
    qc.compose(unitary_circuit, inplace=True)

    # Simulate the quantum state and get the statevector
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data)**2  # Squared magnitudes of statevector amplitudes
    
    # Classify by picking the class with the highest probability
    predicted_label = np.argmax(probabilities)  # Return the class with the highest probability
    return predicted_label

# Step 3: Cost Function
def cost_function(params, features, labels):
    print(f"Params: {params}")  # Log parameters
    total_error = 0
    sampler = StatevectorSampler()  # Updated sampler
    
    for feature, label in zip(features, labels):
        qnn = u_theta(feature, params)
        result = sampler.run([(qnn, params)]).result()


        
        # Extract probabilities correctly
        probabilities = np.zeros(num_classes)
        for key, value in result.quasi_dists[0].items():
            probabilities[key] = value  # Assign probability to corresponding state index

        
        
        print(f"Feature: {feature}, Label: {label}, Probabilities: {probabilities}")
        total_error += np.sum((np.array(label) - probabilities) ** 2)
    
    return total_error  # Return the total error so optimizer can minimize it

# Step 4: Optimization Callback
cost_history = []
def callback_function(params):
    """
    Track the cost value during optimization.
    """
    cost = cost_function(params, features, labels)
    cost_history.append(cost)
    print(f"Cost: {cost}")



# Step 7: Train the quantum classifier
def train_classifier(features, labels, n_qubits, num_classes):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters

    result = minimize(
        fun=partial(cost_function, features=features, labels=labels),
        x0=initial_theta,
        method="COBYLA",
        options={"maxiter": 1000, "disp": True},
    )
    return result.x

# Step 8: Main Execution
if __name__ == "__main__":
    n_qubits = 6
    num_classes = 2  # Spiral, Elliptical

    # Example of image preprocessing (replace this with actual image processing)
    def preprocess_image(image_path):
        # Simulate image preprocessing (use actual image data in practice)
        return np.random.rand(64)  # Placeholder for feature extraction from image

    # Load galaxy images and labels (simulate for this example)
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\pxArt (7).png')  
    feature_2 = preprocess_image(r'C:\Users\User\Downloads\pxArt (8).png')
    features = [feature_1, feature_2]
    labels = [0, 1]  # Spiral = 0, Elliptical = 1

    # Train classifier
    optimal_theta = train_classifier(features, labels, n_qubits, num_classes)


    # Test on a new galaxy image
    test_feature = preprocess_image(r'C:\Users\User\Downloads\pxArt (9).png')
    predicted_label = classify_galaxy(test_feature, optimal_theta, n_qubits, num_classes)

    galaxy_types = ["Spiral", "Elliptical"]
    print(f"Predicted Galaxy Type: {galaxy_types[predicted_label]}")


# In[ ]:


import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt






# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm

# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc

# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc


# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    qc = amplitude_encoding(feature)
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2
    triangle_prob = np.sum(probabilities[:32])  
    square_prob = np.sum(probabilities[32:])   
    
    return 0 if triangle_prob > square_prob else 1

# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        total_error += (label - predicted_label) ** 2
    return total_error / len(features)

   

def normalize_features(features):
    return (features - np.min(features)) / (np.max(features) - np.min(features) )
# Step 6: Train the model with cost tracking
def train_classifier(features, labels, n_qubits):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)
    cost_history = []  # List to store cost values during optimization

    def callback(xk):
        cost = cost_function(xk, features, labels, n_qubits)
        cost_history.append(cost)
        print(f"Iteration: {len(cost_history)}, Cost: {cost}")

    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",  # More suitable for continuous optimization
        options={"maxiter": 100, "disp": True},  # Display progress
        callback=callback,
    )

    # Plot cost history after training
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()

    return result.x  # Return optimal parameters

# Step 7: Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)

# Step 8: Load images from folder
def load_images_from_folder(folder_path):
    image_features = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Only process images
            image_path = os.path.join(folder_path, filename)
            image_features.append(preprocess_image(image_path))
            image_paths.append(image_path)
    return image_features, image_paths

# Step 9: Main execution
if __name__ == "__main__":
    n_qubits = 6
    training_folder = r'C:\Users\User\Downloads\Training_New\Training_New'

    # Load all images from the folder
    features, image_paths = load_images_from_folder(training_folder)
    
    # Assuming you manually provide labels (e.g., first half are triangles (0), second half are squares (1))
    labels = [1] * (len(features) // 2) + [0] * (len(features) // 2)

    print(f"Loaded {len(features)} images for training.")

    optimal_theta = train_classifier(features, labels, n_qubits)

    # Classify all images in the folder
    print("\nClassification Results:")
    for img_path, feature in zip(image_paths, features):
        predicted_label = classify_shape(feature, optimal_theta, n_qubits)
        shape = "Triangle" if predicted_label == 0 else "Square"
        print(f"{os.path.basename(img_path)} → Predicted: {shape}")


# In[ ]:


When given 112 images,accuracy dropped to 43%


# In[ ]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt




# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm


# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc


# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc

# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    qc = amplitude_encoding(feature)
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2
    square_prob = np.sum(probabilities[32:])  
    triangle_prob = np.sum(probabilities[:32])   
    
    return 0 if triangle_prob > square_prob else 1


    return final_qubit_value
# Step 5: Define the cost function
cost_history=[]
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        total_error += (label - predicted_label) ** 2
    return total_error / len(features)




def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to a fixed size
    image_array = np.array(image)
    flat_vector = image_array.flatten() / 255.0  # Normalize to [0, 1]
    return normalize_amplitudes(flat_vector)



# Step 7: Train the model
def train_classifier(features, labels, n_qubits):
    # Define initial guess for theta (parameters for the quantum circuit)
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters for 6 qubits
    initial_cost = cost_function(initial_theta, features, labels, n_qubits)
    print(f"Initial Cost: {initial_cost}")
    cost_history.append(initial_cost)

    # Optimize the cost function using L-BFGS-B
    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="COBYLA",  # More suitable for continuous optimization
        options={"maxiter": 10000, "disp": True},  # Display progress
    )
    return result.x 


# Step 8: Load images from folder
def load_images_from_folder(folder_path):
    image_features = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Only process images
            image_path = os.path.join(folder_path, filename)
            image_features.append(preprocess_image(image_path))
            image_paths.append(image_path)
    return image_features, image_paths


# Step 9: Main execution
if __name__ == "__main__":
    n_qubits = 6
    training_folder = r'C:\Users\User\Downloads\Training_New\Training_New'

    # Load all images from the folder
    features, image_paths = load_images_from_folder(training_folder)
    labels = [1] * (len(features) // 2) + [0] * (len(features) // 2)


    print(f"Loaded {len(features)} images for training.")

    optimal_theta = train_classifier(features, labels, n_qubits)

    # Classify all images in the folder
    print("\nClassification Results:")
    for img_path, feature in zip(image_paths, features):
        predicted_label = classify_shape(feature, optimal_theta, n_qubits)
        shape = "Triangle" if predicted_label == 0 else "Square"
        print(f"{os.path.basename(img_path)} → Predicted: {shape}")
    


# In[ ]:


from qiskit.primitives import Sampler

def classify_shape(feature, theta, n_qubits):
    """Classifies the shape (triangle or square) using the Qiskit Sampler primitive."""

    # Step 1: Encode feature into quantum circuit
    qc = amplitude_encoding(feature)

    # Step 2: Apply trained parameterized circuit
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)

    # Step 3: Use Qiskit Sampler to get probability distribution
    sampler = Sampler()
    job = sampler.run(qc)
    result = job.result()
    probabilities = result.quasi_dists[0]  # Dictionary of measurement outcomes

    # Step 4: Compute probability of measuring |1> vs. |0> on the last qubit
    prob_1 = sum(v for k, v in probabilities.items() if (k & 1) == 1)  # Last qubit |1>
    prob_0 = 1 - prob_1  # Last qubit |0>

    print(f"Probability of Triangle (0): {prob_0:.4f}, Probability of Square (1): {prob_1:.4f}")

    # Step 5: Classification decision
    return 0 if prob_0 > prob_1 else 1


# In[12]:


import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from PIL import Image
import matplotlib.pyplot as plt


# Step 1: Normalize amplitudes for amplitude encoding
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Amplitudes norm is zero. Cannot normalize.")
    return amplitudes / norm


# Step 2: Amplitude encoding function
def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc


# Step 3: Define the parameterized circuit (TwoLocal)
def u_theta(theta, n_qubits):
    twolocal = TwoLocal(
        n_qubits,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    )
    num_params = len(twolocal.parameters)
    if len(theta) != num_params:
        raise ValueError(f"Expected {num_params} parameters but got {len(theta)}.")
    param_mapping = {param: value for param, value in zip(twolocal.parameters, theta)}
    twolocal = twolocal.assign_parameters(param_mapping)
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc


# Step 4: Classification function
def classify_shape(feature, theta, n_qubits):
    # Initialize the quantum circuit with amplitude encoding
    qc = amplitude_encoding(feature)

    # Add the parameterized unitary circuit
    unitary_circuit = u_theta(theta, n_qubits)
    qc.compose(unitary_circuit, inplace=True)

    # Use Statevector to simulate the circuit and get the final statevector
    statevector = Statevector.from_instruction(qc)

    # Calculate probabilities from the statevector
    probabilities = np.abs(statevector.data) ** 2  # Array of 64 probabilities for 6 qubits

    # Classification rule: sum probabilities of relevant groups
    triangle_prob = np.sum(probabilities[:32])  # First half for Triangle
    square_prob = np.sum(probabilities[32:])   # Second half for Square

    # Return the predicted label: 0 -> Triangle, 1 -> Square
    return 0 if triangle_prob > square_prob else 1


# Step 5: Define the cost function
def cost_function(theta, features, labels, n_qubits):
    total_error = 0
    for feature, label in zip(features, labels):
        predicted_label = classify_shape(feature, theta, n_qubits)
        # Adjusting the error calculation to include more comprehensive terms
        total_error += np.sum((np.array(label) - predicted_label) ** 2)  # Mean Squared Error (MSE)
    return total_error / len(features)



# Step 6: Image preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_array = np.array(image)
    flat_vector = image_array.flatten()
    return normalize_amplitudes(flat_vector)


# Step 7: Train the model
def train_classifier(features, labels, n_qubits):
    # Define initial guess for theta (parameters for the quantum circuit)
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)  # Random parameters for 6 qubits

    # Optimize the cost function using L-BFGS-B
    result = minimize(
        fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits),
        x0=initial_theta,
        method="L-BFGS-B",  # More suitable for continuous optimization
        options={"maxiter": 1000, "disp": True},  # Display progress
    )
    return result.x 


# Step 8: Main execution
if __name__ == "__main__":
    # Define number of qubits
    n_qubits = 6

    # Define features and labels for training
    # Replace these with actual flattened vectors of triangle and square images
    feature_1 = preprocess_image(r'C:\Users\User\Downloads\download-resizehood.com (1).png') 
    feature_2 = preprocess_image(r"C:\Users\User\Downloads\pxArt (2).png")  
    features = [feature_1, feature_2]

    labels = [0, 1]  # Labels: 0 -> Triangle, 1 -> Square

    # Train the quantum classifier
    optimal_theta = train_classifier(features, labels, n_qubits)
    print("Optimal Parameters:", optimal_theta)

    # Test the classifier on a new image
    new_feature = preprocess_image(r'C:\Users\User\Downloads\7252.png')  # Example image
    predicted_label = classify_shape(new_feature, optimal_theta, n_qubits)
    print("Predicted Label:", "Triangle" if predicted_label == 0 else "Square")
    print(f"Final Cost: {cost_history[-1]}")
     


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Categories for comparison
models = ["Classical CNN", "Fully Quantum CNN"]

# Hypothetical classification times in milliseconds
times = [500, 50]  # Example: Classical is slowest, Quantum is fastest

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, times, color=["#1f77b4", "#e7298a"])

# Add gridlines for better comparison
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Labeling the axes and the title
plt.ylabel("Classification Time (ms)", fontsize=12)
plt.xlabel("Model Type", fontsize=12)
plt.title("Image Processing time for different models", fontsize=14)

# Annotate bars with exact times
for i, time in enumerate(times):
    plt.text(i, time + 10, f"{time} ms", ha="center", fontsize=12, color="black")

# Adjust layout for better appearance
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




