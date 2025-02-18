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









    
        

        







    
    

    




