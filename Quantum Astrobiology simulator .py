#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from functools import partial
from math import ceil, log2
from astropy.io import fits

# --- Classical ML for Habitability ---
class AstroObject:
    def __init__(self, name, atmosphere, temperature, liquid_water, biosignatures, infrared_signature):
        self.name = name
        self.atmosphere = atmosphere
        self.temperature = temperature
        self.liquid_water = liquid_water
        self.biosignatures = biosignatures  # Additional features for classification
        self.infrared_signature = infrared_signature  # Infrared spectral data

    def get_feature_vector(self):
        return [self.atmosphere, self.temperature, self.liquid_water, self.biosignatures] + list(self.infrared_signature)

class Astrobe:
    def __init__(self):
        self.astro_objects = []
        self.telescope_data = []

    def add_astro_object(self, astro_object):
        self.astro_objects.append(astro_object)

    def add_telescope_data(self, data):
        self.telescope_data.append(data)

    def train_model(self):
        X = np.array([d['features'] for d in self.telescope_data])
        y = np.array([d['has_life'] for d in self.telescope_data])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        return model

# --- Quantum Classification for Infrared Spectra ---
def normalize_amplitudes(amplitudes):
    norm = np.linalg.norm(amplitudes)
    return amplitudes / norm if norm != 0 else amplitudes

def amplitude_encoding(amplitudes):
    n_qubits = ceil(log2(len(amplitudes)))
    padded_length = 2 ** n_qubits
    if len(amplitudes) < padded_length:
        amplitudes = np.pad(amplitudes, (0, padded_length - len(amplitudes)))
    amplitudes = normalize_amplitudes(amplitudes)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amplitudes, range(n_qubits))
    return qc

def u_theta(theta, n_qubits):
    twolocal = TwoLocal(n_qubits, rotation_blocks=['ry'], entanglement_blocks='cx', entanglement='linear', reps=2)
    twolocal = twolocal.assign_parameters(dict(zip(twolocal.parameters, theta)))
    qc = QuantumCircuit(n_qubits)
    qc.compose(twolocal, inplace=True)
    return qc

def classify_infrared(feature, theta, n_qubits):
    qc = amplitude_encoding(feature)
    qc.compose(u_theta(theta, n_qubits), inplace=True)
    statevector = Statevector.from_instruction(qc)
    probabilities = np.abs(statevector.data) ** 2
    return np.argmax(probabilities)

def preprocess_infrared_data(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            spectrum = hdul[1].data['flux'][:128]  # Take the first 128 spectral points
        return normalize_amplitudes(spectrum)
    except Exception as e:
        print(f"Error reading {fits_file}: {e}")
        return None

def train_quantum_classifier(features, labels, n_qubits):
    initial_theta = np.random.uniform(0, 2 * np.pi, 18)
    result = minimize(fun=partial(cost_function, features=features, labels=labels, n_qubits=n_qubits), x0=initial_theta, method="L-BFGS-B")
    return result.x

def cost_function(theta, features, labels, n_qubits):
    return np.mean([(label - classify_infrared(feature, theta, n_qubits))**2 for feature, label in zip(features, labels)])

# --- Integrating Classical ML and Quantum Classification ---
astrobe = Astrobe()

# Preprocess infrared data
infrared_feature_1 = preprocess_infrared_data("jwst_spectrum_1.fits")
infrared_feature_2 = preprocess_infrared_data("jwst_spectrum_2.fits")

# Check if features are valid
if infrared_feature_1 is not None and infrared_feature_2 is not None:
    quantum_theta = train_quantum_classifier([infrared_feature_1, infrared_feature_2], [0, 1], 6)

    astro_object_1 = AstroObject("ExoplanetX", True, 10, True, 0.7, infrared_feature_1)
    astro_object_2 = AstroObject("ExoplanetY", False, -50, False, 0.2, infrared_feature_2)

    astrobe.add_telescope_data({'features': astro_object_1.get_feature_vector(), 'has_life': True})
    astrobe.add_telescope_data({'features': astro_object_2.get_feature_vector(), 'has_life': False})

    ml_model = astrobe.train_model()
    habitability_score = ml_model.predict([astro_object_1.get_feature_vector()])[0]

    print("Habitability Score:", habitability_score)
else:
    print("Failed to preprocess infrared data.")

