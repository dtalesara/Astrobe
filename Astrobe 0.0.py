#!/usr/bin/env python
# coding: utf-8

# Astrobe Programme 0.0
# This programme aims to predict the possibility of life using various factors such as temperature,biosgnatures,stellar hability zones.

# Looking at the prediction of life using conditions such as temperature,atmosphere and presence of water

# In[2]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AstroObject:
    def __init__(self, name, atmosphere, temperature, liquid_water):
        self.name = name
        self.atmosphere = atmosphere
        self.temperature = temperature
        self.liquid_water = liquid_water

    def has_conditions_for_life(self):
        # Conditions for life to develop: atmosphere, temperature between -20 and 50 Celsius, and liquid water
        return self.atmosphere and -20 <= self.temperature <= 50 and self.liquid_water

class Astrobe:
    def __init__(self):
        self.astro_objects = []
        self.telescope_data = []

    def add_astro_object(self, astro_object):
        self.astro_objects.append(astro_object)

    def add_telescope_data(self, data):
        self.telescope_data.append(data)

    def train_model(self):
        # Train a random forest classifier on the telescope data
        X = np.array([[d['atmosphere'], d['temperature'], d['liquid_water']] for d in self.telescope_data])
        y = np.array([d['has_life'] for d in self.telescope_data])
        
        # Check if there are enough samples to split
        if len(X) < 2:
            raise ValueError("Not enough data to split. Add more telescope data.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Model accuracy:", accuracy_score(y_test, y_pred))
        return model

    def predict_life(self, astro_object):
        # Use the trained model to predict the possibility of life on an astro object
        model = self.train_model()
        X = np.array([[astro_object.atmosphere, astro_object.temperature, astro_object.liquid_water]])
        y_pred = model.predict(X)
        return y_pred[0]

    def predict_future_conditions(self, astro_object, years):
        # Use a GCM to simulate the effects of global warming on the astro object's climate
        # This is a simplified example and actual implementation would require more complex modeling
        temperature_increase = 0.02  # degrees Celsius per year
        future_temperature = astro_object.temperature + temperature_increase * years
        return future_temperature

# Create an instance of the Astrobe simulation
astrobe = Astrobe()

# Add some astro objects and telescope data
astrobe.add_astro_object(AstroObject("Earth", True, 15, True))
astrobe.add_telescope_data({'atmosphere': True, 'temperature': 15, 'liquid_water': True, 'has_life': True})

# Adding more data to ensure there's enough for training and testing
astrobe.add_telescope_data({'atmosphere': False, 'temperature': -67, 'liquid_water': False, 'has_life': False})
astrobe.add_telescope_data({'atmosphere': True, 'temperature': 10, 'liquid_water': True, 'has_life': True})

# Predict the possibility of life on an astro object
print("Life prediction for Mars:", astrobe.predict_life(AstroObject("Mars", True, -67, False)))

# Predict the future conditions of an astro object
print("Future temperature of Earth in 10 years:", astrobe.predict_future_conditions(AstroObject("Earth", True, 15, True), 10))


# In[3]:


from sklearn.preprocessing import MinMaxScaler

class Astrobe:
    # Existing methods...

    def habitability_score(self, astro_object):
        # Scores are calculated based on atmosphere, temperature, and liquid water
        atmosphere_score = 1.0 if astro_object.atmosphere else 0.0
        temperature_score = max(0, min(1, (astro_object.temperature + 20) / 70))  # Normalized between -20 and 50
        water_score = 1.0 if astro_object.liquid_water else 0.0
        
        # Weighted sum of the factors (weights can be adjusted based on importance)
        score = 0.4 * atmosphere_score + 0.4 * temperature_score + 0.2 * water_score
        return score

# Example usage
astrobe = Astrobe()
earth = AstroObject("Earth", True, 15, True)
mars = AstroObject("Mars", True, -67, False)
print("Earth habitability score:", astrobe.habitability_score(earth))
print("Mars habitability score:", astrobe.habitability_score(mars))


# Chemical composition analysis

# In[4]:


class ChemicalComposition:
    def __init__(self, composition):
        self.composition = composition  # A dictionary of chemical elements and their proportions

    def analyze_composition(self):
        # Example analysis: Check for the presence of key elements like oxygen and carbon
        key_elements = ['O2', 'CO2', 'H2O']
        presence = {element: self.composition.get(element, 0) > 0 for element in key_elements}
        return presence

# Example usage
astrobe = Astrobe()
earth_composition = ChemicalComposition({'N2': 78, 'O2': 21, 'CO2': 0.04, 'H2O': 0.4})
mars_composition = ChemicalComposition({'CO2': 95, 'N2': 2.6, 'Ar': 1.6, 'O2': 0.15})
print("Earth composition analysis:", earth_composition.analyze_composition())
print("Mars composition analysis:", mars_composition.analyze_composition())


# In[12]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AstroObject:
    def __init__(self, name, atmosphere, temperature, liquid_water):
        self.name = name
        self.atmosphere = atmosphere
        self.temperature = temperature
        self.liquid_water = liquid_water

    def has_conditions_for_life(self):
        # Conditions for life to develop: atmosphere, temperature between -20 and 50 Celsius, and liquid water
        return self.atmosphere and -20 <= self.temperature <= 50 and self.liquid_water

class Astrobe:
    def __init__(self):
        self.astro_objects = []
        self.telescope_data = []

    def add_astro_object(self, astro_object):
        self.astro_objects.append(astro_object)

    def add_telescope_data(self, data):
        self.telescope_data.append(data)

    def train_biosignature_model(self):
        # Training a model based on biosignature data
        X = np.array([[d['biosignature1'], d['biosignature2'], d['biosignature3']] for d in self.telescope_data])
        y = np.array([d['has_life'] for d in self.telescope_data])

        # Check if both classes are present
        if len(np.unique(y)) < 2:
            raise ValueError("Not enough data to split. Add more telescope data or ensure both classes are present.")
        
        # Ensure there is enough data to split
        if len(X) <= 2:
            raise ValueError("Not enough data to split into training and testing sets. Add more data.")
        
        # Adjust test_size to avoid the issue of having too few samples in the test set
        test_size = min(0.5, len(X) // 2)  # Ensure that test_size is not too large

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Biosignature model accuracy:", accuracy_score(y_test, y_pred))
        return model

    def predict_biosignatures(self, astro_object, biosignatures):
        # Predict the possibility of life based on biosignatures
        model = self.train_biosignature_model()
        X = np.array([biosignatures])
        y_pred = model.predict(X)
        return y_pred[0]

# Example usage
astrobe = Astrobe()
astrobe.add_telescope_data({'biosignature1': 0.8, 'biosignature2': 0.6, 'biosignature3': 0.7, 'has_life': True})
astrobe.add_telescope_data({'biosignature1': 0.2, 'biosignature2': 0.1, 'biosignature3': 0.4, 'has_life': False})
astrobe.add_telescope_data({'biosignature1': 0.6, 'biosignature2': 0.4, 'biosignature3': 0.5, 'has_life': True})
astrobe.add_telescope_data({'biosignature1': 0.1, 'biosignature2': 0.2, 'biosignature3': 0.3, 'has_life': False})

print("Biosignature prediction:", astrobe.predict_biosignatures(AstroObject("Exoplanet", True, 25, True), [0.7, 0.5, 0.6]))


# Habitability Zone

# In[13]:


#Looks at the stellar hability zone
class StellarHabitabilityZone:
    def __init__(self, luminosity):
        self.luminosity = luminosity  # In terms of solar luminosity

    def calculate_habitable_zone(self):
        # Simplified model: using the formula for habitable zone distance
        inner_bound = np.sqrt(self.luminosity / 1.1)  # AU
        outer_bound = np.sqrt(self.luminosity / 0.53)  # AU
        return inner_bound, outer_bound

# Example usage
astrobe = Astrobe()
sun = StellarHabitabilityZone(1)  # Sun's luminosity
print("Sun's habitable zone:", sun.calculate_habitable_zone())


# Using infrared spectrums to determine the possibility of life

# In[ ]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define the input shape (infrared spectra)
input_shape = (1000,)  # 1000 wavelengths between 0.5-10 μm

# Define the CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=10, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(64, kernel_size=10, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the dataset (infrared spectra and labels)
X_train, y_train = ...

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# Load the new planet's infrared spectrum
new_spectrum = ...

# Make predictions
prediction = model.predict(new_spectrum)
print(prediction)


# Taking data from telescopes

# In[17]:


#Currently this only uses example data,however we can call on data from real telescopes and put this model to use.
#For example:

import lightkurve as lk


# Search for data from TESS
search_result_tess = lk.search_lightcurve('KIC 3733346', mission='TESS')

# Search for data from Kepler
search_result_kepler = lk.search_lightcurve('KIC 3733346', mission='Kepler')

# Note: JWST data is not yet available in lightkurve, you may need to use other libraries or APIs
# For JWST data, you might consider using the Astroquery library or the Mikulski Archive for Space Telescopes (MAST) API.

# Note: Hubble data is not directly available in lightkurve, you may need to use other libraries or APIs
# For Hubble data, you might consider using the Astroquery library or the Mikulski Archive for Space Telescopes (MAST) API.

# Download the light curve data
lc_tess = search_result_tess.download_all()  # Changed from .download() to .download_all() to handle multiple light curves
lc_kepler = search_result_kepler.download_all()  # Changed from .download() to .download_all() to handle multiple light curves







# In[ ]:




