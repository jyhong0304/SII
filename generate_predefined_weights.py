import logictensornetworks as ltn
import randomly_weighted_feature_networks as rwtn
import numpy as np
import csv
import os

dirpath = os.getcwd()

# Setting
# LTN
ltn.default_layers = 6
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "luk"
ltn.default_aggregator = "hmean"
ltn.default_positive_fact_penality = 0.
ltn.default_clauses_aggregator = "hmean"

# RWTN
rwtn.default_smooth_factor = 1e-10
rwtn.default_tnorm = "luk"
rwtn.default_aggregator = "hmean"
rwtn.default_positive_fact_penality = 0.
rwtn.default_clauses_aggregator = "hmean"

data_training_dir = dirpath + "/data/training/"
data_testing_dir = dirpath + "/data/testing/"
zero_distance_threshold = 6
number_of_features = 65

##
num_layers_object_classification = 200
num_layers_partof_detection = 400

##### Generating weights for object classification
### IN-MB input transformation
rwtn_V_object = rwtn.generate_V(num_layers=num_layers_object_classification, num_features=number_of_features - 1)
with open('./predefined_weights/rwtn_V_object.txt', 'wb') as file_rwtn_V_object:
    np.save(file_rwtn_V_object, rwtn_V_object)

### Random Fourier features
rwtn_R_object = np.random.normal(size=(num_layers_object_classification, number_of_features - 1))
with open('./predefined_weights/rwtn_R_object.txt', 'wb') as file_rwtn_R_object:
    np.save(file_rwtn_R_object, rwtn_R_object)

rwtn_Rb_object = np.random.uniform(low=0, high=2 * np.pi, size=(1, num_layers_object_classification))
with open('./predefined_weights/rwtn_Rb_object.txt', 'wb') as file_rwtn_Rb_object:
    np.save(file_rwtn_Rb_object, rwtn_Rb_object)

##### Generating weights for part-of detection
### IN-MB input transformation
rwtn_V_pair = rwtn.generate_V(num_layers=num_layers_partof_detection, num_features=2 * (number_of_features - 1) + 2)
with open('./predefined_weights/rwtn_V_pair.txt', 'wb') as file_rwtn_V_pair:
    np.save(file_rwtn_V_pair, rwtn_V_pair)

### Random Fourier features
rwtn_R_pair = np.random.normal(size=(num_layers_partof_detection, 2 * (number_of_features - 1) + 2))
with open('./predefined_weights/rwtn_R_pair.txt', 'wb') as file_rwtn_R_pair:
    np.save(file_rwtn_R_pair, rwtn_R_pair)

rwtn_Rb_pair = np.random.uniform(low=0, high=2 * np.pi, size=(1, num_layers_partof_detection))
with open('./predefined_weights/rwtn_Rb_pair.txt', 'wb') as file_rwtn_Rb_pair:
    np.save(file_rwtn_Rb_pair, rwtn_Rb_pair)

print("Generating weights done.")
