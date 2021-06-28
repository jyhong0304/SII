import logictensornetworks as ltn
import randomly_weighted_feature_networks as rwfn
import numpy as np
import csv
import os
import time

dirpath = os.getcwd()

# Setting
# LTN
ltn.default_layers = 6
ltn.default_smooth_factor = 1e-10
ltn.default_tnorm = "luk"
ltn.default_aggregator = "hmean"
ltn.default_positive_fact_penality = 0.
ltn.default_clauses_aggregator = "hmean"

# RWFN
rwfn.default_smooth_factor = 1e-10
rwfn.default_tnorm = "luk"
rwfn.default_aggregator = "hmean"
rwfn.default_positive_fact_penality = 0.
rwfn.default_clauses_aggregator = "hmean"

data_training_dir = dirpath + "/data/training/"
data_testing_dir = dirpath + "/data/testing/"
zero_distance_threshold = 6
number_of_features = 65

######
#### Uncomment the following lines to use predefiend weights
# ## Load weights for object classification
# with open('./predefined_weights/rwtn_V_object.txt', 'rb') as file_rwtn_V_object:
#     rwtn_V_object = np.load(file_rwtn_V_object)
# with open('./predefined_weights/rwtn_R_object.txt', 'rb') as file_rwtn_R_object:
#     rwtn_R_object = np.load(file_rwtn_R_object)
# with open('./predefined_weights/rwtn_Rb_object.txt', 'rb') as file_rwtn_Rb_object:
#     rwtn_Rb_object = np.load(file_rwtn_Rb_object)
# ## Load weights for part-of detection
# with open('./predefined_weights/rwtn_V_pair.txt', 'rb') as file_rwtn_V_pair:
#     rwtn_V_pair = np.load(file_rwtn_V_pair)
# with open('./predefined_weights/rwtn_R_pair.txt', 'rb') as file_rwtn_R_pair:
#     rwtn_R_pair = np.load(file_rwtn_R_pair)
# with open('./predefined_weights/rwtn_Rb_pair.txt', 'rb') as file_rwtn_Rb_pair:
#     rwtn_Rb_pair = np.load(file_rwtn_Rb_pair)
######

types = np.genfromtxt(dirpath + "/data/classes.csv", dtype="S", delimiter=",")

# uncomment this line for training the vehicle object types
# selected_types = np.array(['aeroplane','artifact_wing','body','engine','stern','wheel','bicycle','chain_wheel','handlebar','headlight','saddle','bus','bodywork','door','license_plate','mirror','window','car','motorbike','train','coach','locomotive','boat'])

# uncomment this line for training the indoor object types
selected_types = np.array(['bottle', 'body', 'cap', 'pottedplant', 'plant', 'pot', 'tvmonitor', 'screen', 'chair', 'sofa', 'diningtable'])

# uncomment this line for training the animal object types
# selected_types = np.array(['person','arm','ear','ebrow','foot','hair','hand','mouth','nose','eye','head','leg','neck','torso','cat','tail','bird','animal_wing','beak','sheep','horn','muzzle','cow','dog','horse','hoof'])

# uncomment this line for training all the object types
# selected_types = types[1:]

# LTN
ltn_prep_time_start = time.time()
objects = ltn.Domain(number_of_features - 1, label="a_bounding_box")
pairs_of_objects = ltn.Domain(2 * (number_of_features - 1) + 2, label="a_pair_of_bounding_boxes")

isOfType = {}
for t in selected_types:
    isOfType[t] = ltn.Predicate("is_of_type_" + t, objects)
isPartOf = ltn.Predicate("is_part_of", pairs_of_objects)

objects_of_type = {}
objects_of_type_not = {}
for t in selected_types:
    objects_of_type[t] = ltn.Domain(number_of_features - 1, label="objects_of_type_" + t)
    objects_of_type_not[t] = ltn.Domain(number_of_features - 1, label="objects_of_type_not_" + t)

object_pairs_in_partOf = ltn.Domain((number_of_features - 1) * 2 + 2,
                                    label="object_pairs_in_partof_relation")
object_pairs_not_in_partOf = ltn.Domain((number_of_features - 1) * 2 + 2,
                                        label="object_pairs_not_in_partof_relation")
ltn_prep_time = time.time() - ltn_prep_time_start

# RWFN
rwfn_prep_time_start = time.time()
objects_rwtn = rwfn.Domain(number_of_features - 1, label="rwtn_a_bounding_box")
pairs_of_objects_rwtn = rwfn.Domain(2 * (number_of_features - 1) + 2, label="rwtn_a_pair_of_bounding_boxes")

isOfType_rwtn = {}
for t in selected_types:
    ## uncomment this line if you want to generate weights
    isOfType_rwtn[t] = rwfn.Predicate(label="rwtn_is_of_type_" + t, domain=objects_rwtn, layers=200)
    ## uncomment this line if you want to use predefined weights
    # isOfType_rwtn[t] = rwfn.Predicate(label="rwtn_is_of_type_" + t, domain=objects_rwtn, layers=200,
    #                                   predefined_V=rwtn_V_object, predefined_R=rwtn_R_object,
    #                                   predefined_Rb=rwtn_Rb_object)
## uncomment this line if you want to generate weights
isPartOf_rwtn = rwfn.Predicate(label="rwtn_is_part_of", domain=pairs_of_objects_rwtn, layers=400)
## uncomment this line if you want to use predefined weights
# isPartOf_rwtn = rwfn.Predicate(label="rwtn_is_part_of", domain=pairs_of_objects_rwtn, layers=400,
#                                predefined_V=rwtn_V_pair, predefined_R=rwtn_R_pair,
#                                predefined_Rb=rwtn_Rb_pair)

objects_of_type_rwtn = {}
objects_of_type_not_rwtn = {}
for t in selected_types:
    objects_of_type_rwtn[t] = rwfn.Domain(number_of_features - 1, label="rwtn_objects_of_type_" + t)
    objects_of_type_not_rwtn[t] = rwfn.Domain(number_of_features - 1, label="rwtn_objects_of_type_not_" + t)

object_pairs_in_partOf_rwtn = rwfn.Domain((number_of_features - 1) * 2 + 2,
                                          label="rwtn_object_pairs_in_partof_relation")
object_pairs_not_in_partOf_rwtn = rwfn.Domain((number_of_features - 1) * 2 + 2,
                                              label="rwtn_object_pairs_not_in_partof_relation")
rwfn_prep_time = time.time() - rwfn_prep_time_start


def containment_ratios_between_two_bbxes(bb1, bb2):
    bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3])
    bb2_area = (bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3])
    w_intersec = max(0, min([bb1[-2], bb2[-2]]) - max([bb1[-4], bb2[-4]]))
    h_intersec = max(0, min([bb1[-1], bb2[-1]]) - max([bb1[-3], bb2[-3]]))
    bb_area_intersection = w_intersec * h_intersec
    return [float(bb_area_intersection) / bb1_area, float(bb_area_intersection) / bb2_area]


def get_data(train_or_test_swritch, max_rows=10000000):
    assert train_or_test_swritch == "train" or train_or_test_swritch == "test"

    # Fetching the data from the file system

    if train_or_test_swritch == "train":
        data_dir = data_training_dir
    if train_or_test_swritch == "test":
        data_dir = data_testing_dir
    data = np.genfromtxt(data_dir + "features.csv", delimiter=",", max_rows=max_rows)
    types_of_data = types[np.genfromtxt(data_dir + "types.csv", dtype="i", max_rows=max_rows)]
    idx_whole_for_data = np.genfromtxt(data_dir + "partOf.csv", dtype="i", max_rows=max_rows)
    idx_of_cleaned_data = np.where(np.logical_and(
        np.all(data[:, -2:] - data[:, -4:-2] >= zero_distance_threshold, axis=1),
        np.in1d(types_of_data, selected_types)))[0]
    print "deleting", len(data) - len(idx_of_cleaned_data), "small bb out of", data.shape[0], "bb"
    data = data[idx_of_cleaned_data]
    data[:, -4:] /= 500

    # Cleaning data by removing small bounding boxes and recomputing indexes of partof data

    types_of_data = types_of_data[idx_of_cleaned_data]
    idx_whole_for_data = idx_whole_for_data[idx_of_cleaned_data]
    for i in range(len(idx_whole_for_data)):
        if idx_whole_for_data[i] != -1 and idx_whole_for_data[i] in idx_of_cleaned_data:
            idx_whole_for_data[i] = np.where(idx_whole_for_data[i] == idx_of_cleaned_data)[0]
        else:
            idx_whole_for_data[i] = -1

    # Grouping bbs that belong to the same picture

    pics = {}
    for i in range(len(data)):
        if data[i][0] in pics:
            pics[data[i][0]].append(i)
        else:
            pics[data[i][0]] = [i]

    pairs_of_data = np.array(
        [np.concatenate((data[i][1:], data[j][1:], containment_ratios_between_two_bbxes(data[i], data[j]))) for p in
         pics for i in pics[p] for j in pics[p]])

    pairs_of_bb_idxs = np.array([(i, j) for p in pics for i in pics[p] for j in pics[p]])

    partOf_of_pair_of_data = np.array([idx_whole_for_data[i] == j for p in pics for i in pics[p] for j in pics[p]])

    return data, pairs_of_data, types_of_data, partOf_of_pair_of_data, pairs_of_bb_idxs, pics


def get_part_whole_ontology():
    with open('data/pascalPartOntology.csv') as f:
        ontologyReader = csv.reader(f)
        parts_of_whole = {}
        wholes_of_part = {}
        for row in ontologyReader:
            parts_of_whole[row[0]] = row[1:]
            for t in row[1:]:
                if t in wholes_of_part:
                    wholes_of_part[t].append(row[0])
                else:
                    wholes_of_part[t] = [row[0]]
        for whole in parts_of_whole:
            wholes_of_part[whole] = []
        for part in wholes_of_part:
            if part not in parts_of_whole:
                parts_of_whole[part] = []
    selected_parts_of_whole = {}
    selected_wholes_of_part = {}
    for t in selected_types:
        selected_parts_of_whole[t] = [p for p in parts_of_whole[t] if p in selected_types]
        selected_wholes_of_part[t] = [w for w in wholes_of_part[t] if w in selected_types]
    return selected_parts_of_whole, selected_wholes_of_part


# reporting measures

def precision(conf_matrix, prediction_array=None):
    if prediction_array is not None:
        return conf_matrix.diagonal() / prediction_array
    else:
        return conf_matrix.diagonal() / conf_matrix.sum(1).T


def recall(conf_matrix, gold_array=None):
    if gold_array is not None:
        return conf_matrix.diagonal() / gold_array
    else:
        return conf_matrix.diagonal() / conf_matrix.sum(0)


def f1(precision, recall):
    return np.multiply(2 * precision, recall) / (precision + recall)


print "end of new pascalpart.py"
