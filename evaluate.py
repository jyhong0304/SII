from pascalpart import *
from collections import Counter
import csv
import pdb, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dirpath = os.getcwd()

matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 18

np.set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf)

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU':1})

thresholds = np.arange(.00,1.1,.05)
models_dir = dirpath + "/models/"
results_dir = dirpath + "/results/"

errors_percentage = np.array([0.0])
constraints_choice = ["KB_wc_nr_", "RWTN_KB_wc_nr_"]

paths_to_models = ["baseline"]
labels_of_models = ["baseline"]

for error in errors_percentage:
    for constraints in constraints_choice:        
        paths_to_models.append(models_dir + constraints + str(error) + ".ckpt")
        labels_of_models.append(constraints + "_" +str(error))

# loading test data
test_data, pairs_of_test_data, types_of_test_data, partOF_of_pairs_of_test_data, pairs_of_bb_idxs_test, pics = get_data("test", max_rows=50000)

# generating and printing some report on the test data
number_of_test_data_per_type = Counter(types_of_test_data)
print number_of_test_data_per_type
type_cardinality_array = np.array([number_of_test_data_per_type[t] for t in selected_types])
idxs_for_selected_types = np.concatenate([np.where(types == st)[0] for st in selected_types])
print idxs_for_selected_types

# generating new features for box overlapping
def partof_baseline_test(bb_pair_idx, wholes_of_part, threshold=0.7, with_partof_axioms=False):
    type_compatibility = True
    if with_partof_axioms:
        type_compatibility = False
        part_whole_pair = pairs_of_bb_idxs_test[bb_pair_idx]
        type_part = types_of_test_data[part_whole_pair[0]]
        type_whole = types_of_test_data[part_whole_pair[1]]
        if type_whole in wholes_of_part[type_part]:
            type_compatibility = True

    return (pairs_of_test_data[bb_pair_idx][-2] >= max(threshold, pairs_of_test_data[bb_pair_idx][-1])) and type_compatibility


def plot_prec_rec_curve(precisionW_new, recallW_new, precisionW, recallW, precisionB, recallB, label):
    fig = plt.figure(figsize=(10.0, 8.0))
    label_baseline_legend='FRCNN'
    if 'part-of' in label:
        recallB = [0.0, recallB[0]]
        precisionB = [precisionB[0], precisionB[0]]
        label_baseline_legend = 'FRCNN'

    idx_recallW = np.argsort(recallW)
    idx_recallW_new = np.argsort(recallW_new)
    idx_recallB = np.argsort(recallB)

    aucW = np.trapz(np.array(precisionW)[idx_recallW], x=np.array(recallW)[idx_recallW])
    aucW_new = np.trapz(np.array(precisionW_new)[idx_recallW_new], x=np.array(recallW_new)[idx_recallW_new])
    aucB = np.trapz(np.array(precisionB)[idx_recallB], x=np.array(recallB)[idx_recallB])

    plt.plot(recallW_new, precisionW_new, lw=3, color='blue', label='RWFN: AUC={0:0.3f}'.format(aucW_new))
    plt.plot(recallW, precisionW, lw=3, color='green', label='LTN: AUC={0:0.3f}'.format(aucW))
    plt.plot(recallB, precisionB, lw=3, color='red', label=label_baseline_legend +': AUC={0:0.3f}'.format(aucB))
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.title('Precision-Recall curve '+label.split('_')[1], fontsize=25)
    plt.legend(loc="lower left")

    fig.savefig(os.path.join(results_dir,'prec_rec_curve_'+label+'.png'))
    plt.close(fig)


def confusion_matrix_for_baseline(thresholds,with_partof_axioms=False):
    print ""
    print "computing confusion matrix for the baseline"
    confusion_matrix_for_types = {}
    confusion_matrix_for_pof = {}
    for th in thresholds:
        print th, " ",
        confusion_matrix_for_types[th] = np.matrix([[0.0] * len(selected_types)] * len(selected_types))
        for bb_idx in range(len(test_data)):
            for st_idx in range(len(selected_types)):
                st_feature_of_bb_idx = test_data[bb_idx][1+idxs_for_selected_types[st_idx]]
                if st_feature_of_bb_idx >= th:
                    confusion_matrix_for_types[th][st_idx,np.where(selected_types == types_of_test_data[bb_idx])[0][0]]+= 1

        confusion_matrix_for_pof[th] = np.matrix([[0.0,0.0],[0.0,0.0]])

        wholes_of_part={}
        if with_partof_axioms:
            _, wholes_of_part = get_part_whole_ontology()

        for bb_pair_idx in range(len(pairs_of_test_data)):
            if partof_baseline_test(bb_pair_idx, wholes_of_part, with_partof_axioms=with_partof_axioms):
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][0,0] +=1
                else:
                    confusion_matrix_for_pof[th][0,1] +=1
            else:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][1,0] += 1
                else:
                    confusion_matrix_for_pof[th][1,1] += 1

    return confusion_matrix_for_types, confusion_matrix_for_pof


# determining the values of the atoms isOfType[t](bb) and isPartOf(bb1,bb2) for every type t and for every bounding box bb, bb1 and bb2.
def compute_values_atomic_formulas(path_to_model, use_new=False):
    if use_new:
        predicted_types_values_tensor = tf.concat([isOfType_rwtn[t].tensor() for t in selected_types], 1)
        predicted_partOf_value_tensor = rwfn.Literal(True, isPartOf_rwtn, pairs_of_objects_rwtn).tensor
    else:
        predicted_types_values_tensor = tf.concat([isOfType[t].tensor() for t in selected_types], 1)
        predicted_partOf_value_tensor = ltn.Literal(True,isPartOf,pairs_of_objects).tensor
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, path_to_model)
    if use_new:
        values_of_types = sess.run(predicted_types_values_tensor, {objects_rwtn.tensor: test_data[:, 1:]})
        values_of_partOf = sess.run(predicted_partOf_value_tensor, {pairs_of_objects_rwtn.tensor: pairs_of_test_data})
    else:
        values_of_types = sess.run(predicted_types_values_tensor,{objects.tensor:test_data[:,1:]})
        values_of_partOf = sess.run(predicted_partOf_value_tensor,{pairs_of_objects.tensor:pairs_of_test_data})
    sess.close()
    return values_of_types, values_of_partOf


# computing confusion matrixes for the prediction of a model
def confusion_matrixes_of_model(path_to_model,thresholds, use_new=False):
    print ""
    print "computing confusion matrix for", path_to_model
    global test_data, types_of_test_data, partOF_of_pairs_of_test_data, bb_idxs_pairs
    values_of_types, values_of_partOf = compute_values_atomic_formulas(path_to_model, use_new)
    confusion_matrix_for_types = {}
    confusion_matrix_for_pof = {}
    #pdb.set_trace()
    for th in thresholds:
        print th," ",
        confusion_matrix_for_types[th] = np.matrix([[0.0] * len(selected_types)] * len(selected_types))
        for bb_idx in range(len(test_data)):
            for st_idx in range(len(selected_types)):
                if values_of_types[bb_idx][st_idx] >= th:
                    confusion_matrix_for_types[th][st_idx, np.where(selected_types == types_of_test_data[bb_idx])[0][0]] += 1
        confusion_matrix_for_pof[th] = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        for bb_pair_idx in range(len(pairs_of_test_data)):
            if values_of_partOf[bb_pair_idx] >= th:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][0, 0] += 1
                else:
                    confusion_matrix_for_pof[th][0, 1] += 1
            else:
                if partOF_of_pairs_of_test_data[bb_pair_idx]:
                    confusion_matrix_for_pof[th][1, 0] += 1
                else:
                    confusion_matrix_for_pof[th][1, 1] += 1

    return confusion_matrix_for_types, confusion_matrix_for_pof

measure_per_type = {}
measure_per_pof = {}

measures = ["prec","recall","f1"]

for measure in measures:
    measure_per_pof[measure] = {}
    measure_per_type[measure] = {}

for path_to_model in paths_to_models:
    if path_to_model == "baseline":
        cm_types, cm_pof = confusion_matrix_for_baseline(thresholds, with_partof_axioms=False)
    elif "RWTN" in path_to_model:
        cm_types, cm_pof = confusion_matrixes_of_model(path_to_model,thresholds, True)
    else:
        cm_types, cm_pof = confusion_matrixes_of_model(path_to_model, thresholds, False)
    for measure in measures:
        measure_per_type[measure][path_to_model] = {}
        measure_per_pof[measure][path_to_model] = {}
    for th in thresholds:
        measure_per_type["prec"][path_to_model][th] = precision(cm_types[th])
        measure_per_type["recall"][path_to_model][th] = recall(cm_types[th],gold_array=type_cardinality_array)
        measure_per_type["f1"][path_to_model][th] = f1(measure_per_type["prec"][path_to_model][th],
                                                        measure_per_type["recall"][path_to_model][th])
        measure_per_pof["prec"][path_to_model][th] = precision(cm_pof[th])
        measure_per_pof["recall"][path_to_model][th] = recall(cm_pof[th])
        measure_per_pof["f1"][path_to_model][th] = f1(measure_per_pof["prec"][path_to_model][th],
                                                       measure_per_pof["recall"][path_to_model][th])

print ""
print "writing report in file "+ os.path.join(results_dir,"report.csv")
with open(os.path.join(results_dir,"report.csv"), "w") as report:
    writer = csv.writer(report, delimiter=';')
    writer.writerow(["threshold",""] + [y for x in [[th]*len(measures)*len(paths_to_models) for th in thresholds] for y in x])
    writer.writerow(["measure", ""] + [y for x in [[meas]*len(paths_to_models) for meas in measures] for y in x]*len(thresholds))
    writer.writerow(["models",""] + labels_of_models*len(measures)*len(thresholds))
    writer.writerow(["part of", ""] + [measure_per_pof[measure][mod][th][0, 0] for th in thresholds for measure in measures for mod in paths_to_models])
    writer.writerow(["average x types", ""] + [measure_per_type[measure][mod][th].mean() for th in thresholds for measure in measures for mod in paths_to_models])
    for t in selected_types:
        writer.writerow([t, number_of_test_data_per_type[t]] + [measure_per_type[measure][mod][th][0,np.where(selected_types == t)[0][0]] for th in thresholds for measure in measures for mod in paths_to_models])

ltn_performance_pof_w = []
rtn_performance_pof_w = []
ltn_performance_pof_b = []
ltn_performance_types_w = []
rtn_performance_types_w = []
ltn_performance_types_b = []

def adjust_prec(precision):
    prec = precision
    for idx_prec in range(len(precision)):
        if np.isnan(precision[idx_prec]):
            prec[idx_prec] = precision[idx_prec-1]
    return prec

for error in errors_percentage:
    ap_types_w = []
    ap_types_w_new = []
    ap_types_b = []
    prec_types_w = []
    prec_types_w_new = []
    prec_types_b = []
    rec_types_w = []
    rec_types_w_new = []
    rec_types_b = []

    precisionW_new = [measure_per_pof["prec"][models_dir +"RWTN_KB_wc_nr_"+ str(error) + ".ckpt"][th][0, 0] for th in thresholds]
    recallW_new = [measure_per_pof["recall"][models_dir +"RWTN_KB_wc_nr_"+ str(error) + ".ckpt"][th][0, 0] for th in thresholds]
    precisionW = [measure_per_pof["prec"][models_dir +"KB_wc_nr_"+ str(error) + ".ckpt"][th][0, 0] for th in thresholds]
    recallW = [measure_per_pof["recall"][models_dir +"KB_wc_nr_"+ str(error) + ".ckpt"][th][0, 0] for th in thresholds]
    recallB_pof = [measure_per_pof["recall"]["baseline"][th][0,0] for th in thresholds]
    precisionB_pof = [measure_per_pof["prec"]["baseline"][th][0,0] for th in thresholds]

    precisionW = adjust_prec(precisionW)
    precisionW_new = adjust_prec(precisionW_new)
    precisionB_pof = adjust_prec(precisionB_pof)
    idx_recallW = np.argsort(recallW)
    idx_recallW_new = np.argsort(recallW_new)

    plot_prec_rec_curve(precisionW_new, recallW_new, precisionW, recallW, precisionB_pof, recallB_pof, str(int(error*100)) + '_part-of')

    ltn_performance_pof_w.append(np.trapz(np.array(precisionW)[idx_recallW], x=np.array(recallW)[idx_recallW]))
    rtn_performance_pof_w.append(np.trapz(np.array(precisionW_new)[idx_recallW_new], x=np.array(recallW_new)[idx_recallW_new]))
    recallB = [0.0, recallB_pof[0]]
    precisionB = [precisionB_pof[0], precisionB_pof[0]]
    ltn_performance_pof_b.append(np.trapz(np.array(precisionB), x=np.array(recallB)))

    for t in selected_types:
        index_type = np.where(selected_types == t)[0][0]

        precisionW_new_types = [measure_per_type["prec"][models_dir +"RWTN_KB_wc_nr_"+ str(error) + ".ckpt"][th][0,index_type] for th in thresholds]
        recallW_new_types = [measure_per_type["recall"][models_dir +"RWTN_KB_wc_nr_"+ str(error) + ".ckpt"][th][0,index_type] for th in thresholds]
        precisionW_types = [measure_per_type["prec"][models_dir +"KB_wc_nr_"+ str(error) + ".ckpt"][th][0,index_type] for th in thresholds]
        recallW_types = [measure_per_type["recall"][models_dir +"KB_wc_nr_"+ str(error) + ".ckpt"][th][0,index_type] for th in thresholds]
        precisionB_types = [measure_per_type["prec"]["baseline"][th][0,index_type] for th in thresholds]        
        recallB_types = [measure_per_type["recall"]["baseline"][th][0,index_type] for th in thresholds]

        prec_types_w.append(precisionW_types)
        prec_types_w_new.append(precisionW_new_types)
        prec_types_b.append(precisionB_types)
        rec_types_w.append(recallW_types)
        rec_types_w_new.append(recallW_new_types)
        rec_types_b.append(recallB_types)
        
        precisionW_types = adjust_prec(precisionW_types)
        precisionW_new_types = adjust_prec(precisionW_new_types)
        precisionB_types = adjust_prec(precisionB_types)
        plot_prec_rec_curve(precisionW_new_types, recallW_new_types, precisionW_types, recallW_types, precisionB_types, recallB_types, str(int(error*100)) + "_" + t)

        idx_recallW_types = np.argsort(recallW_types)
        idx_recallW_new_types = np.argsort(recallW_new_types)
        idx_recallB_types = np.argsort(recallB_types)
        ap_types_w.append(np.trapz(np.array(precisionW_types)[idx_recallW_types], x=np.array(recallW_types)[idx_recallW_types]))
        ap_types_w_new.append(np.trapz(np.array(precisionW_new_types)[idx_recallW_new_types], x=np.array(recallW_new_types)[idx_recallW_new_types]))
        ap_types_b.append(np.trapz(np.array(precisionB_types)[idx_recallB_types], x=np.array(recallB_types)[idx_recallB_types]))

    plot_prec_rec_curve(np.mean(prec_types_w_new, axis=0), np.mean(rec_types_w_new, axis=0),
                        np.mean(prec_types_w, axis=0), np.mean(rec_types_w, axis=0),
                        np.mean(prec_types_b, axis=0), np.mean(rec_types_b, axis=0), str(int(error * 100)) + "_types")

    ltn_performance_types_w.append(np.mean(ap_types_w))
    rtn_performance_types_w.append(np.mean(ap_types_w_new))
    ltn_performance_types_b.append(np.mean(ap_types_b))

