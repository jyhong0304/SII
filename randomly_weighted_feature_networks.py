import tensorflow as tf
import numpy as np
import sys
from scipy import sparse


default_smooth_factor = 0.0000001
default_tnorm = "product"
default_optimizer = "gd"
default_aggregator = "min"
default_positive_fact_penality = 1e-6
default_clauses_aggregator = "min"


def train_op(loss, optimization_algorithm):
    if optimization_algorithm == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, learning_rate_power=-0.5)
    if optimization_algorithm == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    if optimization_algorithm == "ada":
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    if optimization_algorithm == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
    if optimization_algorithm == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    return optimizer.minimize(loss)


def PR(tensor):
    global count
    np.set_printoptions(threshold=sys.maxsize)
    return tf.Print(tensor, [tf.shape(tensor), tensor.name, tensor], summarize=200000)


def disjunction_of_literals(literals, label="no_label"):
    list_of_literal_tensors = [lit.tensor for lit in literals]
    literals_tensor = tf.concat(list_of_literal_tensors, 1)
    if default_tnorm == "product":
        result = 1.0 - tf.reduce_prod(1.0 - literals_tensor, 1, keep_dims=True)
    if default_tnorm == "yager2":
        result = tf.minimum(1.0, tf.sqrt(tf.reduce_sum(tf.square(literals_tensor), 1, keep_dims=True)))
    if default_tnorm == "luk":
        result = tf.minimum(1.0, tf.reduce_sum(literals_tensor, 1, keep_dims=True))
    if default_tnorm == "goedel":
        result = tf.reduce_max(literals_tensor, 1, keep_dims=True, name=label)
    if default_aggregator == "product":
        return tf.reduce_prod(result, keep_dims=True)
    if default_aggregator == "mean":
        return tf.reduce_mean(result, keep_dims=True, name=label)
    if default_aggregator == "gmean":
        return tf.exp(tf.muliply(tf.reduce_sum(tf.log(result), keep_dims=True),
                                 tf.reciprocal(tf.to_float(tf.size(result)))), name=label)
    if default_aggregator == "hmean":
        return tf.div(tf.to_float(tf.size(result)), tf.reduce_sum(tf.reciprocal(result), keep_dims=True))
    if default_aggregator == "min":
        return tf.reduce_min(result, keep_dims=True, name=label)


def smooth(parameters):
    norm_of_omega = tf.reduce_sum(
        tf.expand_dims(tf.concat([tf.expand_dims(tf.reduce_sum(tf.square(par)), 0) for par in parameters], 0), 1))
    return tf.multiply(default_smooth_factor, norm_of_omega)


class Domain:
    def __init__(self, columns, dom_type="float", label=None):
        self.columns = columns
        self.label = label
        self.tensor = tf.placeholder(dom_type, shape=[None, self.columns], name=self.label)
        self.parameters = []


class Domain_concat(Domain):
    def __init__(self, domains):
        self.columns = np.sum([dom.columns for dom in domains])
        self.label = "concatenation of" + ",".join([dom.label for dom in domains])
        self.tensor = tf.concat(1, [dom.tensor for dom in domains])
        self.parameters = [par for dom in domains for par in dom.parameters]


class Domain_slice(Domain):
    def __init__(self, domain, begin_column, end_column):
        self.columns = end_column - begin_column
        self.label = "projection of" + domain.label + "from column " + begin_column + " to column " + end_column
        self.tensor = tf.concat(1, tf.split(1, domain.columns, domain.tensor)[begin_column:end_column])
        self.parameters = domain.parameters


class Function(Domain):
    def __init__(self, label, domain, range, value=None):
        self.label = label
        self.domain = domain
        self.range = range
        self.value = value
        if self.value:
            self.parameters = []
        else:
            self.M = tf.Variable(tf.random_normal([self.domain.columns,
                                                   self.range.columns]),
                                 name="M_" + self.label)

            self.n = tf.Variable(tf.random_normal([1, self.range.columns]),
                                 name="n_" + self.label)
            self.parameters = [self.n, self.M]
        if self.value:
            self.tensor = self.value
        else:
            self.tensor = tf.add(tf.matmul(self.domain, self.M), self.n)


def generate_V(num_layers, num_features, num_glom_inputs=7):
    weight = np.zeros((num_layers, num_features))
    for i in range(num_layers):
        final_num_input = np.clip(num_glom_inputs, 1, num_features).item()
        indices = np.random.choice(num_features, final_num_input, replace=False)
        weight[i, indices] = 1.
    return weight


def generate_R(num_layers, num_features):
    return tf.random_normal([num_layers, num_features])

def generate_Rb(num_layers):
    return tf.random_uniform(shape=[1, num_layers], minval=0, maxval=2 * np.pi)



class Predicate:
    def __init__(self, label, domain,
                 layers=None,
                 sigma=1.,
                 predefined_V=None,
                 predefined_R=None,
                 predefined_Rb=None):
        self.label = label
        self.domain = domain
        self.num_features = self.domain.columns
        self.num_layers = layers
        self.sigma = sigma

        # AL-MB projection weight V
        if predefined_V is None:
            self.W = tf.Variable(initial_value=generate_V(self.num_layers, self.num_features), dtype=np.float32,
                                 name="rwtn_V" + label, trainable=False)
        else:
            self.W = tf.Variable(initial_value=predefined_V, dtype=np.float32, name="rwtn_V" + label, trainable=False)

        # Random Fourier feature
        if predefined_R is None:
            self.R = tf.Variable(initial_value=generate_R(self.num_layers, self.num_features),
                                 dtype=np.float32,
                                 name="rwtn_R" + label, trainable=False)
        else:
            self.R = tf.Variable(initial_value=predefined_R, dtype=np.float32, name="rwtn_R" + label, trainable=False)

        if predefined_Rb is None:
            self.b = tf.Variable(initial_value=generate_Rb(self.num_layers),
                                 dtype=np.float32,
                                 name="rwtn_R_b" + label, trainable=False)
        else:
            self.b = tf.Variable(initial_value=predefined_Rb, dtype=np.float32, name="rwtn_R_b" + label, trainable=False)

        # Decoder
        self.beta = tf.Variable(tf.random_normal([2 * self.num_layers, 1]),
                                dtype=np.float32,
                                name="rwtn_u" + label)

        self.parameters = [self.W, self.R, self.b, self.beta]



    def tensor(self, domain=None):
        # Original Code
        if domain is None:
            domain = self.domain
        X = domain.tensor

        # Insect brain-inspired feature
        # AL-MB transformation
        XV = tf.matmul(X, tf.transpose(self.W))
        H1 = tf.nn.relu(XV - tf.reduce_mean(XV, axis=1, keepdims=True))

        # Random Fourier feature
        XR = tf.matmul(X, tf.transpose(self.R))
        tr = self.sigma * XR + self.b
        H2 = 1 / np.sqrt(self.num_layers) * np.sqrt(2) * tf.math.cos(tr)

        # Final feature representation
        H = tf.concat([H1, H2], axis=1)
        betaH = tf.matmul(tf.tanh(H), self.beta)
        result = tf.sigmoid(betaH)
        return result


class Literal:
    def __init__(self, polarity, predicate, domain=None):
        self.predicate = predicate
        self.polarity = polarity
        if domain is None:
            self.domain = predicate.domain
        else:
            self.domain = domain
        if polarity:
            self.tensor = predicate.tensor(domain)
        else:
            if default_tnorm == "product" or default_tnorm == "goedel":
                y = tf.equal(predicate.tensor(domain), 0.0)
                self.tensor = tf.cast(y, tf.float32)
            if default_tnorm == "yager2":
                self.tensor = 1 - predicate.tensor(domain)
            if default_tnorm == "luk":
                self.tensor = 1 - predicate.tensor(domain)

        self.parameters = predicate.parameters + domain.parameters


class Clause:
    def __init__(self, literals, label=None, weight=1.0):
        self.weight = weight
        self.label = label
        self.literals = literals
        self.tensor = disjunction_of_literals(self.literals, label=label)
        self.predicates = set([lit.predicate for lit in self.literals])
        self.parameters = [par for lit in literals for par in lit.parameters]


class KnowledgeBase:
    def __init__(self, label, clauses, save_path=""):
        print "defining the knowledge base", label
        self.label = label
        self.clauses = clauses
        self.parameters = [par for cl in self.clauses for par in cl.parameters]
        if not self.clauses:
            self.tensor = tf.constant(1.0)
        else:
            clauses_value_tensor = tf.concat([cl.tensor for cl in clauses], 0)
            if default_clauses_aggregator == "min":
                print "clauses aggregator is min"
                self.tensor = tf.reduce_min(clauses_value_tensor)
            if default_clauses_aggregator == "mean":
                self.tensor = tf.reduce_mean(clauses_value_tensor)
            if default_clauses_aggregator == "hmean":
                self.tensor = tf.div(tf.to_float(tf.size(clauses_value_tensor)),
                                     tf.reduce_sum(tf.reciprocal(clauses_value_tensor), keep_dims=True))
            if default_clauses_aggregator == "wmean":
                weights_tensor = tf.constant([cl.weight for cl in clauses])
                self.tensor = tf.div(tf.reduce_sum(tf.mul(weights_tensor, clauses_value_tensor)),
                                     tf.reduce_sum(weights_tensor))
        if default_positive_fact_penality != 0:
            self.loss = smooth(self.parameters) + \
                        tf.multiply(default_positive_fact_penality, self.penalize_positive_facts()) - \
                        PR(self.tensor)
        else:
            self.loss = smooth(self.parameters) - PR(self.tensor)
        self.save_path = save_path
        self.train_op = train_op(self.loss, default_optimizer)
        self.saver = tf.train.Saver()

    def penalize_positive_facts(self):
        tensor_for_positive_facts = [tf.reduce_sum(Literal(True, lit.predicate, lit.domain).tensor, keep_dims=True) for
                                     cl in self.clauses for lit in cl.literals]
        return tf.reduce_sum(tf.concat(tensor_for_positive_facts, 0))

    def save(self, sess, version=""):
        save_path = self.saver.save(sess, self.save_path + self.label + version + ".ckpt")

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess, feed_dict={}):
        return sess.run(self.train_op, feed_dict)

    def is_nan(self, sess, feed_dict={}):
        return sess.run(tf.is_nan(self.tensor), feed_dict)
