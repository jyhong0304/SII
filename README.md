# Randomly Weighted Feature Network for Neuro-Symbolic Relational Learning

This repository is the official implementation of [Random Weighted Feature Netowrks~(RWFNs)](https://ceur-ws.org/Vol-2986/paper10.pdf). All source codes were
implemented in Python 2.7.

## Abstract

Knowledge Representation and Reasoning (KRR) is one of the well-known fields in computer science that try to understand,
reason, and interpret knowledge as human beings efficiently do. Since many logical formalism and reasoning methods in
the area have shown the capability of higher-order learning, such as abstract concept learning, integrating artificial
neural networks with KRR methods to handle complex and practical tasks has received much attention. For example, Neural
Tensor Networks (NTNs) are neural network models capable of transforming symbolic representations into vector spaces
where reasoning can be performed through matrix computation and are used in Logic Tensor Networks (LTNs) that enable to
embed first-order logic symbols such as constants, facts, and rules into real-valued tensors. However, such capability
in higher-order learning is not exclusive to humans. Insects, such as fruit flies and honey bees, can solve simple
associative learning tasks and learn abstract concepts such as 'sameness' and 'difference,' which is viewed as a
higher-order cognitive function and typically thought to depend on top-down neocortical processing. With the inspiration
from insect nervous systems, we propose a Randomly Weighted Feature Network (RWFN), which incorporates randomly drawn,
untrained weights in an encoder with an adapted linear model as a decoder. Specifically, RWFN is a single-hidden-layer
neural network with unique latent representations in the hidden layer, derived by integrating the input transformation
between input neurons and Kenyon cells in the insect brain and random Fourier features used for better representing
complex relationships between input using kernel approximation. Based on this unique representation, by training only a
linear model, RWFNs can effectively learn the degree of relationship among inputs. We test RWFNs over LTNs for Semantic
Image Interpretation (SII) tasks that have been used as a representative example of how LTNs utilize reasoning over
first-order logic to surpass the performance of solely data-driven methods. We demonstrate that compared to LTNs, RWFNs
can achieve better or similar performance for the object classification and the detection of the relevant *part-of*
relations between objects in the SII tasks while using much fewer learnable parameters (1:151 ratio) and a faster
learning process (1:2 ratio of running speed). Furthermore, we show that because the randomized weights do not depend on
the data, several decoders can share a single encoder, giving RWFNs a unique economy of spatial scale for simultaneous
classification tasks.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Execution

We directly compare the performances between our method and LTNs for Semantic Image Interpretation (SII).

Original source code of LTNs for SII and dataset are available [here](https://gitlab.fbk.eu/donadello/LTN_IJCAI17).

All the details of best hyperparameters for RWFNs are described in the paper.

1. Select the dataset that you want to train in pascalpart.py

2. To train, run the following command:

```setup
python train.py
```

3. To evaluate, run the following command:
```setup
python evaluate.py
```

## Results

All trained models, predefined weights, and figures are available.

- AUC of RWFNs and LTNs for object type classification using indoor object data
![AUC of RWFNs and LTNs for object type classification](https://github.com/jyhong0304/SII/blob/master/figures/indoor_object_type_classification.png)

- AUC of RWFNs and LTNs for detecting part-of relation using indoor object data
![AUC of RWFNs and LTNs for detecting part-of relation](https://github.com/jyhong0304/SII/blob/master/figures/indoor_part-of_detection.png)

- The comparison of running time, including data configuration and training time, for RWTNs and LTNs
![The comparison of running time, including data configuration and training time, for RWTNs and LTNs](https://github.com/jyhong0304/SII/blob/master/figures/running_time_comparison.png)

## Contributing

All content in this repository is licensed under the MIT license.

