# Topological data analysis implementation example

Below is an easy implementation of a labelling algorithm which uses persistant homology.

## Introduction

I implemented this code during an internship, I was really interested in topological data analysis and tried to use it for a classification task. The code allows, when given some labelled vectors (training data), to predict the labels of the vectors around it with high precision (testing data). The goal of this code is to make the algorithm accessible to people who don't necessarily have a strong background in algebraic topology. 

## Implementation of the paper CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS

This project implements the pseudocode presented in the paper titled "CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS" by Rolando Kindelan, José Frías, Mauricio Cerda
 and Nancy Hitschfeld. The paper can be found arXiv:2102.03709.

## Example

### Description

The dataset I created consists of 3 concentric rings which 100 datapoint each and of 3 points which represent a sepearate class which are located in the upper left corner near the outer ring. Each ring consists of 100 points randomly distributed in the ring representing a specific class. The 3 points in the upper left corner where added to test the algorithm on data with very few examples (on such datapoints machine learning algorithm often fail due to the little amount of data). For each ring 20 points where labelled (class 1 to 3), for the 3 outter points only one was labelled (class 4) and the rest was unlabelled data (class 0) which the algorithm needs to classify. <br>
<img width="324" alt="image001" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/9d2506d5-8896-4597-9a56-bccd0171d2aa"> <br>
After running the algorithm once on the dataset, we get the following results. The points in light green, grey, yellow and pink are the points which have been added in the first run of the algorithm. <br>
<img width="324" alt="image002" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/428fec38-9830-4b6d-97e7-242e010be719"> <br>
After running it multiple times, the results are:<br>
<img width="324" alt="image003" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/3f6f6429-26b0-4c36-bff5-d57dca0af766"><br>
We see that almost all points have been correctly classified.

### Usage

This algorithm can be used for labelling tasks. It works especially well for data with unbalanced data. The advantage to other algorithms like k-nearest neighbours or to k-mean is that this algorithm takes into account the mathematical structure of the dataset.<br>

One specific example I used the algorithm for was for text classification to certain numbers. I first tried to use LLM's but the failed when the classes had to few examples. The algorithm worked well and the only problems which appeared came from the vectorization of the text.

## Installation

I used the following packages: gudhi for the implementation of the simplices and VR simplicial complexes, sympy for the generators which represent the labells (one can also use sagemath) and ripser for the visualization of the data using homological features. The last package is not needed to run the algorithm.

## Usage

Explain how to use the code in your repository. Include examples if applicable.

## Contributing

If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](link_to_contributing_file).

## License

This project is licensed under the [License Name] - see the [LICENSE](link_to_license_file) file for details.

## Acknowledgements

- Acknowledge any individuals or organizations that have contributed to the project.
- Optionally, include any citations or references to external resources that were helpful in the development of your project.
