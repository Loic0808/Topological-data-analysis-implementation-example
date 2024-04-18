# Topological data analysis implementation example

Below is an easy implementation of a labelling algorithm which uses persistant homology.

## Introduction

I implemented this code during an internship, I was really interested in topological data analysis and tried to use it for a classification task. The code allows, when given some labelled vectors (training data), to predict the labels of the vectors around it with high precision (testing data). The goal of this code is to make the algorithm accessible to people who don't necessarily have a strong background in algebraic topology. For more information see the paper below.

## Implementation of the paper CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS

This project implements the pseudocode presented in the paper titled "CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS" by Rolando Kindelan, José Frías, Mauricio Cerda
 and Nancy Hitschfeld. The paper can be found arXiv:2102.03709.

## Example

### Description
The dataset I created consists of three concentric rings, each containing 100 data points. Additionally, there are three points representing a separate class, located in the upper left corner near the outer ring. Within each ring, the 100 points are randomly distributed, each representing a specific class. The three points in the upper left corner were added to test the algorithm on datasets with very few examples, as machine learning algorithms often struggle with limited data. <br>

For each ring, 20 points were labeled (classes 1 to 3), while for the three outer points, only one was labeled (class 4), and the rest were left as unlabeled data (class 0), which the algorithm needs to classify. <br>
<img width="324" alt="image001" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/9d2506d5-8896-4597-9a56-bccd0171d2aa"> <br>
After running the algorithm once on the dataset, we get the following results. The points in light green, grey, yellow and pink are the points which have been added in the first run of the algorithm. <br>
<img width="324" alt="image002" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/428fec38-9830-4b6d-97e7-242e010be719"> <br>
After running it multiple times, the results are:<br>
<img width="324" alt="image003" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/3f6f6429-26b0-4c36-bff5-d57dca0af766"><br>
We see that almost all points have been correctly classified. This result was obtained using the following list of epsilon parameters: epsilon_list = [0.2, 0.3, 0.3, 0.3, 0.4, 0.3, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1]. In thius example I used intuition to implement it, but in general one doesn't have such intuition. This is why I improved the algorithm so, that multiple such lists are generated at random and used for the algorithm. The results for each list are stored in a dataset and at the end we take the label which appeared most. This ensures that even if taken randomly, the epsilon allow to choose the best label. <br>

I also modified the function which creates the Link for faster computations, by only looking at the simplices which are near the point we look at.

### Usage

This algorithm can be utilized for labeling tasks, it works especially well with unbalanced data. Its advantage over other algorithms like k-nearest neighbors or k-means is its consideration of the mathematical structure within the dataset.

One specific example where I employed the algorithm was for text classification into certain categories. Initially, I attempted to use LLMs, but they failed when the classes had too few examples. The algorithm performed well, with the only issues arising from the vectorization of the text.

## Installation

I used the following packages: gudhi for the implementation of the simplices and VR simplicial complexes, sympy for the generators which represent the labels (one can also use sagemath) and ripser for the visualization of the data using homological features. The last package is not needed to run the algorithm.

    $ capture
    $ pip install ripser
    $ pip install gudhi
    $ pip install sympy

## Contributing

If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](link_to_contributing_file).

## License

This project is licensed under the [License Name] - see the [LICENSE](link_to_license_file) file for details.

## Acknowledgements

- Acknowledge any individuals or organizations that have contributed to the project.
- Optionally, include any citations or references to external resources that were helpful in the development of your project.
