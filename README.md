# Topological data analysis implementation example

Below is an easy implementation of a labelling algorithm which uses persistant homology.

## Introduction

I implemented this code during an internship, I was really interested in topological data analysis and tried to use it for a classification task. The code allows, when given some labelled vectors (training data), to predict the labels of the vectors around it with high precision (testing data). The goal of this code is to make the algorithm accessible to people who don't necessarily have a strong background in algebraic topology. 

## Implementation of the paper CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS

This project implements the pseudocode presented in the paper titled "CLASSIFICATION BASED ON TOPOLOGICAL DATA ANALYSIS" by Rolando Kindelan, José Frías, Mauricio Cerda
 and Nancy Hitschfeld. The paper can be found arXiv:2102.03709.

## Example

### Description

The dataset I created consists of 3 concentric rings which 100 datapoint each and of 3 points which represent a sepearate class which are located in the upper left corner near the outer ring. Each ring consists of 100 points randomly distributed in the ring representing a specific class. The 3 points in the upper left corner where added to test the algorithm on data with very few examples (on such datapoints machine learning algorithm often fail due to the little amount of data). For each ring 20 points where labelled (class 1 to 3), for the 3 outter points only one was labelled (class 4) and the rest was unlabelled data (class 0) which the algorithm needs to classify. 
<img width="324" alt="image001" src="https://github.com/Loic0808/Topological-data-analysis-implementation-example/assets/162875696/9d2506d5-8896-4597-9a56-bccd0171d2aa">



### Usage

Explain how to use the provided example. Include any necessary instructions for running the code and interpreting the results.

## Installation

Describe how to install any dependencies required to run the code.

## Usage

Explain how to use the code in your repository. Include examples if applicable.

## Contributing

If you'd like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](link_to_contributing_file).

## License

This project is licensed under the [License Name] - see the [LICENSE](link_to_license_file) file for details.

## Acknowledgements

- Acknowledge any individuals or organizations that have contributed to the project.
- Optionally, include any citations or references to external resources that were helpful in the development of your project.
