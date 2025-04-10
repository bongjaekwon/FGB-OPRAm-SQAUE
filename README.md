
# Fuzzy Granular Balls and OPRAm for Spatial Query Answering in Uncertain Environments

## Overview

This repository contains the code and experimental results for a novel framework that integrates fuzzy granular balls with the OPRAm spatial reasoning calculus to address the problem of spatial query answering in uncertain environments. Traditional methods often fail to capture the inherent imprecision of spatial data, particularly in real-world applications dealing with noisy sensor data, imprecise location reports, or vague spatial descriptions. This framework provides a more robust and flexible solution by leveraging the strengths of both fuzzy logic and qualitative spatial reasoning.

## Key Concepts

*   **Fuzzy Granular Balls:** Spatial objects are represented as fuzzy regions, where the degree of membership reflects the level of uncertainty associated with the object's location. A Gaussian membership function is used to define the fuzzy region, with the radius of the granular ball directly related to the level of uncertainty.

*   **OPRAm Spatial Reasoning Calculus:** A qualitative spatial reasoning calculus that provides a formal framework for representing and reasoning about spatial relationships between objects. OPRAm extends the Region Connection Calculus (RCC-8) by incorporating directional information.

*   **Fuzzy OPRAm:** An adaptation of the OPRAm calculus to operate on fuzzy granular balls, defining fuzzy intersection and containment operations that enable reasoning about spatial relationships under uncertainty.

## Methodology

1.  **Fuzzy Granular Ball Representation:** Spatial objects are represented as fuzzy granular balls centered at a location `l = (x, y)` with radius `r`. The membership function `μ_B(x', y')` assigns a degree of membership to each point `(x', y')`:

    ```
    μ_B(x', y') = e^(-(d(l, (x', y'))^2) / (2σ^2))
    ```

    where `d(l, (x', y'))` is the Euclidean distance and `σ` is the standard deviation, related to the radius `r` by `r = kσ` (where `k` is a constant, typically 2).

2.  **OPRAm Adaptation:** Fuzzy intersection and containment operations are defined for fuzzy granular balls. For example, the fuzzy intersection of two fuzzy granular balls `A` and `B` is defined as:

    ```
    μ_C(x, y) = min(μ_A(x, y), μ_B(x, y))
    ```

    The fuzzy containment of `A` in `B` is defined as:

    ```
    Containment(A, B) = (∫ min(μ_A(x, y), μ_B(x, y)) dx dy) / (∫ μ_A(x, y) dx dy)
    ```

    Fuzzy versions of OPRAm relations (e.g., "north of") are defined based on these operations.

## Experimental Setup

The framework was evaluated through a series of experiments using OpenStreetMap data. The experiments were designed to assess the performance of Fuzzy OPRAm under varying conditions of noise, uncertainty, and query complexity.

### Baseline Models

The performance of Fuzzy OPRAm was compared against the following baseline models:

*   **Crisp Region Baseline:** Represents objects as crisp regions and performs spatial query answering using standard spatial operators.
*   **Probabilistic Baseline (KDE):** Employs kernel density estimation to model spatial uncertainty.
*   **GeoBERT:** A BERT model fine-tuned on spatial text.
*   **ST-Transformer:** A spatial-temporal transformer model.
*   **GNN:** A Graph Neural Network model.

### Datasets

*   **Experiment 1:** Restaurants in Gangnam, Seoul, South Korea (OpenStreetMap).  Gaussian noise added to simulate location uncertainty.
*   **Experiment 2:** Coffee shops and libraries in New York (OpenStreetMap). Angular uncertainty introduced by rotating coffee shop locations.
*   **Experiment 3:** Pharmacies, hospitals, and parks in Chicago (OpenStreetMap). Gaussian noise added to all POI locations.
*   **Experiment 4:** Reported crime locations in San Francisco (public crime database) and school locations (OpenStreetMap).  Real-world imprecise location data.
*   **Experiment 5:** Generated datasets of POIs with varying sizes (10,000 to 10,000,000 data points) and noise levels.

### Queries

Various spatial queries were used, including:

*   Distance-based queries (e.g., "Find all restaurants within 500 meters of a given hotel")
*   Direction-based queries (e.g., "Find all coffee shops north of the library")
*   Combined distance and direction queries (e.g., "Find all pharmacies within 1 km southwest of the hospital and north of the park")

### Evaluation Metrics

*   Precision
*   Recall
*   F1-score
*   Hausdorff distance
*   Directional accuracy
*   Computational time
*   Memory Usage

### Implementation

The experiments were implemented using Python 3.8, leveraging libraries such as:

*   GeoPandas
*   Shapely
*   Scikit-fuzzy
*   Scikit-learn
*   NumPy
*   SciPy
*   PyTorch
*   TensorFlow
*   Transformers
*   PyTorch Geometric
*   DGL
*   Statsmodels
*   Matplotlib
*   Seaborn

All experiments were run on a machine equipped with an NVIDIA Geforce 3080 GPU, and 32 GB of RAM, running Fedora 41.

## Results

The results of the experiments demonstrate that Fuzzy OPRAm offers a significant improvement in accuracy for spatial query answering in uncertain environments, particularly when dealing with noisy or imprecise location data.

*   **Experiment 1:** Fuzzy OPRAm consistently outperformed the Crisp Region Baseline across all noise levels. For example, at a 50m noise level, Fuzzy OPRAm achieved an F1-score of 0.82 compared to the Crisp Region Baseline's 0.60 (p < 0.01).
*   **Experiment 2:** Fuzzy OPRAm significantly outperformed the Crisp Region Baseline in terms of directional accuracy. At an angular uncertainty of 25 degrees, Fuzzy OPRAm maintained an accuracy of 10 degrees, compared to the Crisp Region Baseline's accuracy of 65 degrees (p < 0.001).
*   **Experiment 3:** Fuzzy OPRAm achieved an F1-score of 0.85, a directional accuracy of 8 degrees, and a Hausdorff distance of 75 meters, outperforming the Crisp Region Baseline (p < 0.001).
*   **Experiment 4:** All models performed relatively poorly on the real-world imprecise crime location data, indicating the challenges of handling highly complex and heterogeneous real-world datasets.
*   **Experiment 5:** Fuzzy OPRAm's computational time was higher than the Crisp Region Baseline, but the use of a spatial index (R-tree) significantly reduced the computational time. Fuzzy OPRAm scaled more gracefully than KDE for very large datasets.

**Table 1: F1-score for distance-based queries with varying noise levels**

| Noise Level (m) | Fuzzy OPRAm (F1) | Crisp Baseline (F1) | KDE (F1) | GeoBERT (F1) | ST-Transformer (F1) | GNN (F1) |
| :-------------- | :--------------- | :------------------ | :------- | :---------- | :---------------- | :------- |
| 10              | 0.92             | 0.85                | 0.91     | 0.87        | 0.86              | 0.88     |
| 20              | 0.89             | 0.78                | 0.87     | 0.84        | 0.83              | 0.85     |
| 30              | 0.86             | 0.71                | 0.82     | 0.81        | 0.80              | 0.82     |
| 40              | 0.84             | 0.65                | 0.78     | 0.79        | 0.78              | 0.79     |
| 50              | 0.82             | 0.60                | 0.75     | 0.78        | 0.76              | 0.77     |

**Table 2: Directional accuracy for direction-based queries with varying angular uncertainty**

| Angular Uncertainty (degrees) | Fuzzy OPRAm (Directional Accuracy) | Crisp Baseline (Directional Accuracy) | KDE (Directional Accuracy) | GeoBERT (Directional Accuracy) | ST-Transformer (Directional Accuracy) | GNN (Directional Accuracy) |
| :-------------------------- | :--------------------------------- | :------------------------------------ | :----------------------- | :-------------------------- | :------------------------------------ | :----------------------- |
| 5                           | 3                                  | 15                                    | 8                        | 7                           | 9                                     | 8                        |
| 10                          | 5                                  | 25                                    | 15                       | 10                          | 11                                    | 10                       |
| 15                          | 7                                  | 35                                    | 22                       | 11                          | 12                                    | 11                       |
| 20                          | 9                                  | 50                                    | 30                       | 12                          | 13                                    | 12                       |
| 25                          | 10                                 | 65                                    | 40                       | 12                          | 14                                    | 13                       |

**Table 3: Performance on combined distance and direction queries**

| Metric              | Fuzzy OPRAm | Crisp Baseline | KDE  | GeoBERT | ST-Transformer | GNN  |
| :------------------ | :---------- | :------------- | :--- | :-------- | :--------------- | :--- |
| F1-score            | 0.85        | 0.65           | 0.75 | 0.77      | 0.79             | 0.80 |
| Directional Accuracy | 8           | 40             | 25   | 12        | 14               | 13   |
| Hausdorff Distance  | 75          | 150            | 100  | 90        | 85               | 88   |

**Table 4: Performance on real-world imprecise location data**

| Metric   | Fuzzy OPRAm | Crisp Baseline | KDE  | GeoBERT | ST-Transformer | GNN  |
| :------- | :---------- | :------------- | :--- | :-------- | :--------------- | :--- |
| F1-score | 0.58        | 0.52           | 0.55 | 0.56      | 0.57             | 0.57 |


## Future Work

*   Explore techniques to reduce the computational cost of Fuzzy OPRAm, such as indexing and parallelization.
*   Investigate the application of the approach to other domains, such as environmental monitoring and disaster response.
*   Utilize alternative membership functions for the fuzzy granular balls.
*   Address attribute uncertainty and relational uncertainty in addition to positional uncertainty.
*   Evaluate Fuzzy OPRAm on a wider range of datasets and queries.
*   Integrate Fuzzy OPRAm with existing GIS software and spatial databases.
*   Explore adaptive membership functions for fuzzy granular balls.
*   Integrate Fuzzy OPRAm with other spatial reasoning calculi (e.g., RCC-8, CDC).
*   Develop more efficient algorithms for fuzzy spatial reasoning (e.g., sampling techniques, GPU acceleration).

## References

Chaves Carniel, A. (2023). Defining and designing spatial queries: the role of spatial relationships. Geo-Spatial Information Science, 27(6), 1868–1892. https://doi.org/10.1080/10095020.2022.2163924

Hui, Y. (2000). Spatial correlation description of deformation object based on fuzzy clustering and geological analysis. Geo-Spatial Information Science, 3(3), 69–72. https://doi.org/10.1007/BF02826613

Fisher, P., Comber, A., & Wadsworth, R. (2006). Approaches to Uncertainty in Spatial Data. In R. Devillers & R. Jeansoulin (Eds.), Fundamentals of Spatial Data Quality (Chapter 3). https://doi.org/10.1002/9780470612156.ch3

Schmid, K. A., & Züfle, A. (2019). Representative Query Answers on Uncertain Data. In Proceedings of the 16th International Symposium on Spatial and Temporal Databases (SSTD '19), 140–149. https://doi.org/10.1145/3340964.3340974

Frentzos, E., Gratsias, K., & Theodoridis, Y. (2009). On the Effect of Location Uncertainty in Spatial Querying. IEEE Transactions on Knowledge and Data Engineering, 21(3), 366–383. https://doi.org/10.1109/TKDE.2008.164

Li, B., Shi, L., & Liu, J. (2010). Research on spatial data mining based on uncertainty in Government GIS. In Proceedings of the 2010 Seventh International Conference on Fuzzy Systems and Knowledge Discovery, 2905–2908. https://doi.org/10.1109/FSKD.2010.5569275

Labourg, P., Destercke, S., Guillaume, R., Rohmer, J., Quost, B., et al. (2024). Geospatial Uncertainties: A Focus on Intervals and Spatial Models Based on Inverse Distance Weighting. In Proceedings of IPMU 2024, 377–388. https://doi.org/10.1007/978-3-031-74003-9_30

Xia, S., Lian, X., & Shao, Y. (2022). Fuzzy Granular-Ball Computing Framework and Its Implementation in SVM. CoRR, abs/2210.11675. https://doi.org/10.48550/arXiv.2210.11675

Moratz, R., & Wallgrün, J. O. (2012). Spatial reasoning with augmented points: Extending cardinal directions with local distances. Journal of Spatial Information Science, 5, 1–30.

Sanchez, M. A., Castillo, O., Castro, J. R., & Melin, P. (2014). Fuzzy granular gravitational clustering algorithm for multivariate data. Information Sciences, 279, 498–511. https://doi.org/10.1016/j.ins.2014.04.005

Metternicht, G. I. (2003). Categorical fuzziness: A comparison between crisp and fuzzy class boundary modelling for mapping salt-affected soils. Ecological Modelling, 168(3), 371–389. https://doi.org/10.1016/S0304-3800(03)00147-9

Zheng, Y., Jestes, J., Phillips, J. M., & Li, F. (2013). Quality and efficiency for kernel density estimates in large data. In Proceedings of SIGMOD '13, 433–444. https://doi.org/10.1145/2463676.2465319

Gao, Y., Xiong, Y., Wang, S., & Wang, H. (2022). GeoBERT: Pre-Training Geospatial Representation Learning on Point-of-Interest. Applied Sciences, 12(24), 12942. https://doi.org/10.3390/app122412942

De Sabbata, S., & Liu, P. (2023). A graph neural network framework for spatial geodemographic classification. IJGIS, 37(12), 2464–2486. https://doi.org/10.1080/13658816.2023.2254382

Paglia, J., Eidsvik, J., & Karvanen, J. (2022). Efficient spatial designs using Hausdorff distances and Bayesian optimization. Scandinavian Journal of Statistics, 49(3), 1060–1084. https://doi.org/10.1111/sjos.12554

Mishra, P., Singh, U., Pandey, C. M., Mishra, P., & Pandey, G. (2019). Application of student's t-test, analysis of variance, and covariance. Annals of Cardiac Anaesthesia, 22(4), 407–411. https://doi.org/10.4103/aca.ACA_94_19

Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X

Randell, D. A., & Cohn, A. G. (1989). Modelling topological and metrical properties of physical processes. In KR '89, 55–66.

Randell, D. A., Cui, Z., & Cohn, A. G. (1992). A spatial logic based on regions and connection. In KR '92, 165–176.

Dorr, C. H., Latecki, L. J., & Moratz, R. (2015). Shape Similarity Based on the Qualitative Spatial Reasoning Calculus eOPRAm. In COSIT 2015, LNCS 9368, 97–115. https://doi.org/10.1007/978-3-319-23374-1_7

Zufle, A., Trajcevski, G., Pfoser, D., & Kim, J. S. (2020). Managing Uncertainty in Evolving Geo-Spatial Data. In MDM 2020, 5–8. https://doi.org/10.1109/MDM48529.2020.00021

Cheng, R., & Chen, J. (2018). Probabilistic Spatial Queries. In Encyclopedia of Database Systems, 2847–2852. https://doi.org/10.1007/978-1-4614-8265-9_276

Liu, W., Wang, J., & Özsu, M. T. (2012). Spatial Query Processing for Fuzzy Objects. The VLDB Journal, 21(6), 729–751. https://doi.org/10.1007/s00778-012-0266-x

Worboys, M. F. (1998). Fuzzy Set Approaches to Model Uncertainty in Spatial Data and GIS. In Computing with Words in Information/Intelligent Systems 2, 345–367. https://doi.org/10.1007/978-3-7908-1872-7_16

Abdar, M., et al. (2020). A Review of Uncertainty Quantification in Deep Learning: Techniques, Applications and Challenges. arXiv:2011.06225. https://arxiv.org/abs/2011.06225

Connor, C. B., & Connor, L. J. (2009). Estimating spatial density with kernel methods. In Volcanic and Tectonic Hazard Assessment for Nuclear Facilities, 346–368.

Langrené, N., & Warin, X. (2019). Fast and Stable Multivariate Kernel Density Estimation by Fast Sum Updating. Journal of Computational and Graphical Statistics, 28(3), 596–608. https://doi.org/10.1080/10618600.2018.1549052

Schneider, M. (1999). Uncertainty Management for Spatial Data in Databases: Fuzzy Spatial Data Types. In SSD '99, 330–351.

Pauly, A., & Schneider, M. (2008). Spatial vagueness and imprecision in databases. In SAC '08, 875–879. https://doi.org/10.1145/1363686.1363888

Clementini, E., & Di Felice, P. (2001). A spatial model for complex objects with a broad boundary supporting queries on uncertain data. Data & Knowledge Engineering, 37(3), 285–305. https://doi.org/10.1016/S0169-023X(01)00010-6

Kontchakov, R., Pratt-Hartmann, I., Wolter, F., & Zakharyaschev, M. (2010). Spatial logics with connectedness predicates. Logical Methods in Computer Science, 6.

Schockaert, S., De Cock, M., & Kerre, E. E. (2009). Spatial reasoning in a fuzzy region connection calculus. Artificial Intelligence, 173(2), 258–298. https://doi.org/10.1016/j.artint.2008.10.009

Chen, W., Wang, F., & Sun, H. (2021). S2TNet: Spatio-Temporal Transformer Networks for Trajectory Prediction in Autonomous Driving. In ACML 2021, PMLR 157, 454–469. https://proceedings.mlr.press/v157/chen21a.html
