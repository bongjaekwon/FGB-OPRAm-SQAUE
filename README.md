# FGB-OPRAm: Integrating Fuzzy Granular-Ball and OPRAm for Spatial Query Answering in Uncertain Environments


## Overview

This repository contains the implementation and experimental results of **FGB-OPRAm**, a novel framework that integrates **fuzzy granular-ball representations** with the **OPRAm spatial reasoning calculus** to address the problem of **spatial query answering under uncertainty**.

Traditional approaches to spatial queries often assume precise locations, which is unrealistic in real-world applications involving noisy sensors, imprecise location reports, or vague spatial descriptions. FGB-OPRAm overcomes these limitations by modeling spatial objects as fuzzy regions using Gaussian membership functions and extending qualitative spatial reasoning with directional and topological relationships.

https://pypi.org/project/fgb-opram/

---

## Key Concepts

| Component              | Description |
|-----------------------|-------------|
| **Fuzzy Granular-Balls** | Spatial objects are represented as fuzzy regions where the degree of membership reflects positional uncertainty. A Gaussian function defines the fuzzy boundary. |
| **OPRAm Calculus**        | A qualitative spatial reasoning framework extended to handle directional relations like "north of" or "southwest of" in uncertain environments. |
| **Fuzzy Intersection & Containment** | Operations defined on fuzzy granular-balls to compute overlap and inclusion degrees, enabling robust spatial inference under uncertainty. |

---

## Main Contributions

- **FGB-OPRAm Framework**: First integration of fuzzy logic with OPRAm for spatial query answering under uncertainty.
- **Flexible Uncertainty Modeling**: Uses fuzzy granular-balls with Gaussian membership functions to represent spatial imprecision.
- **Adapted Qualitative Reasoning**: Extends OPRAm to operate on fuzzy regions, supporting both distance and directional queries.
- **Comprehensive Evaluation**: Tested on real-world OpenStreetMap data with varying levels of noise and uncertainty, outperforming baselines like KDE, GeoBERT, ST-Transformer, and GNN.

---

## Methodology

### 1. **Fuzzy Granular-Ball Representation**

Spatial objects are modeled as fuzzy regions centered at location $ l = (x, y) $ with radius $ r $:
$$
\mu_B(x', y') = e^{- \frac{d(l, (x', y'))^2}{2\sigma^2}}, \quad \text{where } r = k\sigma
$$

### 2. **Fuzzy OPRAm Adaptation**

Defines operations for fuzzy intersection and containment:
- **Intersection**: $ \mu_C(x, y) = \min(\mu_A(x, y), \mu_B(x, y)) $
- **Containment**: $ \text{Containment}(A, B) = \frac{\int \min(\mu_A, \mu_B)}{\int \mu_A} $

These allow reasoning about spatial relationships even when object positions are uncertain.

---

## Experimental Results (Summary)

| Experiment | Metric | FGB-OPRAm | Crisp Baseline |
|-----------|--------|-----------|----------------|
| Distance-based (50m noise) | F1-score | **0.82** | 0.60 |
| Directional (25° uncertainty) | Accuracy | **10°** | 65° |
| Combined (distance + direction) | F1-score | **0.85** | 0.65 |
| Real-world crime data | F1-score | **0.58** | 0.52 |

FGB-OPRAm consistently outperforms crisp methods and shows strong performance across various types of spatial queries, especially under high uncertainty.

---

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

The experiments were implemented using Python 3.12, leveraging libraries such as:

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

[1] A. Chaves Carniel. “Defining and designing spatial queries: the role of spatial relationships”. In:
Geo-Spatial Information Science 27.6 (2023), pp. 1868–1892. doi: 10 . 1080 / 10095020 . 2022 .
2163924. url: https://doi.org/10.1080/10095020.2022.2163924.

[2] Y. Hui. “Spatial correlation description of deformation object based on fuzzy clustering and geologi-
cal analysis”. In: Geo-Spatial Information Science 3.3 (2000), pp. 69–72. doi: 10.1007/BF02826613.
url: https://doi.org/10.1007/BF02826613.

[3] P. Fisher, A. Comber, and R. Wadsworth. “Approaches to Uncertainty in Spatial Data”. In: Funda-
mentals of Spatial Data Quality. Ed. by R. Devillers and R. Jeansoulin. John Wiley & Sons, 2006.
Chap. 3. doi: 10.1002/9780470612156.ch3. url: https://doi.org/10.1002/9780470612156.
ch3.

[4] K. A. Schmid and A. Z¨ufle. “Representative Query Answers on Uncertain Data”. In: Proceedings of
the 16th International Symposium on Spatial and Temporal Databases (SSTD ’19). New York, NY,
USA: Association for Computing Machinery, 2019, pp. 140–149. doi: 10.1145/3340964.3340974.
url: https://doi.org/10.1145/3340964.3340974.

[5] E. Frentzos, K. Gratsias, and Y. Theodoridis. “On the Effect of Location Uncertainty in Spatial
Querying”. In: IEEE Transactions on Knowledge and Data Engineering 21.3 (2009), pp. 366–383.
doi: 10.1109/TKDE.2008.164. url: https://doi.org/10.1109/TKDE.2008.164.

[6] B. Li, L. Shi, and J. Liu. “Research on spatial data mining based on uncertainty in Government
GIS”. In: Proceedings of the 2010 Seventh International Conference on Fuzzy Systems and Knowl-
edge Discovery. Yantai, China, 2010, pp. 2905–2908. doi: 10.1109/FSKD.2010.5569275. url:
https://doi.org/10.1109/FSKD.2010.5569275.

[7] P. Labourg et al. “Geospatial Uncertainties: A Focus on Intervals and Spatial Models Based on
Inverse Distance Weighting”. In: Proceedings of the 20th International Conference on Information
Processing and Management of Uncertainty in Knowledge-Based Systems (IPMU 2024). Lisboa,
Portugal, 2024, pp. 377–388. doi: 10.1007/978-3-031-74003-9_30. url: https://doi.org/10.
1007/978-3-031-74003-9_30.

[8] Shuyin Xia et al. “Granular-Ball Fuzzy Set and Its Implement in SVM”. In: IEEE Transactions on
Knowledge and Data Engineering 36.11 (2024), pp. 6293–6304. doi: 10.1109/TKDE.2024.3419184.

[9] R. Moratz and J. O. Wallgr¨un. “Spatial reasoning with augmented points: Extending cardinal
directions with local distances”. In: Journal of Spatial Information Science 5 (2012), pp. 1–30.

[10] M. A. Sanchez et al. “Fuzzy granular gravitational clustering algorithm for multivariate data”.
In: Information Sciences 279 (2014), pp. 498–511. doi: 10 . 1016 / j . ins . 2014 . 04 . 005. url:
https://doi.org/10.1016/j.ins.2014.04.005.

[11] G. I. Metternicht. “Categorical fuzziness: A comparison between crisp and fuzzy class boundary
modelling for mapping salt-affected soils using Landsat TM data and a classification based on anion
ratios”. In: Ecological Modelling 168.3 (2003), pp. 371–389. doi: 10.1016/S0304-3800(03)00147-
9. url: https://doi.org/10.1016/S0304-3800(03)00147-9.21

[12] Y. Zheng et al. “Quality and efficiency for kernel density estimates in large data”. In: Proceedings
of the 2013 ACM SIGMOD International Conference on Management of Data (SIGMOD ’13).
New York, NY, USA: Association for Computing Machinery, 2013, pp. 433–444. doi: 10.1145/
2463676.2465319. url: https://doi.org/10.1145/2463676.2465319.

[13] Y. Gao et al. “GeoBERT: Pre-Training Geospatial Representation Learning on Point-of-Interest”.
In: Applied Sciences 12.24 (2022), p. 12942. doi: 10.3390/app122412942. url: https://doi.
org/10.3390/app122412942.

[14] S. De Sabbata and P. Liu. “A graph neural network framework for spatial geodemographic clas-
sification”. In: International Journal of Geographical Information Science 37.12 (2023), pp. 2464–
2486. doi: 10.1080/13658816.2023.2254382. url: https://doi.org/10.1080/13658816.2023.
2254382.

[15] J. Paglia, J. Eidsvik, and J. Karvanen. “Efficient spatial designs using Hausdorff distances and
Bayesian optimization”. In: Scandinavian Journal of Statistics 49.3 (2022), pp. 1060–1084. doi:
10.1111/sjos.12554. url: https://doi.org/10.1111/sjos.12554.

[16] P. Mishra et al. “Application of student’s t-test, analysis of variance, and covariance”. In: Annals
of Cardiac Anaesthesia 22.4 (2019), pp. 407–411. doi: 10.4103/aca.ACA_94_19. url: https:
//doi.org/10.4103/aca.ACA_94_19.

[17] L. A. Zadeh. “Fuzzy sets”. In: Information and Control 8.3 (1965), pp. 338–353. doi: 10.1016/
S0019-9958(65)90241-X. url: https://doi.org/10.1016/S0019-9958(65)90241-X.

[18] D. A. Randell and A. G. Cohn. “Modelling topological and metrical properties of physical pro-
cesses”. In: Proceedings of the First International Conference on the Principles of Knowledge Rep-
resentation and Reasoning. Ed. by R. J. Brachman, H. J. Levesque, and R. Reiter. Los Altos, CA:
Morgan Kaufmann, 1989, pp. 55–66.

[19] D. A. Randell, Z. Cui, and A. G. Cohn. “A spatial logic based on regions and connection”. In:
Proceedings of the Third International Conference on the Principles of Knowledge Representation
and Reasoning. Ed. by B. Nebel, C. Rich, and W. Swartout. Los Altos, CA: Morgan Kaufmann,
1992, pp. 165–176.

[20] C. H. Dorr, L. J. Latecki, and R. Moratz. “Shape Similarity Based on the Qualitative Spatial
Reasoning Calculus eOPRAm”. In: Spatial Information Theory. COSIT 2015. Ed. by S. Fabrikant
et al. Vol. 9368. Lecture Notes in Computer Science. Cham: Springer, 2015, pp. 95–112. doi:
10.1007/978-3-319-23374-1_7. url: https://doi.org/10.1007/978-3-319-23374-1_7.

[21] A. Zufle et al. “Managing Uncertainty in Evolving Geo-Spatial Data”. In: Proceedings of the 2020
21st IEEE International Conference on Mobile Data Management (MDM 2020). Article 9162308.
IEEE, 2020, pp. 5–8. doi: 10.1109/MDM48529.2020.00021. url: https://doi.org/10.1109/
MDM48529.2020.00021.

[22] R. Cheng and J. Chen. “Probabilistic Spatial Queries”. In: Encyclopedia of Database Systems.
Springer, 2018, pp. 2847–2852. doi: 10.1007/978-1-4614-8265-9_276. url: https://doi.org/
10.1007/978-1-4614-8265-9_276.

[23] W. Liu, J. Wang, and M. T. ¨Ozsu. “Spatial Query Processing for Fuzzy Objects”. In: The VLDB
Journal 21.6 (2012), pp. 729–751. doi: 10.1007/s00778-012-0266-x. url: https://doi.org/
10.1007/s00778-012-0266-x.

[24] M. F. Worboys. “Fuzzy Set Approaches to Model Uncertainty in Spatial Data and Geographic
Information Systems”. In: Computing with Words in Information/Intelligent Systems 2. Springer,
1998, pp. 345–367. doi: 10.1007/978-3-7908-1872-7_16. url: https://doi.org/10.1007/978-
3-7908-1872-7_16.

[25] M. Abdar et al. A Review of Uncertainty Quantification in Deep Learning: Techniques, Applications
and Challenges. arXiv preprint arXiv:2011.06225. 2020. url: https://arxiv.org/abs/2011.
06225.

[26] C. B. Connor and L. J. Connor. “Estimating spatial density with kernel methods”. In: Volcanic
and Tectonic Hazard Assessment for Nuclear Facilities. Ed. by C. B. Connor, N. A. Chapman, and
L. J. Connor. Cambridge University Press, 2009, pp. 346–368.

[27] N. Langren´e and X. Warin. “Fast and Stable Multivariate Kernel Density Estimation by Fast Sum
Updating”. In: Journal of Computational and Graphical Statistics 28.3 (2019), pp. 596–608. doi:
10.1080/10618600.2018.1549052. url: https://doi.org/10.1080/10618600.2018.1549052.

[28] M. Schneider. “Uncertainty Management for Spatial Data in Databases: Fuzzy Spatial Data Types”.
In: Proceedings of the 6th International Symposium on Advances in Spatial Databases (SSD ’99).
Berlin, Heidelberg: Springer-Verlag, 1999, pp. 330–351.22

[29] A. Pauly and M. Schneider. “Spatial Vagueness and Imprecision in Databases”. In: Proceedings of
the 2008 ACM Symposium on Applied Computing (SAC ’08). New York, NY, USA: Association
for Computing Machinery, 2008, pp. 875–879. doi: 10 . 1145 / 1363686 . 1363888. url: https :
//doi.org/10.1145/1363686.1363888.

[30] E. Clementini and P. Di Felice. “A Spatial Model for Complex Objects with a Broad Boundary
Supporting Queries on Uncertain Data”. In: Data & Knowledge Engineering 37.3 (2001), pp. 285–
305. doi: 10 . 1016 / S0169 - 023X(01 ) 00010 - 6. url: https : / / doi . org / 10 . 1016 / S0169 -
023X(01)00010-6.

[31] R. Kontchakov et al. “Spatial logics with connectedness predicates”. In: Logical Methods in Com-
puter Science 6 (2010). url: https://doi.org/10.2168/LMCS-6(3:7)2010.

[32] S. Schockaert, M. De Cock, and E. E. Kerre. “Spatial reasoning in a fuzzy region connection
calculus”. In: Artificial Intelligence 173.2 (2009), pp. 258–298. doi: 10.1016/j.artint.2008.10.
009. url: https://doi.org/10.1016/j.artint.2008.10.009.

[33] W. Chen, F. Wang, and H. Sun. “S2TNet: Spatio-Temporal Transformer Networks for Trajectory
Prediction in Autonomous Driving”. In: Proceedings of The 13th Asian Conference on Machine
Learning. Vol. 157. Proceedings of Machine Learning Research. 2021, pp. 454–469. url: https:
//proceedings.mlr.press/v157/chen21a.html.

[34] Jiang Xie et al. “MGNR: A Multi-Granularity Neighbor Relationship and Its Application in KNN
Classification and Clustering Methods”. In: IEEE Transactions on Pattern Analysis and Machine
Intelligence 46.12 (2024), pp. 7956–7972. doi: 10.1109/TPAMI.2024.3400281.

[35] Jiashun Liu et al. “Unlock the cognitive generalization of deep reinforcement learning via granular
ball representation”. In: Proceedings of the 41st International Conference on Machine Learning.
ICML’24. Vienna, Austria: JMLR.org, 2024.

[36] Xuemei Cao et al. “Open Continual Feature Selection via Granular-Ball Knowledge Transfer”. In:
IEEE Transactions on Knowledge & Data Engineering 36.12 (2024), pp. 8967–8980. issn: 1558-
2191. doi: 10.1109/TKDE.2024.3428485. url: https://doi.ieeecomputersociety.org/10.1109/TKDE.2024.3428485.

[37] Yihao et al. “GRICP: Granular-Ball Iterative Closest Point with Multikernel Correntropy for Point
Cloud Fine Registration”. In: Proceedings of the AAAI Conference on Artificial Intelligence 39.2
(2025), pp. 1710–1718. doi: 10.1609/aaai.v39i2.32164. url: https://ojs.aaai.org/index.php/AAAI/article/view/32164.

[38] Peng Su et al. “Multi-view Granular-ball Contrastive Clustering”. In: Proceedings of the AAAI
Conference on Artificial Intelligence 39.19 (2025), pp. 20637–20645. doi: 10.1609/aaai.v39i19.
34274. url: https://ojs.aaai.org/index.php/AAAI/article/view/34274
