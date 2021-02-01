## Overview

This repository is related to the _RegBL completion_ research project. The _STDL_ was contacted by the Swiss Federal Statistical Office (_OFS_) to determine in which extend it could be possible to complete the construction date of Swiss buildings based on the analysis of a temporal sequence of the Swiss federal maps produced by _swisstopo_. With an initial target of _80%_ of correct guesses, the goal of this research project was to demonstrate the possibility to reach such goal using a reliable validation metric.

Intyearpolator is, as its name suggests, a spatial interpolator designed for predicting years of a random field. 

## Research Project Links

The following links give access to the codes related to the project :

* [Primary pipeline - Construction dates extraction using maps](https://github.com/swiss-territorial-data-lab/regbl-poc)
* [Secondary pipeline - Construction dates extraction without maps (This repository)](https://github.com/swiss-territorial-data-lab/regbl-poc-intyearpolator)
* [Results and analysis tools for the primary pipeline](https://github.com/swiss-territorial-data-lab/regbl-poc-analysis)

The following links give access to official documentations on the considered data :

* [RegBL : Swiss federal register of buildings and dwellings](https://www.bfs.admin.ch/bfs/en/home/registers/federal-register-buildings-dwellings.html)
* [Maps : Swiss national maps 1:25'000](https://shop.swisstopo.admin.ch/en/products/maps/national/lk25)


# intYEARpolator
This predictor was especially designed for infering the year of construction of buildings in Switzerland. Although this is supplementary to an object detection based algorithm working on the presence(or absence) of houses in swisstopo maps in comparison to RegBL (GWR) database. Therefere, this model is only extrapolating regbl-poc (https://github.com/swiss-territorial-data-lab/regbl-poc) predictions to years beyond the oldest map available in some area. This is an add-on and should not replace the results from the main detector.

IntYEARpolator is a simple statistical model which main characteristic is to query for neighbour's data based on different searching radius. First of all, a general searching radius is defines as half of the largest distance (between random variables). For every prediction location, the variance between all data in the 'prior' searching radius will be used to create a 'posterior' searching radius. This way, the higher the variance, the smaller the searching radius, as we tend to trust data less. The exception to this rule is for variances that are higher than 2 x the mean distance between points. In this case, the searching radius increases again in order to avoid clusters of very old houses that during tests caused understimation. The figure below demonstrates the logic behing the creation of buffers.

![](doc/image/f1.png)

being *d* the distance between points, \mi the mean and s² the sample variance. 

It is important to state that the model will not predict values higher that the oldest map, this way letting the major responsability to the regbl-poc detector. 

The most recent update on the model includes the clusterin of the results from the porsterior mean and points coordinates with gaussian mixed models and the usage of the mean of every cluster, what could bring more detail to predictions.

# Usage

There are two input files called in the prompt, being one a table of the outputs of regbl-poc detector (deduce_location folder), once **compacted**, and the other the entire regbl database. The output for is a table adding **predicted year** and **prediction variance**. The second one was majorly thought for evaluation of the searching radiuses creation. An optional input refers to creating a plo tof the clusters performed or not with the **plot** argument, that can be assigned as 1 (True), or 0(False, default).

In your terminal, try:

```
$python3 intyearpolator.py -i /path/to/deduce_compacted_table -r /path/to/RegBL/GWR_database - /path/to/output/predictions
```

## Copyright and License

**regbl-poc-intyearpolator** - Huriel Reichel, Nils Hamel <br >
Copyright (c) 2020-2021 Republic and Canton of Geneva

This program is licensed under the terms of the GNU GPLv3. Documentation and illustrations are licensed under the terms of the CC BY-NC-SA.

## Dependencies

* Python 3.6 or superior

Packages can be installed either by pip or conda:

* pandas 1.2.0

* numpy 1.18.4

* scipy 1.5.4

* scikit-learn 0.24.0

