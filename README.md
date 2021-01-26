# intYEARpolator
Intyearpolator is, as its name suggests, a spatial interpolator designed for predicting years of a random field. 

# Overview
This predictor was especially designed for infering the year of construction of buildings in Switzerland. Although this is supplementary to an object detection based algorithm working on the presence(or absence) of houses in swisstopo maps in comparison to RegBL (GWR) database. Therefere, this model is only extrapolating regbl-poc (https://github.com/swiss-territorial-data-lab/regbl-poc) predictions to years beyond the oldest map available in some area. This is an add-on and should not replace the results from the main detector.

IntYEARpolator is a simple statistical model which main characteristic is to query for neighbour's data based on different searching radius. First of all, a general searching radius is defines as half of the largest distance (between random variables). For every prediction location, the variance between all data in the 'prior' searching radius will be used to create a 'posterior' searching radius. This way, the higher the variance, the smaller the searching radius, as we tend to trust data less. The exception to this rule is for variances that are higher than 2 x the mean distance between points. In this case, the searching radius increases again in order to avoid clusters of very old houses that during tests caused understimation. The figure below demonstrates the logic behing the creation of buffers.

![](f1.png)

being *d* the distance between points, \mi the mean and s² the sample variance. 

It is important to state that the model will not predict values higher that the oldest map, this way letting the major responsability to the regbl-poc detector. 

# Usage

There are two input files called in the prompt, being one a table of the outputs of regbl-poc detector (deduce_location folder), once **compacted**, and the other the entire regbl database. The output for is a table adding **predicted year** and **prediction varaince**. The second one was majorly thought for evaluation of the searching radiuses creation. 

In your terminal, try:

```
$python3 intyearpolator.py -i /path/to/deduce_compacted_table -r /path/to/RegBL/GWR_database - /path/to/output/predictions
```

## Copyright and License

**intYEARpolator** - Huriel Reichel, Nils Hamel <br >
Copyright (c) 2020 Republic and Canton of Geneva

This program is licensed under the terms of the GNU GPLv3. Documentation and illustrations are licensed under the terms of the CC BY-NC-SA.

## Dependencies

List of Python packages (can be installed either by pip or conda) used:

* pandas 1.2.0

* numpy 1.18.4

* rpy2 (for now) 3.4.2

One may expect that older and newer versions can work as well.

R must be installed in your machine for now. 

R Version: 4.0.2 (or superior)

