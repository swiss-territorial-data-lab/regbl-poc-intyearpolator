#!/usr/bin/env python3
#  mean-year
#
#     Huriel Reichel - huriel.ruan@gmail.com
#     Nils Hamel - nils.hamel@bluewin.ch
#     Copyright (c) 2020 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import argparse
import math
from scipy.spatial.distance import cdist
from sklearn import mixture

# Avoid unnecessary warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# create argument parser #
pm_argparse = argparse.ArgumentParser()

# argument and parameter directive #
pm_argparse.add_argument( '-i', '--input', type=str, help='input table path' )
pm_argparse.add_argument( '-r', '--regbl', type=str, help='input regbl path' )
pm_argparse.add_argument( '-o', '--output', type=str, help='output table path' )
pm_argparse.add_argument( '-p', '--plot', type=int, default=0, help='whether cluster should be plotted (1) or not (0). Default to False (0)')

# read argument and parameters #
pm_args = pm_argparse.parse_args()

# function to measure distance
def dist(x1, y1, x2, y2):
    
    if x1 == x2:
    
        return abs(y2 - y1)
    
    elif y1 == y2:
    
        return abs(x2 - x1)
    
    else:
    
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# import datasets
pts = pd.read_table(pm_args.input, sep = ' ')
regbl = pd.read_table(pm_args.regbl, low_memory=False)

# detect oldest map
last_map = pts.iloc[:,1].min()

# query for input values below oldest map
ran_var = pts[pts.iloc[:,1] == last_map]
ran_var.columns = ['EGID','year1','year2']

# merge data frames based on EGID
merged = pd.merge(left=ran_var, right=regbl, left_on='EGID', right_on='EGID')
merged['year'] = merged['GBAUJ'].astype(float)

# select NaN year values among the original years (GBAUJ) given by RegBL 
pred_pts = merged[merged['year'].isnull()]

regbl_coords = merged[["GKODE", "GKODN"]]
regbl_coords = regbl_coords.dropna()
pred_coords = pred_pts[["GKODE", "GKODN"]]

# matrix of distances
m = cdist(pred_coords, regbl_coords, 'euclidean')
t = np.matrix.transpose(m)

# define prior searching radius
prior_sr = (dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5
print("prior searching radius is ", prior_sr, "\n")

# query in the transposed matrix by prior searching radius
which = []

for i in range(len(pred_pts)):
    
    w = np.where(t[i] <= prior_sr)
    which.append(w)
    
which = list(which)

# Calculate prior mean and variance for posterior searching radius definition
prior_mean = []
prior_var = []
for i in range(len(pred_pts)): 
    years = merged.iloc[which[i][0], -1] 
    mean = years.mean() 
    var = years.var()
    prior_mean.append(mean)
    prior_var.append(var)

# Create searching radius based on variance
print("Creating posterior searching radius based on variance \n" )
def searching_radius(var, mean):
    
    if var <= 0.5 * mean:
        
        return (dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.75
    
    elif var > 0.5 * mean and var <= 0.75 * mean:
        
        return ((dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5) * 0.65
    
    elif var > 0.75 * mean and var <= 1 * mean:
        
        return ((dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5) * 0.55
    
    elif var > 1 * mean and var <= 1.5 * mean:
        
        return ((dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5) * 0.25
    
    elif var > 1.5 * mean and var <= 2.0 * mean:
        
        return ((dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5) * 0.125
    
    else :
    
        return ((dist(merged['GKODE'].min(), merged['GKODN'].min(), merged['GKODE'].max(), merged['GKODN'].max())) * 0.5) * 0.4

sr_posterior = []
for i in range(len(pred_pts)):
    sr = searching_radius(prior_var[i], prior_mean[i])
    sr_posterior.append(sr)
    
sr = np.array(sr_posterior)

# query in the transposed matrix by posterior searching radius
inrange = []

for i in range(len(pred_pts)):
    
    w = np.where(t[i] <= sr[i])
    inrange.append(w)
    
inrange = list(inrange)
    
# calculate posterior mean and variance
print('Calculating posterior mean \n')
mean_posterior = []
var_posterior = []
for i in range(len(pred_pts)): 
    years = merged.iloc[inrange[i][0], -1]
    mean = years.mean() 
    
   # if mean > last_map:
    #    mean = last_map
    
    var = years.var()
    mean_posterior.append(mean)
    var_posterior.append(var)

# join predictions
predicted_year = list(mean_posterior)
prediction_variance = list(var_posterior)

pred_pts = pd.DataFrame(pred_pts)
pred_pts['pred_year'] = predicted_year
pred_pts['pred_var'] = prediction_variance
pred_pts['pred_year'] = pred_pts['pred_year'].round()
pred_pts = pred_pts[['EGID', 'pred_year', 'pred_var', 'year']]

pts.columns = ['EGID', 'year1', 'year2']
out = pd.merge(left=pts, right=pred_pts, left_on='EGID', right_on='EGID', how = 'left')

mergo = pd.merge(left=regbl, right=out, left_on='EGID', right_on='EGID', how = 'right')

# data wrangling for unsupervised learning
summary_year = []
for j in range(len(mergo)):
    if math.isnan(mergo['GBAUJ'].iloc[j]) == 0:
        year = mergo['GBAUJ'].iloc[j].round()
    
    elif math.isnan(mergo['GBAUJ'].iloc[j]) == 1 and math.isnan(mergo['year1'].iloc[j]) == 0 and math.isnan(mergo['pred_year'].iloc[j]) == 1:
        year = (mergo['year1'].iloc[j] +  mergo['year2'].iloc[j]) / 2
    
    elif math.isnan(mergo['pred_year'].iloc[j]) == 0:
        year = mergo['pred_year'].iloc[j].round()
    
    else: 
        year = 0

    summary_year.append(year)
     
mergo['filled'] = summary_year
mergo = mergo[['EGID', 'GKODE', 'GKODN', 'filled']]

X = mergo.iloc[:,1:4].to_numpy()

# Gaussian Mixture Model for clustering
n_comp = int((len(mergo)) * 0.02 - 56 )
print("running gaussian mixture model clustering with ", n_comp, " components \n" )
gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type = 'full').fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

if pm_args.plot != 0:
    import matplotlib
    import matplotlib.pyplot as plt
    size = 50 * probs.max(1) ** 2
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size);
    matplotlib.pyplot.show()

mergo['cluster'] = labels

# calculate mean of values in each cluster
def createList(n_comp): 
    return list(range(0, n_comp)) 
seq_clusters = createList(n_comp) 

m_c = {}
for i in seq_clusters:
    m_c['c%s' % i] = [np.array(mergo[mergo['cluster'] == i].iloc[:,-2]).mean()]
m_c = np.array(pd.DataFrame(m_c, index = [0]))
m_c = m_c.astype(int)

# lambda of weights (probabilities of each point being in a certain cluster) * mean of values inside cluster
a = probs * m_c

# making final predictions
Z = []
for i in range(len(a)):
    z = a[i].sum().astype(int)
    Z.append(z)
mergo['Z'] = Z 
    
# export results
pd.DataFrame.to_csv(mergo, pm_args.output)
