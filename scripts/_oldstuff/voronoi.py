# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 11:40:03 2016

@author: heistermann
"""

from scipy.spatial import Voronoi

from scipy.spatial import voronoi_plot_2d

import pylab as plt
import numpy as np

points = np.array([[0, 0.1], [0, 1.05], [0, 2.1], [1, 0], [1, 1], [1, 2],
                   [2, 0.1], [2.01, 1], [2.2, 2]])

vor = Voronoi(points)

vor.vertices
vor.regions
vor.ridge_vertices

plt.plot(points[:,0], points[:,1], 'o')
plt.plot(vor.vertices[:,0], vor.vertices[:,1], '*')
plt.xlim(-1, 3); plt.ylim(-1, 3)

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k')

#for i,simplex in enumerate(vor.ridge_vertices):
#    simplex = np.asarray(simplex)
#    if np.all(simplex >= 0):
#        plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], color=colors[i])

voronoi_plot_2d(vor)

#center = points.mean(axis=0)
#for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
#    simplex = np.asarray(simplex)
#    if np.any(simplex < 0):
#        i = simplex[simplex >= 0][0] # finite end Voronoi vertex
#        t = points[pointidx[1]] - points[pointidx[0]] # tangent
#        t /= np.linalg.norm(t)
#        n = np.array([-t[1], t[0]]) # normal
#        midpoint = points[pointidx].mean(axis=0)
#        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
#        plt.plot([vor.vertices[i,0], far_point[0]],
#                 [vor.vertices[i,1], far_point[1]], 'k--')

