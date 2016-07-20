# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         util
# Purpose:
#
# Authors:      Maik Heistermann
#
# Created:      2015-11-6
# Copyright:    (c) Maik Heistermann
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python


import numpy as np
from scipy.spatial import cKDTree


def mask_from_bbox(x, y, bbox):
    """Return index array based on spatial selection from a bounding box.
    """
    ny, nx = x.shape
    
    ix = np.arange(x.size).reshape(x.shape)

    # Find bbox corners
    #    Plant a tree
    tree = cKDTree(np.vstack((x.ravel(),y.ravel())).transpose())
    # find lower left corner index
    dists, ixll = tree.query([bbox["left"], bbox["bottom"]], k=1)
    ill, jll = np.array(np.where(ix==ixll))[:,0]
    ill = (ixll / nx)#-1
    jll = (ixll % nx)#-1
    # find lower left corner index
    dists, ixur = tree.query([bbox["right"],bbox["top"]], k=1)
    iur, jur = np.array(np.where(ix==ixur))[:,0]
    iur = (ixur / nx)#+1
    jur = (ixur % nx)#+1
    
    mask = np.repeat(False, ix.size).reshape(ix.shape)
    if iur>ill:
        iur += 1
        jur += 1
        mask[ill:iur,jll:jur] = True
        shape = (iur-ill, jur-jll)
    else:
        ill += 1
        jur += 1
        mask[iur:ill,jll:jur] = True
        shape = (ill-iur, jur-jll)
    
    return mask, shape
        
#    return ix[ill:iur,jll:jur].ravel() 
    

def reduce_multipolygons(verts):
    """
    """
    for i, vert in enumerate(verts):
        if vert.ndim==1:
            # Multi-Polygons - keep only the largest polygon 
            verts[i] = vert[np.argmax([len(subpoly) for subpoly in vert])]
    return verts
    
    
def make_ids_unique(verts, ids):
    """Selects the longest polygon in case of duplicate IDs.
    """
    ids = np.array(ids)
    mask = np.repeat(False, len(ids))
    for id in np.unique(ids):
        ix = np.where( ids==id)[0]
        if len(ix) > 1:
            # More than one key matching? Find largest matching polygon
            mask[ ix[np.argmax([len(verts[i]) for i in ix])] ] = True
        else:
            mask[ix[0]] = True
    return verts[mask], ids[mask]

    



if __name__ == '__main__':
    
    pass