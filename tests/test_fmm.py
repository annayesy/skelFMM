import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import util 
from src import skel_fmm

import numpy as np
from simpletree import BalancedTree, get_point_dist,morton

from time import time
import argparse
import pickle

import matplotlib.patches as patches

import matplotlib.pyplot as plt

def add_patches(ax,tree,box_list,keys=False,\
                edgecolor='black',facecolor='white',alpha=0.7,linewidth=2.0,text_label=False,fontsize=20):
    for box in box_list:
        if (keys):
            c,L = morton.get_key_params(box,tree.c0,tree.L0)
        else:
            c = tree.get_box_center(box)
            L = tree.get_box_length(box)
        rect = patches.Rectangle((c[0]-L/2, c[1]-L/2), L, L, \
                             linewidth=linewidth, edgecolor=edgecolor, \
                                 facecolor=facecolor,alpha=alpha)
        if (not keys and text_label):
            ax.text(c[0],c[1],"${%d}$" % (box),horizontalalignment='center',fontsize=fontsize)
        ax.add_patch(rect)


def get_problem_setup(N,npoints_max,tol=1e-5):

  ############# build the problem #############
  np.random.seed(0)
  XX1 = get_point_dist(N,'square')
  XX1 = XX1 * 0.49

  XX2  = get_point_dist(80,'square')
  XX2  = XX2 * 0.49
  XX2 += 0.51

  XX = np.vstack((XX1, XX2))

  kernel_func = util.laplace_2d

  tree = BalancedTree(XX,leaf_size=100)
  fmm = skel_fmm.SkelFMM(tree,kernel_func,kappa=0,tol=tol)

  fmm.skel_tree(npoints_max=npoints_max,verbose=True)
  fmm.setup_lists()

  ############# plot the test case #############
  fig,ax = plt.subplots()
  balanced_keys   = fmm.tree.leaf_keys
  ax.scatter(XX[:,0],XX[:,1],s=4,c='tab:blue')
  ax.axis('off')

  add_patches(ax,tree,tree.get_leaves(),keys=False,\
            edgecolor='tab:gray',facecolor='none',text_label=True,linewidth=2.0,alpha=0.5)

  plt.gca().set_aspect('equal', adjustable='box')
  plt.savefig("figures/test_case.png",bbox_inches='tight')
  return fmm

def get_problem_setup_annulus(N,npoints_max,tol=1e-5):

  ############# build the problem #############
  np.random.seed(0)
  XX = get_point_dist(N,'curvy_annulus')

  kernel_func = util.laplace_2d

  tree = BalancedTree(XX,leaf_size=100)
  fmm = skel_fmm.SkelFMM(tree,kernel_func,kappa=0,tol=tol)

  fmm.skel_tree(npoints_max=npoints_max,verbose=True)
  fmm.setup_lists()

  return fmm

############################################################################################

def err_eval(J,fmm):

  x = np.random.rand(fmm.N)
  resultvec         = fmm.matvec(x.copy())

  truevec   = fmm.kernel_func(fmm.tree.XX[J],fmm.tree.XX,fmm.kappa) @ x
  errvec    = truevec - resultvec[J]

  err    = np.linalg.norm(errvec)
  relerr_2    = np.linalg.norm(errvec) / np.linalg.norm(truevec)
  relerr_inf  = np.max(np.abs(errvec)) / np.max(np.abs(truevec))

  assert relerr_inf < fmm.tol


def skel_setup(npoints_max):
  N = int(400); root_level = 2

  fmm = get_problem_setup(N,npoints_max)
  assert fmm.root_level == root_level

  nleaves_above = 0
  for b in fmm.root_boxes:
    if (fmm.tree.get_box_level(b) < root_level):
      nleaves_above += 1
  assert nleaves_above == 1

  skel_boxes = np.setdiff1d(np.arange(fmm.nboxes), fmm.root_boxes)
  
  for b in fmm.root_boxes:
    assert fmm.rank_list[b] == 0
    assert np.linalg.norm(fmm.proj_list[b]) == 0

  for b in skel_boxes:
    if (fmm.tree.get_box_level(b) <= root_level):
      continue
    assert fmm.rank_list[b] > 0

  for l,t in fmm.u_from_qskel_list:
    assert fmm.tree.get_box_level(l) >= root_level
    assert fmm.tree.get_box_level(t) > root_level

  for t,l in fmm.uskel_from_q_list:
    assert fmm.tree.get_box_level(l) >= root_level
    assert fmm.tree.get_box_level(t) > root_level

  for t1,t2 in fmm.uskel_from_qskel_list:
    assert fmm.tree.get_box_level(t1) > root_level
    assert fmm.tree.get_box_level(t2) > root_level
  return fmm

def test_leafabove():
  fmm = skel_setup(400)

  ### test whether the matvec is correct on the indices of the leaf above
  leaf_above = fmm.root_boxes[0]
  J = fmm.tree.get_box_inds(leaf_above)
  err_eval(J,fmm)

def test_rand():
  fmm = skel_setup(400)

  J = np.random.choice(fmm.N,10,replace=False)
  err_eval(J,fmm)

def test_annulus():

  N = int(3000);

  fmm = get_problem_setup_annulus(N,100)

  J = np.random.choice(N,10,replace=False)
  err_eval(J,fmm)