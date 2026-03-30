import os
import multiprocessing as mp

import psutil
from . import operators, util
import numpy as np
from   simpletree.morton_tree import findin_sorted
from   time import time

############################################ SKEL FMM CLASS ############################################

class RecursiveSkeletonization:
    """
    A class implementing Skeletonized Fast Multipole Method (SkelFMM).
    This method relies on hierarchical tree structures to accelerate computations involving
    kernel functions, making it suitable for applications like potential theory and N-body problems.
    """

    def __init__(self,tree,operator,kappa,tol,\
        max_bs = -1, max_rank = -1, geometry=None):
        """
        Initialize the SkelFMM object.

        Parameters:
        - tree: A hierarchical tree structure (e.g., quadtree or octree) used for spatial decomposition.
        - operator: Kernel operator used for interactions.
        - kappa: Parameter for the kernel function (e.g., wave number).
        - tol: Tolerance for numerical approximations.
        - max_bs: Maximum block size for skeletonization (default: -1, automatically computed).
        - max_rank: Maximum rank for low-rank approximations (default: -1, automatically computed).
        - geometry: Optional general geometry object aligned with tree.XX.
          The operator interprets this object when points act as sources and/or targets.
        """
        self.N = tree.N  # Number of points in the tree
        self.tree = tree  # Tree structure for spatial decomposition
        self.nlevels = tree.nlevels  # Number of levels in the tree
        self.nboxes = tree.nboxes  # Number of boxes in the tree

        # Kernel-related parameters
        self.kappa = kappa
        self.operator = operators._require_operator(operator)
        self.ndim = tree.ndim  # Dimensionality of the space (2D, 3D, etc.)
        self.tol = tol  # Tolerance for numerical approximation
        self.geometry = geometry

        if self.operator.ndim > 0 and self.operator.ndim != self.ndim:
            raise ValueError("Operator dimensionality does not match the tree dimensionality")
        if self.operator.requires_geometry and self.geometry is None:
            raise ValueError("Geometry is required for this operator")
        if self.geometry is not None and self.geometry.shape[0] != self.N:
            raise ValueError("Geometry must be aligned with tree.XX")

        # Extract tree information
        leaves = self.tree.get_leaves()  # Retrieve leaf nodes
        self.tree_lev_sep = np.zeros(self.nlevels + 1, dtype=int)  # Level separators
        self.tree_boxes = np.zeros(self.nboxes, dtype=int)  # Array for storing box indices
        self.ntree_boxes = 0  # Counter for non-leaf tree boxes

        # Populate tree_boxes with non-leaf box indices and calculate level separators
        for lev in range(self.nlevels):
            boxes_lev = self.tree.get_boxes_level(lev)  # Get all boxes at this level
            treeboxes_lev = np.setdiff1d(boxes_lev, leaves)  # Exclude leaf boxes
            ntreeboxes_lev = treeboxes_lev.shape[0]  # Number of non-leaf boxes

            self.tree_boxes[self.ntree_boxes:self.ntree_boxes + ntreeboxes_lev] = treeboxes_lev
            self.ntree_boxes += ntreeboxes_lev
            self.tree_lev_sep[lev + 1] = self.ntree_boxes
        self.tree_boxes = self.tree_boxes[:self.ntree_boxes]

        self.max_bs_leaf = tree.leaf_size  # Maximum block size for leaf nodes

        # Set proxy points depending on the level
        ndigits = int(-np.log10(self.tol/100))
        self.nproxy_level  = np.zeros(self.nlevels,dtype=int)
        for lev in range(self.nlevels):
            box_length = self.tree.L0 * 2**(-lev)

            if (self.ndim == 2):
                nproxy  = int( ndigits * 10)
                nproxy += int( 4*self.kappa*box_length)
            else:
                nproxy  = np.max(np.array([ int( ndigits**2 * 8 ),400]))
                nproxy += 2 * self.kappa * box_length * np.max( np.array([5,ndigits-1]) )\
                + (10 if self.kappa > 0 else 0)
            self.nproxy_level[lev] = nproxy

        if ((max_bs > 0) and (max_rank > 0)):
            self.max_rank = max_rank
            self.max_bs   = np.max( np.array([max_bs,self.max_bs_leaf]) )

        else:
            self.max_rank   = int(self.max_bs_leaf*0.75)
            self.max_bs     = int((2**self.ndim) *self.max_bs_leaf)
        self.max_rank_leaf  = self.max_rank

        # Initialize data structures for skeletonization
        K_tmp = self.operator.numpy_matrix(
            np.zeros((1, self.ndim)),
            np.zeros((1, self.ndim)),
            geom_src=None if self.geometry is None else np.zeros((1,) + self.geometry.shape[1:], dtype=self.geometry.dtype),
            geom_trg=None if self.geometry is None else np.zeros((1,) + self.geometry.shape[1:], dtype=self.geometry.dtype),
            param=kappa,
        )
        kernel_dtype = K_tmp.dtype

        self.XX_list   = np.zeros((self.nboxes,self.max_bs,self.ndim))
        self.geom_list = None
        if self.geometry is not None:
            self.geom_list = np.zeros((self.nboxes,self.max_bs) + self.geometry.shape[1:], dtype=self.geometry.dtype)
        self.idx_list  = np.zeros((self.nboxes,self.max_bs),dtype=int)

        self.proj_list_tree = np.zeros((self.ntree_boxes,self.max_rank,\
            self.max_bs),dtype=kernel_dtype)

        self.proj_list   = np.zeros((self.nboxes,self.max_rank_leaf,\
            self.max_bs_leaf),dtype=kernel_dtype)

        self.bs_list   = np.zeros(self.nboxes,dtype=int)
        self.rank_list = np.zeros(self.nboxes,dtype=int)
        self.rank_max = 0

        self.assign_leaves()

    @property
    def nbytes_proj(self):
        """
        Calculate the total memory usage of projection matrices in bytes.

        Returns:
        - Memory usage in bytes.
        """
        const_mult = 8 if (self.kappa == 0) else 16 # Float (8 bytes) or complex (16 bytes)
        nitems = self.proj_list.size
        leaves = self.tree.get_leaves()
        print("\t shape proj",self.proj_list.shape,"shape proj_tree",self.proj_list_tree.shape)
        nitems += self.proj_list_tree.size
        return nitems * const_mult

    ################################# BUILD BY LEVEL UTILITY METHODS ########################


    def assign_leaves(self):
        """
        Assign proxy points and skeleton indices for leaf nodes.
        """

        self.leaf_size = 0

        for leaf in self.tree.get_leaves():
            I_Borig = self.tree.get_box_inds(leaf).copy()
            assert I_Borig.shape[0] > 0, "Leaf box must contain points"
            if (I_Borig.shape[0] > self.leaf_size):
                self.leaf_size = I_Borig.shape[0]

            XX_B    = self.tree.XX[I_Borig]
            bs_leaf = XX_B.shape[0]

            if bs_leaf > self.max_bs_leaf:
                raise ValueError("Leaf block size exceeds maximum allowed")

            # Assign data to leaf
            self.XX_list[leaf,:bs_leaf]  = XX_B.copy()
            if self.geom_list is not None:
                self.geom_list[leaf,:bs_leaf] = self.geometry[I_Borig].copy()
            self.bs_list[leaf] = bs_leaf
            self.idx_list[leaf,:bs_leaf] = np.arange(bs_leaf)

    def relerr_fmm_apply_check(self, x, *, ncheck=64, seed=0, indices=None):
        """
        Return a relative ``ell_infinity`` subset apply error for this FMM.

        This compares the fast matvec against a dense operator evaluation on a
        subset of targets:

            ||y_dense - y_fmm||_inf / ||y_dense||_inf
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("relerr_fmm_apply_check expects a 1D source vector")
        if x.shape[0] != self.N:
            raise ValueError("Source vector length must match the number of tree points")

        if indices is None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(self.N, size=min(int(ncheck), self.N), replace=False)
        else:
            indices = np.asarray(indices, dtype=int)

        geom_src = None
        geom_trg = None
        if self.operator.geometry_side == "source":
            geom_src = self.geometry
        elif self.operator.geometry_side == "target":
            geom_trg = self.geometry[indices]

        y_fast = self.matvec(x.copy())
        y_true = self.operator.numpy_matrix(
            self.tree.XX[indices],
            self.tree.XX,
            geom_trg=geom_trg,
            geom_src=geom_src,
            param=self.kappa,
        ) @ x

        denom = max(np.linalg.norm(y_true, ord=np.inf), 1e-30)
        return np.linalg.norm(y_true - y_fast[indices], ord=np.inf) / denom


    def update_lev_skel(self,lev):
        """
        Update skeletons for all boxes at a specific level.
        
        Parameters:
        - lev: Level in the tree to update.
        """

        for box in self.tree.get_boxes_level(lev):
            if (self.tree.is_leaf(box)):
                continue
            for c in self.tree.get_box_children(box):
                # Update skeletons from child boxes

                XX_C   = self.XX_list[c,:self.bs_list[c]]
                geom_C = None if self.geom_list is None else self.geom_list[c,:self.bs_list[c]]
                rank_c = self.rank_list[c]

                XX_C_skel = XX_C.copy()[:rank_c]
                geom_C_skel = None if geom_C is None else geom_C.copy()[:rank_c]

                # Check for memory reallocation
                if (self.bs_list[box] + rank_c > self.max_bs):

                    # Dynamically increase block size and update data structures

                    updated_max_bs = int(self.bs_list[box] + 1.2 * rank_c)
                    print("\t Max bs increased to %d" % updated_max_bs)
                    XX_tmp = self.XX_list; idx_tmp = self.idx_list; proj_tmp = self.proj_list_tree
                    geom_tmp = self.geom_list

                    self.XX_list   = np.zeros((self.nboxes,updated_max_bs,self.ndim),dtype=XX_tmp.dtype)
                    self.XX_list[:,:self.max_bs] = XX_tmp

                    if geom_tmp is not None:
                        self.geom_list = np.zeros((self.nboxes,updated_max_bs) + geom_tmp.shape[2:], dtype=geom_tmp.dtype)
                        self.geom_list[:,:self.max_bs] = geom_tmp

                    self.proj_list_tree = np.zeros((self.ntree_boxes,self.max_rank,updated_max_bs),\
                        dtype=proj_tmp.dtype)
                    self.proj_list_tree[:,:,:self.max_bs] = proj_tmp

                    self.idx_list  = np.zeros((self.nboxes,updated_max_bs),dtype=idx_tmp.dtype)
                    self.idx_list[:,:self.max_bs]    = idx_tmp
                    self.max_bs = updated_max_bs

                # Append skeleton points to parent box
                self.XX_list[box, self.bs_list[box] : self.bs_list[box] + XX_C_skel.shape[0]] = XX_C_skel
                if geom_C_skel is not None:
                    self.geom_list[box, self.bs_list[box] : self.bs_list[box] + XX_C_skel.shape[0]] = geom_C_skel
                self.bs_list[box] += XX_C_skel.shape[0]

            # Update indices for parent box
            self.idx_list[box,:self.bs_list[box]] = np.arange(self.bs_list[box])

    def set_skel_box_info(self,box,XX_B,*args):
        """
        Updates the skeleton information for a specific box in the tree.

        Parameters:
        - box: Index of the box to update.
        - XX_B: Array of points (proxy or skeleton points) in the box.
        - args: Either `(rank_box, idx_box, proj_box)` or
          `(geom_B, rank_box, idx_box, proj_box)`.
        """
        if len(args) == 3:
            geom_B = None
            rank_box, idx_box, proj_box = args
        elif len(args) == 4:
            geom_B, rank_box, idx_box, proj_box = args
        else:
            raise ValueError("Unexpected set_skel_box_info argument format")

        bs_box = XX_B.shape[0]
        self.rank_list[box] = rank_box
        self.bs_list[box]   = bs_box

        if ( self.tree.is_leaf(box) and (rank_box > self.max_rank_leaf)):

            # Dynamically adjust storage for leaf nodes if the rank exceeds current max rank

            updated_max_rank = int(1.2 * rank_box)
            print("\t Max rank leaf increased to %d" % (updated_max_rank))
            proj_tmp = self.proj_list
            self.proj_list = np.zeros((self.nboxes,updated_max_rank,self.max_bs_leaf),\
                dtype=proj_tmp.dtype)
            self.proj_list[:,:self.max_rank_leaf,:self.max_bs_leaf] = proj_tmp
            self.max_rank_leaf = updated_max_rank

        elif ( (not self.tree.is_leaf(box)) and (rank_box > self.max_rank) ):

             # Dynamically adjust storage for tree nodes if the rank exceeds current max rank
           
            updated_max_rank = int(1.2 * rank_box)
            print("\t Max rank tree increased to %d" % (updated_max_rank))
            proj_tmp = self.proj_list_tree
            self.proj_list_tree = np.zeros((self.ntree_boxes,updated_max_rank,self.max_bs),\
                dtype=proj_tmp.dtype)
            self.proj_list_tree[:,:self.max_rank,:self.max_bs] = proj_tmp
            self.max_rank = updated_max_rank

        # Assign interpolation matrix for leaf or non-leaf boxes
        if (self.tree.is_leaf(box)):
            self.proj_list[box,:rank_box,rank_box:bs_box] = proj_box
        else:

            ind = findin_sorted(self.tree_boxes,box)
            assert ind > 0, "Box not found in tree boxes."
            self.proj_list_tree[ind,:rank_box,rank_box:bs_box] = proj_box
        
        # Store the skeleton point indices and data

        self.idx_list[box,:bs_box] = idx_box
        self.XX_list[box,:bs_box]  = XX_B[idx_box].copy()
        if self.geom_list is not None and geom_B is not None:
            self.geom_list[box,:bs_box] = geom_B[idx_box].copy()

    def skel_tree(self,npoints_max,p=None,verbose=False,min_parallel_items=64):

        """
        Computes a skeleton indices and corresponding interpolation matrix
        for each box in the tree. The interpolation matrix translates outgoing representations for children
        to the parent, and likewise, translates incoming representations for parent boxes to the children.

        Parameters:
        - npoints_max: Maximum number of points allowed at root level.
        - p: Parallel processing pool for distributed computation. Pass ``"auto"`` to
          lazily create and manage a local process pool only when a level is large
          enough to benefit; pass ``None`` to force serial skeletonization.
        - verbose: If True, prints additional debug information.
        """

        def get_default_num_workers():
            physical = psutil.cpu_count(logical=False)
            logical = os.cpu_count() or 1
            return max(1, physical or logical)

        def get_skel_level_data(self,lev):
            """
            Prepare box ids and geometry for one level.

            Parameters:
            - lev: Level of the tree.

            Returns:
            - boxes_lev: Array of boxes at the level.
            - centers_l: Array of box centers.
            - len_l: Box length at the level.
            """
            boxes_lev = self.tree.get_boxes_level(lev)
            len_l = self.tree.get_box_length(boxes_lev[0])
            centers_l = self.tree.get_boxes_centers(boxes_lev)
            return boxes_lev, centers_l, len_l

        owned_pool = None
        owned_pool_workers = 0
        # Small trees can finish without ever hitting the early-exit branch below.
        # Default to the coarsest non-leaf level so later setup code always has a root level.
        self.root_level = max(0, self.tree.nlevels - 1)
        try:
            # Iterate from the highest level to the root level
            for lev in range(self.tree.nlevels-1,0,-1):

                tic = time()
                self.update_lev_skel(lev)
                boxes_lev, centers_l, len_l = get_skel_level_data(self,lev)

                # Identify active boxes
                active_boxes = np.hstack((self.tree.get_leaves_above(lev),\
                    boxes_lev))

                # Determine if we should stop at this level
                npoints_lev = np.sum(self.bs_list[active_boxes])
                print ('\t Level %2.0f has %6.0f npoints'%(lev,npoints_lev))
                if ( (lev == 2) or (npoints_lev <= npoints_max)):
                    self.root_level = lev
                    break

                toc_serial   = time() - tic
                toc_parallel = 0

                num_items = boxes_lev.shape[0]
                pool = p
                num_workers = 1

                if p == "auto":
                    num_workers = max(1, min(get_default_num_workers(), num_items))
                    should_parallel = num_items >= max(min_parallel_items, 4 * num_workers)
                    if should_parallel and owned_pool is None:
                        start_method = "spawn"
                        ctx = mp.get_context(start_method)
                        owned_pool = ctx.Pool(processes=num_workers, maxtasksperchild=64)
                        owned_pool_workers = num_workers
                        if verbose:
                            print("\t Skeleton pool: %d workers via %s" % (num_workers, start_method))
                    pool = owned_pool
                    if owned_pool is not None:
                        num_workers = owned_pool_workers
                else:
                    if p is not None:
                        num_workers = max(1, min(getattr(p, "_processes", get_default_num_workers()), num_items))
                    should_parallel = (p is not None) and (num_items >= max(min_parallel_items, 4 * num_workers))

                if (not should_parallel):
                    # Serial processing
                    tic = time()
                    nproxy = int(self.nproxy_level[lev])
                    for box, box_center in zip(boxes_lev, centers_l):
                        bs = int(self.bs_list[box])
                        XX_B = self.XX_list[box, :bs]
                        geom_B = None if self.geom_list is None else self.geom_list[box, :bs]
                        rank_box, idx_box, proj_box = util.skel_box_helper(
                            self.operator,
                            self.kappa,
                            XX_B,
                            box_center,
                            len_l,
                            self.tol,
                            nproxy=nproxy,
                            geom=geom_B,
                        )
                        box_id = int(box)
                        if geom_B is None:
                            self.set_skel_box_info(box_id, XX_B, rank_box, idx_box, proj_box)
                        else:
                            self.set_skel_box_info(box_id, XX_B, geom_B, rank_box, idx_box, proj_box)
                    toc_serial += time() - tic
                    toc_parallel += 0
                else:
                    # Parallel processing
                    chunksize = max(1, num_items // (8 * num_workers))
                    tic = time()
                    bs_level = np.ascontiguousarray(self.bs_list[boxes_lev])
                    max_bs_level = int(np.max(bs_level))
                    XX_level = np.ascontiguousarray(self.XX_list[boxes_lev, :max_bs_level])
                    geom_level = None
                    if self.geom_list is not None:
                        geom_level = np.ascontiguousarray(self.geom_list[boxes_lev, :max_bs_level])

                    if geom_level is None:
                        task_items = (
                            (
                                item_idx,
                                int(boxes_lev[item_idx]),
                                self.operator,
                                self.kappa,
                                XX_level[item_idx, : int(bs_level[item_idx])],
                                centers_l[item_idx],
                                len_l,
                                int(self.nproxy_level[lev]),
                                self.tol,
                            )
                            for item_idx in range(num_items)
                        )
                    else:
                        task_items = (
                            (
                                item_idx,
                                int(boxes_lev[item_idx]),
                                self.operator,
                                self.kappa,
                                XX_level[item_idx, : int(bs_level[item_idx])],
                                geom_level[item_idx, : int(bs_level[item_idx])],
                                centers_l[item_idx],
                                len_l,
                                int(self.nproxy_level[lev]),
                                self.tol,
                            )
                            for item_idx in range(num_items)
                        )

                    for item_idx, box, rank_box, idx_box, proj_box in pool.imap_unordered(
                        util.skel_box_info,
                        task_items,
                        chunksize=chunksize,
                    ):
                        bs = int(bs_level[item_idx])
                        XX_B = XX_level[item_idx, :bs]
                        if geom_level is None:
                            self.set_skel_box_info(box, XX_B, rank_box, idx_box, proj_box)
                        else:
                            geom_B = geom_level[item_idx, :bs]
                            self.set_skel_box_info(box, XX_B, geom_B, rank_box, idx_box, proj_box)
                    toc_parallel = time() - tic

                # Print performance information

                if (verbose):
                    print("\t Level %2.0f ops: serial %5.2f, parallel %5.2f" \
                          % (lev,toc_serial,toc_parallel))
        finally:
            if owned_pool is not None:
                owned_pool.close()
                owned_pool.join()

    def setup_lists(self,verbose=False):
        """
        Finalizes and prepares the data structures for the skeletonized tree.
        
        Parameters:
        - verbose: If True, prints additional debug information during setup.
        """
        self.prune_skel()
        self.get_parallel_lists(verbose)
        self.precompute_inds(verbose=verbose)

    def prune_skel(self):
        """
        Optimizes the skeletonization data structures by pruning unused space
        and updating the bounds of ranks and block sizes.
        """

        self.max_bs   = np.max(self.bs_list)
        self.max_rank = np.max(self.rank_list)
        self.rank_max = int(self.max_rank)

        positive_ranks = self.rank_list[self.rank_list > 0]
        # Small problems can legitimately produce no compressed ranks.
        self.min_rank = np.min(positive_ranks) if positive_ranks.size > 0 else 0

        leaves              = self.tree.get_leaves()
        self.max_bs_leaf    = np.max(self.bs_list[leaves])
        self.max_rank_leaf  = np.max(self.rank_list[leaves])

        self.proj_list      = self.proj_list[:,:self.max_rank_leaf,self.min_rank:self.max_bs_leaf]
        self.proj_list_tree = self.proj_list_tree[:,:self.max_rank,self.min_rank:self.max_bs]

        self.XX_list       = self.XX_list[:,:self.max_bs]
        if self.geom_list is not None:
            self.geom_list = self.geom_list[:,:self.max_bs]
        self.idx_list      = self.idx_list[:,:self.max_bs]

    ##################################### BUILD UTILITY METHODS #####################################

    def get_skel_box_info(self,box):
        """
        Retrieves detailed skeleton information for a given box.

        Parameters:
        - box: Index of the box for which skeleton information is required.

        Returns:
        - XX_B: Array of points in the box.
        - bs_box: Block size (number of points) in the box.
        - rank_box: Rank of the skeleton for the box.
        - idx_box: Indices of the selected skeleton points.
        - proj_box: Projection matrix representing the skeletonization of the box.
        """
        assert box >= 0, "Box index must be non-negative."

        bs_box    = self.bs_list[box]
        XX_B      = self.XX_list[box,:bs_box]

        rank_box = self.rank_list[box]
        idx_box  = self.idx_list[box,:bs_box]

        if ( self.tree.is_leaf(box) ):
            proj_box = self.proj_list[box,:rank_box,(rank_box-self.min_rank):(bs_box-self.min_rank)]

        else:
            ind = findin_sorted(self.tree_boxes,box); assert ind > 0

            proj_box = self.proj_list_tree[ind,:rank_box,(rank_box-self.min_rank):(bs_box-self.min_rank)]
        
        # Consistency checks for the retrieved data
        try:
            assert rank_box > 0
            assert proj_box.shape[0] == rank_box
            assert proj_box.shape[1] == (bs_box - rank_box)
            assert np.linalg.norm(idx_box.astype(float)) > 0 if bs_box > 1 else True
        except:
            raise ValueError("Something weird with box %d, is leaf=%s on level %d" \
                %(box,self.tree.is_leaf(box), self.tree.get_box_level(box)))
        return XX_B,bs_box,rank_box,idx_box,proj_box

    def get_box_info(self,box):
        """
        Retrieves basic information for a given box.

        Parameters:
        - box: Index of the box for which information is required.

        Returns:
        - XX_B: Array of points in the box.
        - bs_box: Block size (number of points) in the box.
        - idx_box: Indices of the points in the box.
        """
        bs_box    = self.bs_list[box]
        XX_B      = self.XX_list[box,:bs_box]

        idx_box  = self.idx_list[box,:bs_box]

        # Consistency checks for retrieved data
        try:
            assert box >= 0
            assert bs_box > 0
            assert np.linalg.norm(idx_box.astype(float)) > 0 if bs_box > 1 else True
            assert np.linalg.norm(XX_B.astype(float)) > 0
        except:
            raise ValueError("Something weird with box %d is_leaf %s on level %d" \
                             %(box, self.tree.is_leaf(box), self.tree.get_box_level(box)))
        return XX_B,bs_box,idx_box

    def _get_geometry(self, box, rank_only=False):
        if self.geom_list is None:
            return None
        if rank_only:
            return self.geom_list[box, :self.rank_list[box]]
        return self.geom_list[box, :self.bs_list[box]]

    def _kernel_matrix(self, XX_U, XX_Q, geom_U=None, geom_Q=None):
        return self.operator.numpy_matrix(XX_U, XX_Q, geom_trg=geom_U, geom_src=geom_Q, param=self.kappa)

    def get_parallel_lists(self,verbose):

        """
        Prepares data structures suitable for parallel computations.

        uskel_from_qskel_list is used to translate incoming from outgoing (ifo) expansions.
        u_from_qskel_list     is used to translate targets from outgoing (tfo) expansions.
        uskel_from_q_list     is used to translate incoming from source (ifs) expansions.
        u_from_q_list         is used to translate incoming from outgoing expansions at the root level.

        Parameters:
        - verbose: If True, prints additional debug information about the prepared lists.
        """

        nentries = 2*self.tree.nboxes*( 3**self.tree.ndim )
        uskel_from_q_list     = np.zeros((nentries,2),dtype=int)
        u_from_qskel_list     = np.zeros((nentries,2),dtype=int)

        leaf_uskel_from_qskel_list = np.zeros((nentries,2),dtype=int)
        tree_uskel_from_qskel_list = np.zeros((nentries,2),dtype=int)

        leaf_mask = self.tree.are_leaves(np.arange(self.tree.nboxes))

        acc_coarse = 0; acc_coll_leaf = 0; acc_coll_tree = 0
        for lev in range(self.root_level + 1, self.nlevels):
            boxes_lev = self.tree.get_boxes_level(lev)
            colleague_rows = self.tree.get_boxes_colleague_neigh_fullvec(boxes_lev)
            coarse_rows = self.tree.get_boxes_coarse_neigh_fullvec(boxes_lev)

            for box, colleagues, coarse_leaves in zip(boxes_lev, colleague_rows, coarse_rows):
                colleagues = colleagues[colleagues >= 0]
                if colleagues.size > 0:
                    if leaf_mask[box]:
                        colleague_leaf_mask = leaf_mask[colleagues]
                        leaf_colleagues = colleagues[colleague_leaf_mask]
                        tree_colleagues = colleagues[np.logical_not(colleague_leaf_mask)]

                        n_leaf_colleagues = leaf_colleagues.shape[0]
                        if n_leaf_colleagues > 0:
                            leaf_uskel_from_qskel_list[acc_coll_leaf:acc_coll_leaf + n_leaf_colleagues, 0] = box
                            leaf_uskel_from_qskel_list[acc_coll_leaf:acc_coll_leaf + n_leaf_colleagues, 1] = leaf_colleagues
                            acc_coll_leaf += n_leaf_colleagues

                        n_tree_colleagues = tree_colleagues.shape[0]
                        if n_tree_colleagues > 0:
                            tree_uskel_from_qskel_list[acc_coll_tree:acc_coll_tree + n_tree_colleagues, 0] = box
                            tree_uskel_from_qskel_list[acc_coll_tree:acc_coll_tree + n_tree_colleagues, 1] = tree_colleagues
                            acc_coll_tree += n_tree_colleagues
                    else:
                        n_colleagues = colleagues.shape[0]
                        tree_uskel_from_qskel_list[acc_coll_tree:acc_coll_tree + n_colleagues, 0] = box
                        tree_uskel_from_qskel_list[acc_coll_tree:acc_coll_tree + n_colleagues, 1] = colleagues
                        acc_coll_tree += n_colleagues

                coarse_leaves = coarse_leaves[coarse_leaves >= 0]
                n_coarse = coarse_leaves.shape[0]
                if n_coarse > 0:
                    uskel_from_q_list[acc_coarse:acc_coarse + n_coarse, 0] = box
                    uskel_from_q_list[acc_coarse:acc_coarse + n_coarse, 1] = coarse_leaves
                    u_from_qskel_list[acc_coarse:acc_coarse + n_coarse, 0] = coarse_leaves
                    u_from_qskel_list[acc_coarse:acc_coarse + n_coarse, 1] = box
                    acc_coarse += n_coarse

        self.leaf_uskel_from_qskel_list = leaf_uskel_from_qskel_list[:acc_coll_leaf]
        self.tree_uskel_from_qskel_list = tree_uskel_from_qskel_list[:acc_coll_tree]
        self.uskel_from_qskel_list      = np.vstack((self.leaf_uskel_from_qskel_list,\
            self.tree_uskel_from_qskel_list))
        self.uskel_from_q_list          = uskel_from_q_list[:acc_coarse]
        self.u_from_qskel_list          = u_from_qskel_list[:acc_coarse]

        boxes_level  = self.tree.get_boxes_level(self.root_level);
        leaves_above = self.tree.get_leaves_above(self.root_level);
        boxes_active = np.hstack((leaves_above,boxes_level))

        self.root_boxes      = boxes_active

        leaves            = self.tree.get_leaves()
        leaves_above      = self.tree.get_leaves_above(self.root_level+1)
        self.skel_leaves  = np.setdiff1d(leaves,leaves_above)

        if (verbose):
            print ("\t There are %d boxes on root level %d and %d leaves above and %d inds active" \
                % (boxes_level.shape[0], self.root_level, leaves_above.shape[0],\
                    np.sum(self.bs_list[self.root_boxes])))

        self.u_from_q_list = np.zeros((self.root_boxes.shape[0] ** 2,2),dtype=int)
        acc = 0
        for box in self.root_boxes:
            for b_prime in self.root_boxes:
                self.u_from_q_list[acc] = np.array([box,b_prime])
                acc += 1

    def precompute_inds(self,verbose=False):
        """
        Precomputes indices for efficiently embedding and assigning skeletonized data during upward
        and downward traversals in the tree.

        Parameters:
        - verbose: If True, prints additional debug information during computation.
        """

        ### indices needed to embed short vector into long vector for assigning leaves
        short_vec_inds = np.zeros(self.max_bs*self.nboxes,dtype=int)
        long_vec_inds  = np.zeros(self.max_bs*self.nboxes,dtype=int)

        offset = 0
        for leaf in self.tree.get_leaves():
            bs_box = int(self.bs_list[leaf])
            idx_box = self.idx_list[leaf,:bs_box]
            I_B = self.tree.get_box_inds(leaf)

            short_vec_inds[offset : offset + bs_box] = np.arange(bs_box) + leaf*self.max_bs
            long_vec_inds [offset : offset + bs_box] = I_B[idx_box]
            offset += bs_box


        self.short_vec_inds = short_vec_inds[:offset]
        self.long_vec_inds  = long_vec_inds[:offset]

        ### indices needed to assign skeleton points on upward and downward traversals
        lev_offset      = np.zeros(self.nlevels+1,dtype=int)
        box_skel_inds   = np.zeros(self.nlevels*self.max_bs*self.nboxes,dtype=int)
        child_skel_inds = np.zeros(self.nlevels*self.max_bs*self.nboxes,dtype=int)

        leaf_mask = self.tree.are_leaves(np.arange(self.nboxes))

        for lev in range(self.root_level,self.nlevels):

            offset = lev_offset[lev]
            level_boxes      = self.tree.get_boxes_level(lev)
            tree_level_boxes = level_boxes[np.logical_not(leaf_mask[level_boxes])]
            children_rows = self.tree.get_boxes_children_fullvec(tree_level_boxes)


            for box, child_row in zip(tree_level_boxes, children_rows):

                bs_box = int(self.bs_list[box])
                idx_box = self.idx_list[box,:bs_box]
                idx_box_order = np.argsort(idx_box)
                acc = 0
                for child in child_row:

                    if (child == -1):
                        continue
                    rank_child = int(self.rank_list[child])


                    box_skel_inds[offset : offset + rank_child]  = \
                    idx_box_order[acc:acc + rank_child] + box*self.max_bs

                    child_skel_inds[ offset: offset+ rank_child] = np.arange(rank_child) + child*self.max_bs

                    acc     += rank_child
                    offset  += rank_child
            lev_offset[lev+1] = offset

        self.box_skel_inds   = box_skel_inds[:offset]
        self.child_skel_inds = child_skel_inds[:offset]
        self.inds_lev_offset = lev_offset

    ##################################### SIMPLIFIED FMM #####################################

    def u_from_q(self,u_box,q_box,u_vec,q_vec):
        """
        Updates the incoming potential (`u_vec`) at the target box (`u_box`) 
        using the outgoing source (`q_vec`) from the source box (`q_box`).
        Used for translations at the root level.

        Parameters:
        - u_box: Index of the target box.
        - q_box: Index of the source box.
        - u_vec: Vector storing the potential at the target box.
        - q_vec: Vector storing the source strength at the source box.
        """
        XX_U, bs_U, _ = self.get_box_info(u_box)
        XX_Q, bs_Q, _ = self.get_box_info(q_box)
        geom_U = self._get_geometry(u_box, rank_only=False)
        geom_Q = self._get_geometry(q_box, rank_only=False)

        K_UQ               = self._kernel_matrix(XX_U, XX_Q, geom_U=geom_U, geom_Q=geom_Q)

        u_vec[:bs_U]      += K_UQ @ q_vec[:bs_Q]

    def uskel_from_qskel(self,u_box,q_box,uskel_vec,u_vec,qskel_vec,q_vec):
        """
        Updates both the incoming potentials (`u_vec`,`uskel_vec`) at the target box
        based on outgoing potentials (`qskel_vec`,`q_vec`) from the source box.

        Parameters:
        - u_box: Index of the target box.
        - q_box: Index of the source box.
        - uskel_vec: Skeletonized incoming potential for the target box.
        - u_vec: Incoming potential for the target box.
        - qskel_vec: Skeletonized outgoing potential for the source box.
        - q_vec: outgoing potential for the source box.
        """
        XX_U, bs_U, rank_U, _, proj_U = self.get_skel_box_info(u_box)
        XX_Q, bs_Q, rank_Q, _, proj_Q = self.get_skel_box_info(q_box)
        geom_U = self._get_geometry(u_box, rank_only=False)
        geom_Q = self._get_geometry(q_box, rank_only=False)

        K_UQ               = self._kernel_matrix(XX_U, XX_Q, geom_U=geom_U, geom_Q=geom_Q)

        u_vec[:bs_U]      += K_UQ @ q_vec[:bs_Q]
        uskel_vec[:rank_U]-= K_UQ[:rank_U][:,:rank_Q] @ qskel_vec[:rank_Q]

    def uskel_from_q(self,u_box,q_box,uskel_vec,u_vec,q_vec):
        """
        Updates both the incoming potentials (`u_vec`,`uskel_vec`) at the target box
        based on outgoing potential (`q_vec`) from the source box.

        Parameters:
        - u_box: Index of the target box.
        - q_box: Index of the source box.
        - uskel_vec: Skeletonized incoming potential for the target box.
        - u_vec: Incoming potential for the target box.
        - q_vec: outgoing potential for the source box.
        """
        XX_U, bs_U, rank_U, _, proj_U = self.get_skel_box_info(u_box)
        XX_Q,bs_Q,_                   = self.get_box_info(q_box)
        geom_U = self._get_geometry(u_box, rank_only=False)
        geom_Q = self._get_geometry(q_box, rank_only=False)

        K_UQ             = self._kernel_matrix(XX_U, XX_Q, geom_U=geom_U, geom_Q=geom_Q)

        u_vec[:bs_U]       += K_UQ @ q_vec[:bs_Q]
        uskel_vec[:rank_U] -= K_UQ[:rank_U] @ q_vec[:bs_Q]


    def u_from_qskel(self,u_box,q_box,u_vec,qskel_vec,q_vec):
        """
        Updates the incoming potentials (`u_vec`) at the target box
        based on outgoing potential (`q_vec`,`qskel_vec`) from the source box.

        Parameters:
        - u_box: Index of the target box.
        - q_box: Index of the source box.
        - u_vec: Incoming potential for the target box.
        - qskel_vec: outgoing skeletonized potential for the source box.
        - q_vec:     outgoing potential for the source box.
        """
        XX_U,bs_U,_                   = self.get_box_info(u_box)
        XX_Q, bs_Q, rank_Q, _, proj_Q = self.get_skel_box_info(q_box)
        geom_U = self._get_geometry(u_box, rank_only=False)
        geom_Q = self._get_geometry(q_box, rank_only=False)

        K_UQ = self._kernel_matrix(XX_U, XX_Q, geom_U=geom_U, geom_Q=geom_Q)

        u_vec[:bs_U]   += K_UQ @ q_vec[:bs_Q]
        u_vec[:bs_U]   -= K_UQ[:,:rank_Q] @ qskel_vec[:rank_Q]

    def compute_uskel(self,q_vec,qskel_vec,verbose):
        """
        Computes outgoing potential for all boxes.

        Parameters:
        - q_vec, qskel_vec: incoming potentials for all boxes.
        - verbose: If True, prints debug information.

        Returns:
        - u_vec,uskel_vec: outgoing potentials for all boxes.
        """

        uskel_vec = np.zeros((self.nboxes,self.max_bs),dtype=q_vec.dtype)
        u_vec     = np.zeros((self.nboxes,self.max_bs),dtype=q_vec.dtype)

        for box,colleague in self.uskel_from_qskel_list:
            self.uskel_from_qskel(box,colleague,uskel_vec[box],u_vec[box],\
                                  qskel_vec[colleague],q_vec[colleague])


        for box,colleague in self.u_from_q_list:
            self.u_from_q(box,colleague,u_vec[box],q_vec[colleague])

        for box,leaf in self.uskel_from_q_list:
            # by definition, leaf is level above
            # sources have not been skeletonized yet
            self.uskel_from_q(box,leaf,uskel_vec[box],u_vec[box],q_vec[leaf])

        # add contributions to leaves
        for leaf,z_box in self.u_from_qskel_list:
            self.u_from_qskel(leaf,z_box,u_vec[leaf],\
                              qskel_vec[z_box],q_vec[z_box])

        return u_vec,uskel_vec

    def get_qorig(self,q):
        """
        Reorganizes given charges q from long vector into box ordering.

        Parameters:
        - q: charges, as a long vector.

        Returns:
        - q_vec: original charges for all boxes.
        """

        q_orig = np.zeros(self.nboxes*self.max_bs,dtype=q.dtype)

        q_orig[self.short_vec_inds] = q[self.long_vec_inds]
        return q_orig.reshape(self.nboxes,self.max_bs)

    def get_result(self,result_vec):
        """
        Reorganizes computed potential from box ordering into a long vector.

        Parameters:
        - result_vec: computed potential for all boxes.

        Returns:
        - result: computed potential, as a long vector.
        """

        result = np.zeros(self.N,dtype=result_vec.dtype)
        result[self.long_vec_inds] = result_vec.flatten()[self.short_vec_inds]
        return result

    def addto_skel_lev(self,lev,vec_res,vec_mv,upward_pass):
        """
        Depending on the boolean value of `upward pass` parameter, 
        adds to `vec_res` from the children to the parents, or from the parents to the children.

        Parameters:
        - lev: level.
        - vec_res: updated vector.
        - upward_pass: boolean parameter
        """

        tmp = np.arange(self.inds_lev_offset[lev], self.inds_lev_offset[lev+1])

        if (tmp.shape[0] == 0):
            raise ValueError("no inds selected")

        box_skel_inds_lev   = self.box_skel_inds[ tmp ]
        child_skel_inds_lev = self.child_skel_inds[ tmp ]

        vec_mv  = vec_mv.reshape(self.nboxes*self.max_bs)
        vec_res = vec_res.reshape(self.nboxes*self.max_bs)
        if (upward_pass):
            vec_res[box_skel_inds_lev] += vec_mv[child_skel_inds_lev].copy()
        else:
            vec_res[child_skel_inds_lev] += vec_mv[box_skel_inds_lev].copy()

        vec_mv  = vec_mv.reshape(self.nboxes,self.max_bs)
        vec_res = vec_res.reshape(self.nboxes,self.max_bs)


    def apply_proj(self,boxes,vec_res,vec_mv,upward_pass):
        """
        Depending on the boolean value of `upward pass` parameter, 
        interpolates equivalent charges from children to the parents, or equivalent potentials the parents to the children.

        Parameters:
        - lev: level.
        - vec_res: updated vector.
        - vec_mv:  interpolated vector
        - upward_pass: boolean parameter
        """

        for box in boxes:

            _,bs_box,rank_box,_,proj_box = self.get_skel_box_info(box)

            if (upward_pass):
                vec_res[box,:rank_box]       += vec_mv[box,:rank_box] + proj_box @ vec_mv[box,rank_box:bs_box]
            else:
                vec_res[box,:rank_box]       += vec_mv[box,:rank_box]
                vec_res[box,rank_box:bs_box] += proj_box.T @ vec_mv[box,:rank_box]

    def get_tree_boxes_lev(self,lev):
        return self.tree_boxes[ self.tree_lev_sep[lev] : self.tree_lev_sep[lev+1] ]

    def compute_qskel(self,q_vec):
        """
        Computes the outgoing expansion (`qskel`,`q_vec`) for each box in an
        upward pass through the tree.

        Parameters:
        - q_vec: vector contains given charges at leaf boxes.

        Returns:
        - q_vec, q_skel: Outgoing potentials for each box.
        """

        q_skel = np.zeros(q_vec.shape)

        self.apply_proj(self.skel_leaves,q_skel,q_vec,upward_pass=True)

        for lev in range(self.nlevels-1,self.root_level,-1):

            tree_boxes = self.get_tree_boxes_lev(lev)

            self.apply_proj(tree_boxes,q_skel,q_vec,upward_pass=True)
            self.addto_skel_lev(lev-1,q_vec,q_skel,upward_pass=True)

        return q_vec,q_skel

    def resultvec_from_uskel(self,u_vec,uskel_vec):
        """
        Computes resulting potential at the leaf boxes
        using a downward pass through the tree.

        Parameters:
        - u_vec,uskel_vec: Outgoing potentials at each box.

        Returns:
        - u_vec: Computed potential at each box, including the result at the leaf boxes.
        """

        for lev in range(self.root_level+1,self.nlevels):

            tree_boxes = self.get_tree_boxes_lev(lev)

            self.addto_skel_lev(lev-1,uskel_vec,u_vec,upward_pass=False)
            self.apply_proj(tree_boxes,u_vec,uskel_vec,upward_pass=False)

        self.apply_proj(self.skel_leaves,u_vec,uskel_vec,upward_pass=False)
        return u_vec

    def matvec(self,q,verbose=False):
        """
        Performs a matrix-vector product using the hierarchical representation of the kernel matrix.

        Parameters:
        - q: Input vector representing the sources in the original ordering.
        - verbose: If True, prints timing information for each pass.

        Returns:
        - result: Output vector representing the computed potentials in the original ordering.
        """

        assert q.ndim == 1; assert q.shape[0] == self.N
        kernel_dtype = self.operator.numpy_matrix(
            np.zeros((1, self.ndim)),
            np.zeros((1, self.ndim)),
            geom_src=None if self.geometry is None else np.zeros((1,) + self.geometry.shape[1:], dtype=self.geometry.dtype),
            geom_trg=None if self.geometry is None else np.zeros((1,) + self.geometry.shape[1:], dtype=self.geometry.dtype),
            param=self.kappa,
        ).dtype

        q = q.astype(kernel_dtype)

        # Upward pass
        tic = time()
        q_vec = self.get_qorig(q)
        q_vec,qskel_vec = self.compute_qskel(q_vec)
        toc_qskel = time() - tic

        # Translate incoming expansions from outgoing expansions
        tic = time()
        u_vec,uskel_vec = self.compute_uskel(q_vec,qskel_vec,verbose=verbose)
        toc_uskel = time() - tic

        # Downward pass
        tic = time()
        result_vec = self.resultvec_from_uskel(u_vec,uskel_vec)
        result     = self.get_result(result_vec)
        toc_result = time() - tic

        if (verbose):
            print("\t Time for (upward pass, qskel to uskel, downward pass) = (%5.2f,%5.2f,%5.2f)" %\
                  (toc_qskel,toc_uskel,toc_result))
        return result
