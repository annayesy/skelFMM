import numpy as np
from . import operators, util_batched
import torch
from torch.profiler import record_function

class SkelFMMBatched(torch.nn.Module):
	
	def __init__(self,tree,operator,kappa:float=0.0):
		super().__init__()
		
		self.N                   = tree.N
		self.tree                = tree
		self.nlevels             = tree.nlevels
		self.nboxes              = tree.nboxes

		self.operator           = operators._require_operator(operator)
		self.kappa               = kappa
		
		self.ndim                = tree.ndim
		self.level_sep           = self.tree.level_sep
		self.kernel_backend, self.kernel_apply = self._select_kernel_backend()
		self.kernel_const = self._get_kernel_const()

	def _select_kernel_backend(self):
		if util_batched.PYKEOPS_AVAILABLE and self.operator.has_keops:
			return "pykeops", lambda q, xx_u, xx_q, geom_u, geom_q, kappa, self_bool: self.operator.keops_apply(
				q,
				xx_u,
				xx_q,
				geom_trg=geom_u,
				geom_src=geom_q,
				param=kappa,
				self_interaction=self_bool,
			)

		return "torch", lambda q, xx_u, xx_q, geom_u, geom_q, kappa, self_bool: self.operator.torch_apply(
			q,
			xx_u,
			xx_q,
			geom_trg=geom_u,
			geom_src=geom_q,
			param=kappa,
			self_interaction=self_bool,
		)

	def _get_kernel_const(self):
		# Empirical safety factors for translation chunk sizing. The explicit torch
		# kernels materialize dense pairwise blocks, so they need a larger memory
		# multiplier than the PyKeOps lazy kernels.
		torch_consts = {
			(2, False): 1.05,
			(2, True): 2.05,
			(3, False): 2.55,
			(3, True): 2.55,
		}
		pykeops_consts = {
			(2, False): 1.00,
			(3, False): 1.00,
			(3, True): 1.00,
		}

		key = (self.ndim, self.kappa > 0)
		if self.kernel_backend == "pykeops":
			return pykeops_consts.get(key, torch_consts[key])
		return torch_consts[key]
		
	@torch.jit.export
	def nbytes_proj(self)->int:
		return self.proj_list.nelement() * self.proj_list.element_size() + self.proj_list_tree.nelement() * self.proj_list_tree.element_size()

	@torch.jit.export
	def nbytes_lists(self)->int:
		
		def nbytes(tt):
			if tt is None:
				return 0
			return tt.nelement() * tt.element_size()

		acc_sum  = 0
		acc_sum += nbytes(self.XX_list)
		acc_sum += nbytes(self.rank_list)
		acc_sum += nbytes(self.bs_list)
		acc_sum += nbytes(self.uskel_from_qskel_list)
		acc_sum += nbytes(self.uskel_from_q_list)
		acc_sum += nbytes(self.u_from_qskel_list)
		acc_sum += nbytes(self.box_skel_inds)
		acc_sum += nbytes(self.child_skel_inds)
		acc_sum += nbytes(self.inds_lev_offset)
		return acc_sum
		
	@torch.jit.export
	def build_from_copy(self,skel_fmm,use_single_precision=False)->None:

		device = torch.device('cpu')
		if (torch.cuda.is_available()):
			device = torch.device('cuda')
		
		def store(obj, on_cpu=False):
			if obj is None:
				return None

			tensor = torch.tensor(obj,requires_grad=False)

			if(not on_cpu):
				tensor = tensor.to(device)

			if (tensor.dtype==torch.float64 and use_single_precision):
				tensor = tensor.to(torch.float32)

			elif (tensor.dtype==torch.complex128 and use_single_precision):
				tensor = tensor.to(torch.complex64)
			return tensor
			
		self.max_rank      = skel_fmm.max_rank
		self.max_bs        = skel_fmm.max_bs
		self.min_rank      = skel_fmm.min_rank
		
		self.max_rank_leaf = skel_fmm.max_rank_leaf
		self.max_bs_leaf   = skel_fmm.max_bs_leaf
		self.geometry      = skel_fmm.geometry
		
		XX_copy = np.random.rand(*skel_fmm.XX_list.shape)
		for box in range(skel_fmm.nboxes):
			XX_copy[box,:skel_fmm.bs_list[box]] = skel_fmm.XX_list[box,:skel_fmm.bs_list[box]]
		
		## from build stage
		self.XX_list     = store(XX_copy)
		self.geom_list   = store(skel_fmm.geom_list)
		self.proj_list   = store(skel_fmm.proj_list,on_cpu=True)
		self.rank_list   = store(skel_fmm.rank_list)
		self.bs_list     = store(skel_fmm.bs_list)
		self.root_boxes  = store(skel_fmm.root_boxes,on_cpu=True)

		XX_copy = np.random.rand(*skel_fmm.XX_list.shape)
		for box in range(skel_fmm.nboxes):
			XX_copy[box,:skel_fmm.rank_list[box]] = skel_fmm.XX_list[box,:skel_fmm.rank_list[box]]

		self.XX_rank_list= store(XX_copy)
		geom_rank_copy = None
		if skel_fmm.geom_list is not None:
			geom_rank_copy = np.zeros_like(skel_fmm.geom_list)
			for box in range(skel_fmm.nboxes):
				geom_rank_copy[box,:skel_fmm.rank_list[box]] = skel_fmm.geom_list[box,:skel_fmm.rank_list[box]]
		self.geom_rank_list = store(geom_rank_copy)
		
		self.isleaf_list = store(skel_fmm.tree.are_leaves(np.arange(skel_fmm.nboxes)),on_cpu=True)
		self.skel_leaves = store(skel_fmm.skel_leaves,on_cpu=True)
		
		self.tree_boxes   = store(skel_fmm.tree_boxes,on_cpu=True)
		self.tree_lev_sep = store(skel_fmm.tree_lev_sep,on_cpu=True)
		self.proj_list_tree = store(skel_fmm.proj_list_tree,on_cpu=True)
		
		## lists for matvec
		self.uskel_from_qskel_list      = store(skel_fmm.uskel_from_qskel_list)
		self.leaf_uskel_from_qskel_list = store(skel_fmm.leaf_uskel_from_qskel_list)
		self.tree_uskel_from_qskel_list = store(skel_fmm.tree_uskel_from_qskel_list)

		self.uskel_from_q_list     = store(skel_fmm.uskel_from_q_list)
		self.u_from_qskel_list     = store(skel_fmm.u_from_qskel_list)
		self.u_from_q_list         = store(skel_fmm.u_from_q_list)
		
		## precomputed indices
		self.short_vec_inds,self.long_vec_inds = store(skel_fmm.short_vec_inds,on_cpu=True),\
		store(skel_fmm.long_vec_inds,on_cpu=True)
		
		self.box_skel_inds   = store(skel_fmm.box_skel_inds,on_cpu=True)
		self.child_skel_inds = store(skel_fmm.child_skel_inds,on_cpu=True)
		self.inds_lev_offset = store(skel_fmm.inds_lev_offset,on_cpu=True)
		
		## these calculations done on GPU
		self.root_level     = skel_fmm.root_level
		self.device         = device
		self.tol            = skel_fmm.tol
		print("\t Allocated %5.2f GB on GPU for storage" % (torch.cuda.memory_allocated(self.device)*1e-9))


#################################################################################################################################
##################################################### translation operators #####################################################
		
	@torch.jit.export
	def apply_kernel_batched_matrices(self,vec,uq_boxes:torch.Tensor,\
		uskel_bool:bool,qskel_bool:bool,self_bool:bool)->torch.Tensor:

		u_boxes = uq_boxes[:,0]; q_boxes = uq_boxes[:,1]
		max_bs_uboxes   = torch.max(self.bs_list[u_boxes])
		max_rank_uboxes = torch.max(self.rank_list[u_boxes])

		max_bs_qboxes   = torch.max(self.bs_list[q_boxes])
		max_rank_qboxes = torch.max(self.rank_list[q_boxes])

		XX_bs   = self.XX_list
		XX_rank = self.XX_rank_list

		if uskel_bool:
			XX_u = torch.index_select(XX_rank[:,:max_rank_uboxes],0,u_boxes)
		else:
			XX_u = torch.index_select(XX_bs[:,:max_bs_uboxes],0,u_boxes)

		if (qskel_bool):
			XX_q = torch.index_select(XX_rank[:,:max_rank_qboxes],0,q_boxes)
		else:
			XX_q = torch.index_select(XX_bs[:,:max_bs_qboxes],0,q_boxes)

		geom_u = None
		geom_q = None
		if self.geom_list is not None:
			if uskel_bool:
				geom_u = torch.index_select(self.geom_rank_list[:,:max_rank_uboxes],0,u_boxes)
			else:
				geom_u = torch.index_select(self.geom_list[:,:max_bs_uboxes],0,u_boxes)

			if qskel_bool:
				geom_q = torch.index_select(self.geom_rank_list[:,:max_rank_qboxes],0,q_boxes)
			else:
				geom_q = torch.index_select(self.geom_list[:,:max_bs_qboxes],0,q_boxes)

		return self.kernel_apply(vec, XX_u, XX_q, geom_u, geom_q, self.kappa, self_bool)

	@torch.jit.export
	def addto_translation(self,uq_boxes:torch.Tensor,u_vec:torch.Tensor,q_vec:torch.Tensor,\
		max_bs_loc:int,self_bool:bool):

		ngrouped = uq_boxes.shape[1]
		if (ngrouped == 0):
			return
		device  = self.XX_list.device

		u_boxes = uq_boxes[:,0,0]; q_boxes = uq_boxes[:,0,1]
		npairs_loc      = u_boxes.shape[0]
		max_bs_uboxes   = torch.max(self.bs_list[u_boxes])
		max_bs_qboxes   = torch.max(self.bs_list[q_boxes])

		### temporary copy of q_vec
		q_boxes_flat =  torch.flatten(uq_boxes[:,:,1])
		tmp = torch.index_select(q_vec[...,:max_bs_qboxes], dim=0, index=q_boxes_flat)
		tmp = tmp.reshape(npairs_loc,ngrouped,max_bs_qboxes)
		tmp = torch.transpose(tmp,-1,-2).contiguous()

		res_vec = torch.zeros(npairs_loc,self.max_bs,ngrouped,device=device,\
			 dtype=u_vec.dtype,requires_grad=False)

		res_tmp = self.apply_kernel_batched_matrices(tmp,uq_boxes[:,0],False,False,self_bool)

		res_vec[:,:max_bs_uboxes] +=  res_tmp
		res_vec = torch.transpose(res_vec,-1,-2)
		res_vec = res_vec.reshape(npairs_loc*ngrouped,self.max_bs)
	
		u_vec.index_add_(0, torch.flatten(uq_boxes[:,:,0]),res_vec)

	@torch.jit.export
	def subtractskel_translation(self,uq_boxes:torch.Tensor,u_vec:torch.Tensor,u_skel:torch.Tensor,\
	 q_vec:torch.Tensor,q_skel:torch.Tensor, max_bs_loc:int,self_bool:bool):

		if (u_skel is None and q_skel is None):
			return

		npairs_loc = uq_boxes.shape[0]
		ngrouped   = uq_boxes.shape[1]
		if (ngrouped == 0):
			return
		u_boxes    = uq_boxes[:,0,0]; q_boxes    = uq_boxes[:,0,1]
		device     = self.XX_list.device

		uskel_bool = u_skel is not None
		qskel_bool = q_skel is not None

		if (uskel_bool):
			max_bs_uboxes = torch.max(self.rank_list[u_boxes])
		else:
			max_bs_uboxes = torch.max(self.bs_list[u_boxes])

		if (qskel_bool):
			max_bs_qboxes = torch.max(self.rank_list[q_boxes])
		else:
			max_bs_qboxes = torch.max(self.bs_list[q_boxes])

		res_vec = torch.zeros(npairs_loc,self.max_bs,ngrouped,device=device,\
			 dtype=q_vec.dtype,requires_grad=False)

		q_boxes_flat = torch.flatten(uq_boxes[:,:,1])
		if (qskel_bool):
			tmp = torch.index_select(q_skel[..., :max_bs_qboxes ],dim = 0, index=q_boxes_flat)
			tmp = tmp.reshape(npairs_loc,ngrouped,max_bs_qboxes)
			tmp = util_batched.zero_col_bnd(tmp,self.rank_list[q_boxes])
		else:
			tmp = torch.index_select(q_vec [ : , :max_bs_qboxes], dim = 0, index=q_boxes_flat)
			tmp = tmp.reshape(npairs_loc, ngrouped,max_bs_qboxes)
		tmp     = torch.transpose(tmp,-1,-2).contiguous()

		res_tmp = self.apply_kernel_batched_matrices(tmp, uq_boxes[:,0],\
			uskel_bool,qskel_bool,self_bool)

		res_vec[:,:max_bs_uboxes] -= res_tmp
		res_vec = torch.transpose(res_vec,-1,-2)
		if (uskel_bool):
			res_vec = util_batched.zero_col_bnd(res_vec,self.rank_list[u_boxes])
		res_vec = res_vec.reshape(npairs_loc*ngrouped,self.max_bs)

		if (u_skel is not None):
			u_skel.index_add_(0, torch.flatten(uq_boxes[:,:,0]),res_vec)
		else:
			u_vec.index_add_(0, torch.flatten(uq_boxes[:,:,0]),res_vec)
		
	@torch.jit.export
	def translation_helper(self,uq_boxes:torch.Tensor,u_vec:torch.Tensor,u_skel:torch.Tensor,\
	 q_vec:torch.Tensor,q_skel:torch.Tensor, max_bs_loc:int,self_bool:bool):

		if (u_skel is None and q_skel is None):
			self.addto_translation(uq_boxes,u_vec,q_vec,max_bs_loc,self_bool)
		else:
			self.subtractskel_translation(uq_boxes,u_vec,u_skel,q_vec,q_skel,\
				max_bs_loc,self_bool)
	 
	def translation(self, uq_boxes,u_vec, u_skel, q_vec, q_skel,max_bs_loc,verbose=True):
		
		def translate(uq_tmp,bnd0,self_bool):
			bytes_free  = util_batched.get_bytes_available(self.device)

			nbytes_pair = self.kernel_const * max_bs_loc * max_bs_loc * u_vec.element_size()

			npairs_next = max( 50 if self.ndim == 3 else 500, int(bytes_free / nbytes_pair ))

			bnd1        = min( npairs, bnd0 +  npairs_next)
			uq_loc      = uq_tmp[bnd0:bnd1]
			not_univ    = uq_tmp.ndim == 2

			if (not_univ):
				self.translation_helper(uq_loc.unsqueeze(1),\
					u_vec,u_skel, q_vec,q_skel,\
					max_bs_loc,self_bool)
				
			else:
				
				self.translation_helper(uq_loc,\
					u_vec,u_skel,q_vec,q_skel,\
					max_bs_loc,self_bool)
			return bnd1

		# pre-coded constants based on kernel implementation
		npairs = uq_boxes.shape[0]; ndim = self.XX_list.shape[-1]
		if (npairs == 0):
			return
			
		print('\t Bs loc is %d, Batched kernel const is %5.2f' % \
			  (max_bs_loc,self.kernel_const)) if verbose else None

		if (uq_boxes.ndim == 2):
			bool_self   = uq_boxes[:,0] == uq_boxes[:,1]
		elif (uq_boxes.ndim == 3 and uq_boxes.shape[1]>0):
			bool_self   = uq_boxes[:,0,0] == uq_boxes[:,0,1]
		else:
			return

		npairs_self = torch.sum(bool_self).item()
		npairs_neigh= npairs - npairs_self
		uq_self     = uq_boxes[bool_self]
		uq_neigh    = uq_boxes[torch.logical_not(bool_self)]

		ndone = 0
		while (ndone < npairs_neigh):
			bnd = translate(uq_neigh,ndone,False)
			ndone = bnd

		ndone = 0
		while (ndone < npairs_self):
			bnd = translate(uq_self,ndone,True)
			ndone = bnd
			
		if(verbose and torch.cuda.is_available()): 
			bytes_allocated = torch.cuda.memory_allocated(self.device)
			print("\t %5.2f GB reserved and %5.2f GB allocated by Pytorch" %\
				  (torch.cuda.max_memory_reserved(self.device)*1e-9, bytes_allocated*1e-9))

#################################################################################################################################
##################################################### FMM level traversal #######################################################

	@torch.jit.export
	def get_tree_boxes(self,lev:int)->torch.Tensor:
		return self.tree_boxes[ self.tree_lev_sep[lev] : self.tree_lev_sep[lev+1] ]
		
	@torch.jit.export
	def addto_skel_lev(self,lev:int,vec_res:torch.Tensor,vec_mv:torch.Tensor,upward_pass:bool)->torch.Tensor:
		
		bnd0                = self.inds_lev_offset[lev]
		bnd1                = self.inds_lev_offset[lev+1]
		box_skel_inds_lev   = self.box_skel_inds[ bnd0:bnd1 ]
		child_skel_inds_lev = self.child_skel_inds[ bnd0:bnd1 ]
		
		vec_mv  = vec_mv.reshape(self.nboxes*self.max_bs)
		vec_res = vec_res.reshape(self.nboxes*self.max_bs)

		if (upward_pass):

			vec_res.index_add_(0,box_skel_inds_lev,vec_mv[child_skel_inds_lev].contiguous())
			
		else:
			vec_res.index_add_(0,child_skel_inds_lev,vec_mv[box_skel_inds_lev].contiguous())

		vec_res = vec_res.reshape(self.nboxes,self.max_bs)    
		vec_mv  = vec_mv.reshape(self.nboxes,self.max_bs)

	@torch.jit.export
	def apply_proj_levtree(self,lev,vec,upward_pass):

		tree_boxes = self.get_tree_boxes(lev)
		min_rank   = self.min_rank; max_bs = self.max_bs; max_rank = self.max_rank
		
		if (tree_boxes.shape[0] == 0):
			return
		
		vec  = vec.unsqueeze(-1)

		if (upward_pass):
			vec[tree_boxes,:max_rank] += self.proj_list_tree[self.tree_lev_sep[lev] : self.tree_lev_sep[lev+1]\
															 ,:max_rank,:(max_bs-min_rank)]\
			@ vec[tree_boxes,min_rank:max_bs]
		
		else:
			tmp = torch.transpose(vec[tree_boxes,:max_rank],-1,-2) @ \
			self.proj_list_tree[self.tree_lev_sep[lev] : self.tree_lev_sep[lev+1],\
								:max_rank,:(max_bs-min_rank)]
			
			vec[tree_boxes,min_rank:max_bs] += torch.transpose(tmp,-1,-2)
			
		vec = vec.squeeze(-1)

	@torch.jit.export
	def apply_proj_leaves(self,vec:torch.Tensor,upward_pass:bool):

		vec  = vec.unsqueeze(-1)
		if (upward_pass):
			vec[:, :self.max_rank_leaf] += self.proj_list @ \
			vec[:,self.min_rank:self.max_bs_leaf]
		else:
			vec[:,self.min_rank:self.max_bs_leaf] += torch.transpose( \
				torch.transpose(vec[:,:self.max_rank_leaf],-1,-2) @ self.proj_list, -1,-2)

		vec = vec.squeeze(-1)


	@torch.jit.export
	def get_qorig(self,q:torch.Tensor)->torch.Tensor:
		
		q_orig = torch.zeros(self.nboxes*self.max_bs,dtype=q.dtype,requires_grad=False)
		
		q_orig.index_add_(0,self.short_vec_inds,q[self.long_vec_inds].contiguous())
		return q_orig.reshape(self.nboxes,self.max_bs)
		
		
	def compute_qskel(self,q_vec):
		
		q_skel     = torch.zeros(q_vec.shape,dtype=q_vec.dtype)
		q_skel[self.skel_leaves] += q_vec[self.skel_leaves]
		self.apply_proj_leaves(q_skel,upward_pass=True)

		for lev in range(self.nlevels-1,self.root_level,-1):

			tree_boxes = self.get_tree_boxes(lev)
			q_skel[tree_boxes] += q_vec[tree_boxes]
			self.apply_proj_levtree(lev,q_skel,upward_pass=True)
				
			self.addto_skel_lev(lev-1,q_vec,q_skel,upward_pass=True)

		# move vectors to device
		q_vec = q_vec.to(self.device,non_blocking=True); q_skel = q_skel.to(self.device,non_blocking=True);
			
		return q_vec,q_skel
	
	def resultvec_from_uskel(self,u_vec,uskel_vec):
		u_vec    = u_vec.cpu(); uskel_vec = uskel_vec.cpu()
			
		for lev in range(self.root_level,self.nlevels):
			
			tree_boxes = self.get_tree_boxes(lev)
			self.addto_skel_lev(lev-1,uskel_vec,u_vec,upward_pass=False)
			
			self.apply_proj_levtree(lev,uskel_vec,upward_pass=False)
			u_vec[tree_boxes] += uskel_vec[tree_boxes]
			
		self.apply_proj_leaves(uskel_vec,upward_pass=False)
		u_vec[self.skel_leaves] += uskel_vec[self.skel_leaves]
		
		return u_vec
	

	@torch.jit.export
	def get_result(self,result_vec:torch.Tensor)->torch.Tensor:
		result = torch.zeros(self.N,dtype=result_vec.dtype,requires_grad=False)
		result.index_add_(0,self.long_vec_inds, result_vec.flatten()[self.short_vec_inds].contiguous())
		return result


#################################################################################################################################
##################################################### FMM translation calls #####################################################
	
	def addto_uvec(self,u_vec,q_vec,verbose):

		# combine lists
		u_from_q_bs = torch.concat(( self.u_from_qskel_list, self.u_from_q_list,\
		self.tree_uskel_from_qskel_list, self.uskel_from_q_list ))

		with record_function("add tree translations"):

			self.translation(u_from_q_bs, u_vec,None, q_vec,None,\
				max_bs_loc=self.max_bs,verbose=verbose)

		with record_function("add leaf translations"):

			self.translation(self.leaf_uskel_from_qskel_list,\
				u_vec,None, q_vec, None, \
				max_bs_loc=self.max_bs_leaf, verbose=verbose)
			
		
	def subtract_from_uskel(self,u_vec,u_skel,q_vec,q_skel,verbose):
	
		u_from_q_bs = torch.concat(( self.tree_uskel_from_qskel_list, self.leaf_uskel_from_qskel_list))

		with record_function("subtract uskel,qskel translations"):

			self.translation(u_from_q_bs,\
				u_vec,u_skel, q_vec,q_skel, \
				max_bs_loc = self.max_rank, verbose=verbose)

		# leaf to tree (and vice-versa) operations

		with record_function("subtract tree-leaf translations"):
	
			self.translation(self.uskel_from_q_list,\
				u_vec,u_skel, q_vec, None, \
				max_bs_loc = self.max_bs,verbose=verbose)

			self.translation(self.u_from_qskel_list,\
				u_vec, None, q_vec, q_skel,\
				max_bs_loc = self.max_bs,verbose=False)
#################################################################################################################################
##################################################### FMM definition  ###########################################################
	
	
	def matvec(self,q,verbose=False):

		if (q.dtype == torch.float32 and self.kappa>0):
			q = q.to(torch.complex64)
		elif (q.dtype == torch.float64 and self.kappa>0):
			q = q.to(torch.complex128)

		with record_function("get_q,qskel"):

			q_vec = self.get_qorig(q)
			q_vec, q_skel = self.compute_qskel(q_vec)

		u_skel  = torch.zeros(self.nboxes,self.max_bs,dtype=q_skel.dtype,\
			device=self.device,requires_grad=False)
		u_vec   = torch.zeros(self.nboxes,self.max_bs,dtype=q_skel.dtype,\
			device=self.device,requires_grad=False)

		self.addto_uvec(u_vec, q_vec, verbose=verbose)
		self.subtract_from_uskel(u_vec,u_skel,q_vec,q_skel,verbose=verbose)

		with record_function("get_resultvec"):

			result_vec = self.resultvec_from_uskel(u_vec, u_skel)
			result = self.get_result(result_vec)

		return result
