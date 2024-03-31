# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch

from . import obj
from . import util

######################################################################################
# Base mesh class
######################################################################################
class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out

    def eval(self, params={}):
        return self

######################################################################################
# Mesh loeading helper
######################################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values

######################################################################################
# Compute unique edge list from attribute/vertex index list
######################################################################################
def compute_edges(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Eliminate duplicates and return inverse mapping
        return torch.unique(sorted_edges, dim=0, return_inverse=return_inverse)

######################################################################################
# Compute unique edge to face mapping from attribute/vertex index list
######################################################################################
def compute_edge_to_face_mapping(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

######################################################################################
# Align base mesh to reference mesh:move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.v_pos - (vmax + vmin) / 2 # Center mesh on origin
        v_pos = v_pos * scale                  # Rescale to unit size

        return Mesh(v_pos, base=mesh)

######################################################################################
# Center & scale mesh for rendering
######################################################################################
def center_by_reference(base_mesh, ref_aabb, scale):
    center = (ref_aabb[0] + ref_aabb[1]) * 0.5
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)

######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(imesh):

    i0 = imesh.t_pos_idx[:, 0]
    i1 = imesh.t_pos_idx[:, 1]
    i2 = imesh.t_pos_idx[:, 2]

    v0 = imesh.v_pos[i0, :]
    v1 = imesh.v_pos[i1, :]
    v2 = imesh.v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(imesh):
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
        tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
        vn_idx[i] = imesh.t_nrm_idx[:, i]

    tangents = torch.zeros_like(imesh.v_nrm)
    tansum   = torch.zeros_like(imesh.v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)


######################################################################################
# Displacement mapping
######################################################################################
# def displace(mesh, displacement_map, scale=1.0, keep_connectivity=True):


def mesh_displace(input, displacement_map, scale=1.0, keep_connectivity=True):
    imesh = input
    vd = torch.zeros_like(imesh.v_pos)
    vd_n = torch.zeros_like(imesh.v_pos)
    for i in range(0, 3):
        v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
        n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
        t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
        v_displ = v + n * scale * util.tex_2d(displacement_map, t)

        splat_idx = imesh.t_pos_idx[:, i, None].repeat(1, 3)
        vd.scatter_add_(0, splat_idx, v_displ)
        vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

    return Mesh(v_pos=vd/vd_n, base=imesh)



class mesh_op_displace:
    def __init__(self, input, displacement_map, scale=1.0, keep_connectivity=True):
        self.input = input
        self.displacement_map = displacement_map
        self.scale = scale
        self.keep_connectivity = keep_connectivity

    def displace(self):
        imesh = self.input
        vd = torch.zeros_like(imesh.v_pos)
        vd_n = torch.zeros_like(imesh.v_pos)
        for i in range(0, 3):
            v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
            n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
            t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
            v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)

            splat_idx = imesh.t_pos_idx[:, i, None].repeat(1, 3)
            vd.scatter_add_(0, splat_idx, v_displ)
            vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

        return Mesh(vd / vd_n, base=imesh)

    #     def eval(self, params={}):
    #         imesh = self.input.eval(params)
    #
    #         if self.keep_connectivity:
    #             vd = torch.zeros_like(imesh.v_pos)
    #             vd_n = torch.zeros_like(imesh.v_pos)
    #             for i in range(0, 3):
    #                 v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
    #                 n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
    #                 t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
    #                 v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)
    #
    #                 splat_idx = imesh.t_pos_idx[:, i, None].repeat(1, 3)
    #                 vd.scatter_add_(0, splat_idx, v_displ)
    #                 vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))
    #
    #             return Mesh(vd / vd_n, base=imesh)
    #         else:
    #             vd = torch.zeros([imesh.v_tex.shape[0], 3], dtype=torch.float32, device='cuda')
    #             vd_n = torch.zeros([imesh.v_tex.shape[0], 3], dtype=torch.float32, device='cuda')
    #             for i in range(0, 3):
    #                 v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
    #                 n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
    #                 t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
    #                 v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)
    #
    #                 splat_idx = imesh.t_tex_idx[:, i, None].repeat(1, 3)
    #                 vd.scatter_add_(0, splat_idx, v_displ)
    #                 vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))
    #
    #             return Mesh(vd / vd_n, mesh.t_tex_idx, base=imesh)
    #
    # return mesh_op_displace(mesh, displacement_map, scale, keep_connectivity)


def laplace_regularizer_const(opt_mesh, base_mesh=None):
    class mesh_op_laplace_regularizer_const:
        def __init__(self, opt_mesh, base_mesh):
            self.inputs = [opt_mesh, base_mesh]

            opt_mesh = opt_mesh.eval()
            self.nVerts = opt_mesh.v_pos.shape[0]
            t_pos_idx = opt_mesh.t_pos_idx.detach().cpu().numpy()

            # Build vertex neighbor rings
            vtx_n = [[] for _ in range(self.nVerts)]
            for tri in t_pos_idx:
                for (i0, i1) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    vtx_n[i0].append(i1)

            # Collect index/weight pairs to compute each Laplacian vector for each vertex.
            # Similar notation to https://mgarland.org/class/geom04/material/smoothing.pdf
            ix_j, ix_i, w_ij = [], [], []
            for i in range(self.nVerts):
                m = len(vtx_n[i])
                ix_i += [i] * m
                ix_j += vtx_n[i]
                w_ij += [1.0 / m] * m

            # Setup torch tensors
            self.ix_i = torch.tensor(ix_i, dtype=torch.int64, device='cuda')
            self.ix_j = torch.tensor(ix_j, dtype=torch.int64, device='cuda')
            self.w_ij = torch.tensor(w_ij, dtype=torch.float32, device='cuda')[:, None]

        def eval(self, params={}):
            opt_mesh = self.inputs[0].eval(params)
            base_mesh = self.inputs[1].eval(params) if self.inputs[1] is not None else None

            # differences or absolute version (see paper)
            if base_mesh is not None:
                v_pos = opt_mesh.v_pos - base_mesh.v_pos
            else:
                v_pos = opt_mesh.v_pos

            # Gather edge vertex pairs
            x_i = v_pos[self.ix_i, :]
            x_j = v_pos[self.ix_j, :]

            # Compute Laplacian differences: (x_j - x_i) * w_ij
            term = (x_j - x_i) * self.w_ij

            # Sum everyhing
            term = util.segment_sum(term, self.ix_i)

            return torch.mean(term ** 2)

    return mesh_op_laplace_regularizer_const(opt_mesh, base_mesh)