##
# @file   PlaceObj.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Placement model class defining the placement objective.
#

import os
import sys
import time
import math
import numpy as np
import itertools
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import dreamplace.ops.density_overflow.density_overflow as density_overflow
import dreamplace.ops.electric_potential.electric_overflow as electric_overflow
import dreamplace.ops.electric_potential.electric_potential as electric_potential
import dreamplace.ops.density_potential.density_potential as density_potential
import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pin_utilization.pin_utilization as pin_utilization
import dreamplace.ops.nctugr_binary.nctugr_binary as nctugr_binary
import dreamplace.ops.adjust_node_area.adjust_node_area as adjust_node_area
import dreamplace.ops.gift_init.gift_init as gift_init


class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """
    def __init__(self, placedb, data_collections, op_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []
        if len(placedb.regions) > 0:
            self.movablenode2fence_region_map_clamp = (
                data_collections.node2fence_region_map[: placedb.num_movable_nodes]
                .clamp(max=len(placedb.regions))
                .long()
            )
            self.filler2fence_region_map = torch.zeros(
                placedb.num_filler_nodes, device=data_collections.pos[0].device, dtype=torch.long
            )
            for i in range(len(placedb.regions) + 1):
                filler_beg, filler_end = self.placedb.filler_start_map[i : i + 2]
                self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            # The preconditioning step in python is time-consuming, as in each gradient
            # pass, the total net weight should be re-calculated.
            sum_pin_weights_in_nodes = self.op_collections.pws_op(self.data_collections.net_weights)
            if density_weight.size(0) == 1:
                precond = (sum_pin_weights_in_nodes
                    + self.alpha * density_weight * self.data_collections.node_areas
                )
            else:
                ### only precondition the non fence region
                node_areas = self.data_collections.node_areas.clone()

                mask = self.data_collections.node2fence_region_map[: self.placedb.num_movable_nodes] >= len(
                    self.placedb.regions
                )
                node_areas[: self.placedb.num_movable_nodes].masked_scatter_(
                    mask, node_areas[: self.placedb.num_movable_nodes][mask] * density_weight[-1]
                )
                filler_beg, filler_end = self.placedb.filler_start_map[-2:]
                node_areas[
                    self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_beg : self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_end
                ] *= density_weight[-1]
                precond = sum_pin_weights_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0 : self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)

            ### stop gradients for terminated electric field
            if update_mask is not None:
                grad = grad.view(2, -1)
                update_mask = ~update_mask
                movable_mask = update_mask[self.movablenode2fence_region_map_clamp]
                filler_mask = update_mask[self.filler2fence_region_map]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[0, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad[1, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)
            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                grad[0, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad[1, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad = grad.view(-1)
            self.iteration += 1

            # only work in benchmarks without fence region, assume overflow has been updated
            if len(self.placedb.regions) > 0 and self.overflows and self.overflows[-1].max() < 0.3 and self.alpha < 1024:
                if (self.iteration % 20) == 0:
                    self.alpha *= 2
                    logging.info(
                        "preconditioning alpha = %g, best_overflow %g, overflow %g"
                        % (self.alpha, self.best_overflow, self.overflows[-1])
                    )

        return grad


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """
    def __init__(self, density_weight, params, placedb, data_collections,
                 op_collections, global_place_params):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObj, self).__init__()

        ### quadratic penalty
        self.density_quad_coeff = 2000
        self.init_density = None
        ### increase density penalty if slow convergence
        self.density_factor = 1

        if(len(placedb.regions) > 0):
            ### fence region will enable quadratic penalty by default
            self.quad_penalty = True
        else:
            ### non fence region will use first-order density penalty by default
            self.quad_penalty = False

        ### fence region
        ### update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None
        self.fix_nodes_mask = None 
        if len(placedb.regions) > 0:
            ### for subregion rough legalization, once stop updating, perform immediate greddy legalization once
            ### this is to avoid repeated legalization
            ### 1 represents already legal
            self.legal_mask = torch.zeros(len(placedb.regions) + 1)

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params

        self.gpu = params.gpu
        self.data_collections = data_collections
        self.op_collections = op_collections
        if len(placedb.regions) > 0:
            ### different fence region needs different density weights in multi-electric field algorithm
            self.density_weight = torch.tensor(
                [density_weight]*(len(placedb.regions)+1),
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        else:
            self.density_weight = torch.tensor(
                [density_weight],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        ### Note: even for multi-electric fields, they use the same gamma
        num_bins_x = global_place_params["num_bins_x"] if "num_bins_x" in global_place_params and global_place_params["num_bins_x"] > 1 else placedb.num_bins_x
        num_bins_y = global_place_params["num_bins_y"] if "num_bins_y" in global_place_params and global_place_params["num_bins_y"] > 1 else placedb.num_bins_y
        name = "Global placement: %dx%d bins by default" % (num_bins_x, num_bins_y)
        logging.info(name)
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        self.gamma = torch.tensor(10 * self.base_gamma(params, placedb),
                                  dtype=self.data_collections.pos[0].dtype,
                                  device=self.data_collections.pos[0].device)

        # compute weighted average wirelength from position

        name = "%dx%d bins" % (num_bins_x, num_bins_y)
        self.name = name

        if global_place_params["wirelength"] == "weighted_average":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_weighted_average_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        elif global_place_params["wirelength"] == "logsumexp":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_logsumexp_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        else:
            assert 0, "unknown wirelength model %s" % (
                global_place_params["wirelength"])

        self.op_collections.density_overflow_op = self.build_electric_overflow(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y)

        self.op_collections.density_op = self.build_electric_potential(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=name)
        ### build multiple density op for multi-electric field
        if len(self.placedb.regions) > 0:
            self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op = self.build_multi_fence_region_density_op()
        self.op_collections.update_density_weight_op = self.build_update_density_weight(
            params, placedb)
        self.op_collections.precondition_op = self.build_precondition(
            params, placedb, self.data_collections, self.op_collections)
        self.op_collections.noise_op = self.build_noise(
            params, placedb, self.data_collections)
        if params.routability_opt_flag:
            # compute congestion map, RISA/RUDY congestion map
            self.op_collections.route_utilization_map_op = self.build_route_utilization_map(
                params, placedb, self.data_collections)
            self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
                params, placedb, self.data_collections)
            self.op_collections.nctugr_congestion_map_op = self.build_nctugr_congestion_map(
                params, placedb, self.data_collections)
            # adjust instance area with congestion map
            self.op_collections.adjust_node_area_op = self.build_adjust_node_area(
                params, placedb, self.data_collections)
            
        #创建 rass_state，填充 JSON 配置的参数，并把这些配置传进 GiFt 初始化器
        self.rass_state = None
        # 初始化 rass_state 为 None，表示目前还没有 RASS 状态。
        if params.rass_place_flag:
            # 如果启用了 RASS 逻辑（即 params.rass_place_flag 为 True），进入 RASS 初始化流程。
            if params.rass_route_weight > 0 and self.op_collections.route_utilization_map_op is None:
                # 如果启用了路由利用率权重（params.rass_route_weight > 0），并且当前没有创建路由利用率图算子
                self.op_collections.route_utilization_map_op = self.build_route_utilization_map(
                    params, placedb, self.data_collections)
                # 调用 build_route_utilization_map 函数，基于当前的 params、placedb 和 data_collections 创建路由利用率图算子，并保存到 op_collections 中
            if params.rass_pin_weight > 0 and self.op_collections.pin_utilization_map_op is None:
                        # 如果启用了引脚利用率权重（params.rass_pin_weight > 0），并且当前没有创建引脚利用率图算子
                self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
                    params, placedb, self.data_collections)
                # 调用 build_pin_utilization_map 函数，基于当前的 params、placedb 和 data_collections 创建引脚利用率图算子，并保存到 op_collections 中
            self.rass_state = self.build_rass_state(params, placedb, self.data_collections)
                # 调用 build_rass_state 函数，构建 RASS 状态，生成风险图，并将其保存在 rass_state 中
            self.data_collections.rass_risk_map = self.rass_state["risk_map"]
            # 将构建出的风险图（risk_map）保存到 data_collections 中，供后续使用
        else:
            self.data_collections.rass_risk_map = None
             # 如果未启用 RASS，则将 rass_risk_map 设置为 None，表示不使用风险图

        rass_options = None
        if params.rass_place_flag and self.rass_state is not None:
            rass_options = self._compose_rass_options(params, placedb)
        #如果启用了 RASS（params.rass_place_flag 为 True）且成功构建了 rass_state，那么调用 _compose_rass_options 函数，生成包含 RASS 配置的字典 rass_options。如果条件不满足（RASS 未启用或没有构建 rass_state），rass_options 会保持为 None。

        # GiFt initialization 
        if params.global_place_flag and params.gift_init_flag: 
            self.op_collections.gift_init_op = gift_init.GiFtInit(
                    flat_netpin=self.data_collections.flat_net2pin_map, 
                    netpin_start=self.data_collections.flat_net2pin_start_map, 
                    pin2node_map=self.data_collections.pin2node_map, 
                    net_weights=self.data_collections.net_weights, 
                    net_mask=self.data_collections.net_mask_ignore_large_degrees, 
                    xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh,
                    num_nodes=placedb.num_physical_nodes,  
                    num_movable_nodes=placedb.num_movable_nodes, 
                    node_size_x=self.data_collections.node_size_x[:placedb.num_physical_nodes],
                    node_size_y=self.data_collections.node_size_y[:placedb.num_physical_nodes],
                    scale=params.gift_init_scale,
                    rass_options=rass_options
                    ) 

        self.Lgamma_iteration = global_place_params["iteration"]
        if 'Llambda_density_weight_iteration' in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params[
                'Llambda_density_weight_iteration']
        else:
            self.Llambda_density_weight_iteration = 1
        if 'Lsub_iteration' in global_place_params:
            self.Lsub_iteration = global_place_params['Lsub_iteration']
        else:
            self.Lsub_iteration = 1
        if 'routability_Lsub_iteration' in global_place_params:
            self.routability_Lsub_iteration = global_place_params[
                'routability_Lsub_iteration']
        else:
            self.routability_Lsub_iteration = self.Lsub_iteration
        self.start_fence_region_density = False

    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        self.wirelength = self.op_collections.wirelength_op(pos)
        if len(self.placedb.regions) > 0:
            self.density = self.op_collections.fence_region_density_merged_op(pos)
        else:
            self.density = self.op_collections.density_op(pos)

        if self.init_density is None:
            ### record initial density
            self.init_density = self.density.data.clone()
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density > 0, 1 /self.init_density[self.init_density > 0])
            self.quad_penalty_coeff = self.density_quad_coeff / 2 * self.density_weight_grad_precond
        if self.quad_penalty:
            ### quadratic density penalty
            self.density = self.density * (1 + self.quad_penalty_coeff * self.density)
        if len(self.placedb.regions) > 0:
            result = self.wirelength + self.density_weight.dot(self.density)
        else:
            result = torch.add(self.wirelength, self.density, alpha=(self.density_factor * self.density_weight).item())

        if self.rass_state is not None:
            self.rass_state["last_penalty"] = self.compute_rass_penalty(pos)
            result = result + self.rass_state["last_penalty"]

        return result

    def obj_and_grad_fn_old(self, pos_w, pos_g=None, admm_multiplier=None):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        if not self.start_fence_region_density:
            obj = self.obj_fn(pos_w, pos_g, admm_multiplier)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            obj.backward()
        else:
            num_nodes = self.placedb.num_nodes
            num_movable_nodes = self.placedb.num_movable_nodes
            num_filler_nodes = self.placedb.num_filler_nodes


            wl = self.op_collections.wirelength_op(pos_w)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            wl.backward()
            wl_grad = pos_w.grad.data.clone()
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.init_density is None:
                self.init_density = self.op_collections.density_op(pos_w.data).data.item()

            if self.quad_penalty:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)
                inner_density = inner_density + self.density_quad_coeff / 2 / self.init_density  * inner_density**2
            else:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)

            inner_density.backward()
            inner_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map > 1e3
            inner_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            inner_density_grad[-num_filler_nodes:].mul_(0.5)
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.quad_penalty:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)
                outer_density = outer_density + self.density_quad_coeff / 2 / self.init_density  * outer_density ** 2
            else:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)

            outer_density.backward()
            outer_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map < 1e3
            outer_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            outer_density_grad[-num_filler_nodes:].mul_(0.5)

            if self.quad_penalty:
                density = self.op_collections.density_op(pos_w.data)
                obj = wl.data.item() + self.density_weight * (density + self.density_quad_coeff / 2 / self.init_density * density ** 2)
            else:
                obj = wl.data.item() + self.density_weight * self.op_collections.density_op(pos_w.data)

            pos_w.grad.data.copy_(wl_grad + self.density_weight * (inner_density_grad + outer_density_grad))


        self.op_collections.precondition_op(pos_w.grad, self.density_weight, 0)

        return obj, pos_w.grad

    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        if pos.grad is not None:
            pos.grad.zero_()
        obj = self.obj_fn(pos)

        if obj.requires_grad:
          obj.backward()

        self.op_collections.precondition_op(pos.grad, self.density_weight, self.update_mask, self.fix_nodes_mask)

        return obj, pos.grad

    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_collections.pos[0])

    def check_gradient(self, pos):
        """
        @brief check gradient for debug
        @param pos locations of cells
        """
        wirelength = self.op_collections.wirelength_op(pos)

        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        pos.grad.zero_()
        density = self.density_weight * self.op_collections.density_op(pos)
        density.backward()
        density_grad = pos.grad.clone()

        wirelength_grad_norm = wirelength_grad.norm(p=1)
        density_grad_norm = density_grad.norm(p=1)

        logging.info("wirelength_grad norm = %.6E" % (wirelength_grad_norm))
        logging.info("density_grad norm    = %.6E" % (density_grad_norm))
        pos.grad.zero_()

    def estimate_initial_learning_rate(self, x_k, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)

        return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

    def build_weighted_average_wl(self, params, placedb, data_collections,
                                  pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        # use WeightedAverageWirelength atomic
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_logsumexp_wl(self, params, placedb, data_collections,
                           pin_pos_op):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        wirelength_for_pin_op = logsumexp_wirelength.LogSumExpWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_density_overflow(self, params, placedb, data_collections,
                               num_bins_x, num_bins_y):
        """
        @brief compute density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return density_overflow.DensityOverflow(
            data_collections.node_size_x,
            data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0)

    def build_electric_overflow(self, params, placedb, data_collections,
                                num_bins_x, num_bins_y):
        """
        @brief compute electric density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return electric_overflow.ElectricOverflow(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask)

    def build_density_potential(self, params, placedb, data_collections,
                                num_bins_x, num_bins_y, padding, name):
        """
        @brief NTUPlace3 density potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        local_num_bins_x = num_bins_x + 2 * padding
        local_num_bins_y = num_bins_y + 2 * padding
        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x) + 4 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y) + 4 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, local_num_bins_x, local_num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, padding))
        if local_num_bins_x < max_num_bins:
            logging.warning("local_num_bins_x (%d) < max_num_bins (%d)" %
                            (local_num_bins_x, max_num_bins))
        if local_num_bins_y < max_num_bins:
            logging.warning("local_num_bins_y (%d) < max_num_bins (%d)" %
                            (local_num_bins_y, max_num_bins))

        node_size_x = placedb.node_size_x
        node_size_y = placedb.node_size_y

        # coefficients
        ax = (4 / (node_size_x + 2 * bin_size_x) /
              (node_size_x + 4 * bin_size_x)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        bx = (2 / bin_size_x / (node_size_x + 4 * bin_size_x)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])
        ay = (4 / (node_size_y + 2 * bin_size_y) /
              (node_size_y + 4 * bin_size_y)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        by = (2 / bin_size_y / (node_size_y + 4 * bin_size_y)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])

        # bell shape overlap function
        def npfx1(dist):
            # ax will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return 1.0 - ax.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfx2(dist):
            # bx will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return bx.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_x / 2 - 2 * bin_size_x).reshape(
                [placedb.num_nodes, 1])

        def npfy1(dist):
            # ay will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return 1.0 - ay.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfy2(dist):
            # by will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return by.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_y / 2 - 2 * bin_size_y).reshape(
                [placedb.num_nodes, 1])

        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_x = npfx1(0) + 2 * npfx1(bin_size_x) + 2 * npfx2(
            2 * bin_size_x)
        cx = (node_size_x.reshape([placedb.num_nodes, 1]) /
              integral_potential_x).reshape([placedb.num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_y = npfy1(0) + 2 * npfy1(bin_size_y) + 2 * npfy2(
            2 * bin_size_y)
        cy = (node_size_y.reshape([placedb.num_nodes, 1]) /
              integral_potential_y).reshape([placedb.num_nodes, 1])

        return density_potential.DensityPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            ax=torch.tensor(ax.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bx=torch.tensor(bx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cx=torch.tensor(cx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            ay=torch.tensor(ay.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            by=torch.tensor(by.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cy=torch.tensor(cy.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bin_center_x=data_collections.bin_center_x_padded(placedb, padding, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, padding, num_bins_y),
            target_density=data_collections.target_density,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            padding=padding,
            sigma=(1.0 / 16) * placedb.width / bin_size_x,
            delta=2.0)

    def build_electric_potential(self, params, placedb, data_collections,
                                 num_bins_x, num_bins_y, name, region_id=None, fence_regions=None):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param name string for printing
        @param fence_regions a [n_subregions, 4] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x[0:placedb.num_movable_nodes]) +
             2 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y[0:placedb.num_movable_nodes]) +
             2 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, num_bins_x, num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, 0))
        if num_bins_x < max_num_bins:
            logging.warning("num_bins_x (%d) < max_num_bins (%d)" %
                            (num_bins_x, max_num_bins))
        if num_bins_y < max_num_bins:
            logging.warning("num_bins_y (%d) < max_num_bins (%d)" %
                            (num_bins_y, max_num_bins))
        #### for fence region, the target density is different from different regions
        target_density = data_collections.target_density.item() if fence_regions is None else placedb.target_density_fence_region[region_id]
        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            fast_mode=params.RePlAce_skip_energy_flag,
            region_id=region_id,
            fence_regions=fence_regions,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb)

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        wirelength = self.op_collections.wirelength_op(
            self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        self.data_collections.pos[0].grad.zero_()

        if len(self.placedb.regions) > 0:
            density_list = []
            density_grad_list = []
            for density_op in self.op_collections.fence_region_density_ops:
                density_i = density_op(self.data_collections.pos[0])
                density_list.append(density_i.data.clone())
                density_i.backward()
                density_grad_list.append(self.data_collections.pos[0].grad.data.clone())
                self.data_collections.pos[0].grad.zero_()

            ### record initial density
            self.init_density = torch.stack(density_list)
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density > 0, 1/self.init_density[self.init_density > 0])
            ### compute u
            self.density_weight_u = self.init_density * self.density_weight_grad_precond
            self.density_weight_u += 0.5 * self.density_quad_coeff * self.density_weight_u ** 2
            ### compute s
            density_weight_s = 1 + self.density_quad_coeff * self.init_density * self.density_weight_grad_precond
            ### compute density grad L1 norm
            density_grad_norm = sum(self.density_weight_u[i] * density_weight_s[i] * density_grad_list[i].norm(p=1) for i in range(density_weight_s.size(0)))

            self.density_weight_u *= params.density_weight * wirelength_grad_norm / density_grad_norm
            ### set initial step size for density weight update
            self.density_weight_step_size_inc_low = 1.03
            self.density_weight_step_size_inc_high = 1.04
            self.density_weight_step_size = (self.density_weight_step_size_inc_low - 1) * self.density_weight_u.norm(p=2)
            ### commit initial density weight
            self.density_weight = self.density_weight_u * density_weight_s

        else:
            density = self.op_collections.density_op(self.data_collections.pos[0])
            ### record initial density
            self.init_density = density.data.clone()
            density.backward()
            density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

            grad_norm_ratio = wirelength_grad_norm / density_grad_norm
            self.density_weight = torch.tensor(
                [params.density_weight * grad_norm_ratio],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)

        return self.density_weight

    def build_update_density_weight(self, params, placedb, algo="overflow"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        ### params for overflow mode from elfPlace
        assert algo in {"hpwl", "overflow"}, logging.error("density weight update not supports hpwl mode or overflow mode")

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(
                        np.power(0.9999, float(iteration)), 0.98)
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(
                            min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            assert self.quad_penalty == True, logging.error("density weight update based on overflow only works for quadratic density penalty")
            ### based on overflow
            ### stop updating if a region has lower overflow than stop overflow
            with torch.no_grad():
                density_norm = cur_metric.density * self.density_weight_grad_precond
                density_weight_grad = density_norm + self.density_quad_coeff / 2 * density_norm ** 2
                density_weight_grad /= density_weight_grad.norm(p=2)

                self.density_weight_u += self.density_weight_step_size * density_weight_grad
                density_weight_s = 1 + self.density_quad_coeff * density_norm

                density_weight_new = (self.density_weight_u * density_weight_s).clamp(max=10)

                ### conditional update if this region's overflow is higher than stop overflow
                if(self.update_mask is None):
                    self.update_mask = cur_metric.overflow >= self.params.stop_overflow
                else:
                    ### restart updating is not allowed
                    self.update_mask &= cur_metric.overflow >= self.params.stop_overflow
                self.density_weight.masked_scatter_(self.update_mask, density_weight_new[self.update_mask])

                ### update density weight step size
                rate = torch.log(self.density_quad_coeff * density_norm.norm(p=2)).clamp(min=0)
                rate = rate / (1 + rate)
                rate = rate * (self.density_weight_step_size_inc_high - self.density_weight_step_size_inc_low) + self.density_weight_step_size_inc_low
                self.density_weight_step_size *= rate

        if not self.quad_penalty and algo == "overflow":
            logging.warning("quadratic density penalty is disabled, density weight update is forced to be based on HPWL")
            algo = "hpwl"
        if len(self.placedb.regions) == 0 and algo == "overflow":
            logging.warning("for benchmark without fence region, density weight update is forced to be based on HPWL")
            algo = "hpwl"

        update_density_weight_op = {"hpwl":update_density_weight_op_hpwl,
                                    "overflow": update_density_weight_op_overflow}[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        return params.gamma * (self.bin_size_x + self.bin_size_y)

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        ### overflow can have multiple values for fence regions, use their weighted average based on movable node number
        if overflow.numel() == 1:
            overflow_avg = overflow
        else:
            overflow_avg = overflow
        coef = torch.pow(10, (overflow_avg - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_((base_gamma * coef).item())
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y],
            dim=0).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                if self.fix_nodes_mask is not None:
                    noise = noise.view(2, -1)
                    noise[0, :placedb.num_movable_nodes].masked_fill_(self.fix_nodes_mask[:placedb.num_movable_nodes], 0)
                    noise[1, :placedb.num_movable_nodes].masked_fill_(self.fix_nodes_mask[:placedb.num_movable_nodes], 0)
                    noise = noise.view(-1)
                noise[placedb.num_movable_nodes:placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                noise[placedb.num_nodes +
                      placedb.num_movable_nodes:2 * placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb,
                           data_collections, op_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param op_collections a collection of all ops
        """
        return PreconditionOp(placedb, data_collections, op_collections)

    def build_route_utilization_map(self, params, placedb, data_collections):
        """
        @brief routing congestion map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        congestion_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            initial_horizontal_utilization_map=data_collections.
            initial_horizontal_utilization_map,
            initial_vertical_utilization_map=data_collections.
            initial_vertical_utilization_map,
            deterministic_flag=params.deterministic_flag)

        def route_utilization_map_op(pos):
            pin_pos = self.op_collections.pin_pos_op(pos)
            return congestion_op(pin_pos)

        return route_utilization_map_op

    def build_pin_utilization_map(self, params, placedb, data_collections):
        """
        @brief pin density map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        return pin_utilization.PinUtilization(
            pin_weights=data_collections.pin_weights,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            pin_stretch_ratio=params.pin_stretch_ratio,
            deterministic_flag=params.deterministic_flag)

#扫描固定宏、宏 pin 等，以面积重叠和 pin 数累加成基础风险图。可选叠加 RUDY 拥塞图、pin 利用率图，并做 3×3／5×5 高斯平滑。归一化风险图，缓存网格尺寸、阈值、初始风险权重形成rass_state
def build_rass_state(self, params, placedb, data_collections):
    # 构建 RASS 所需的“风险状态”（主要是风险热图）

    device = data_collections.pos[0].device
    # 从位置张量里拿到当前的设备（CPU/GPU），后续把张量都放到同一设备上

    dtype = data_collections.pos[0].dtype
    # 取出位置张量的数据类型（如 torch.float32），保证后续计算类型一致

    num_bins_x = self.num_bins_x
    num_bins_y = self.num_bins_y
    # 风险网格（heatmap）的 x/y 方向划分的 bin 数

    bin_size_x = self.bin_size_x
    bin_size_y = self.bin_size_y
    # 每个 bin 在物理坐标中的宽和高

    bin_area = max(bin_size_x * bin_size_y, 1e-12)
    # 每个 bin 的面积（带下限，防止后面做除法出现除 0）

    risk_np = np.zeros((num_bins_x, num_bins_y), dtype=np.float64)
    # 用 numpy 初始化一个全 0 的风险二维数组（双精度，之后再转 torch）

    pin_counts = data_collections.pin_weights.detach().cpu().numpy()
    # 取出每个节点/单元的“引脚权重”（引脚数或其权重），拷到 CPU，转为 numpy

    node_x = placedb.node_x
    node_y = placedb.node_y
    node_size_x = placedb.node_size_x
    node_size_y = placedb.node_size_y
    # 从 place 数据库中获取每个节点（单元/宏/IO）的几何信息：左下角坐标和宽高

    fixed_start = placedb.num_movable_nodes
    fixed_end = placedb.num_movable_nodes + placedb.num_terminals
    # 计算固定对象（不可移动的宏/IO）在数组中的索引范围：[fixed_start, fixed_end)

    for idx in range(fixed_start, fixed_end):
        # 遍历所有固定对象：把它们的几何影响投影到风险网格上

        width = node_size_x[idx]
        height = node_size_y[idx]
        if width <= 0 or height <= 0:
            continue
        # 跳过宽高非法的对象

        x0 = node_x[idx]
        y0 = node_y[idx]
        x1 = x0 + width
        y1 = y0 + height
        # 固定对象的包围盒坐标（物理坐标系）

        bx0 = max(int(np.floor((x0 - placedb.xl) / bin_size_x)), 0)
        bx1 = min(int(np.ceil((x1 - placedb.xl) / bin_size_x)), num_bins_x)
        by0 = max(int(np.floor((y0 - placedb.yl) / bin_size_y)), 0)
        by1 = min(int(np.ceil((y1 - placedb.yl) / bin_size_y)), num_bins_y)
        # 把物理坐标投射到离散网格：求出与该对象重叠的 bin 的 x/y 索引范围
        # placedb.xl/yl 是芯片左下角；用 floor/ceil 找到覆盖的离散区间，并裁剪在合法范围内

        if bx0 >= bx1 or by0 >= by1:
            continue
        # 没有有效重叠就跳过

        weight = 1.0 + float(pin_counts[idx]) if idx < pin_counts.shape[0] else 1.0
        # 为该固定对象设置权重：1 + 引脚权重（如果 pin_counts 有这个 idx）
        # 引脚越多/权重越大，对应风险贡献越大；没有则退化为 1.0。实现式中的 (1+𝑐_𝑓)

        for bx in range(bx0, bx1):
            # 遍历与该对象重叠的所有 x 向 bin

            bin_x0 = placedb.xl + bx * bin_size_x
            bin_x1 = bin_x0 + bin_size_x
            # 该 x-bin 的左右边界（物理坐标）

            overlap_x = max(0.0, min(x1, bin_x1) - max(x0, bin_x0))
            # 计算对象与该 x-bin 的重叠长度（x 方向）

            if overlap_x <= 0:
                continue
            # 如果 x 方向没有重叠，跳过对应 y 循环

            for by in range(by0, by1):
                # 遍历与该对象重叠的所有 y 向 bin

                bin_y0 = placedb.yl + by * bin_size_y
                bin_y1 = bin_y0 + bin_size_y
                # 该 y-bin 的下/上边界

                overlap_y = max(0.0, min(y1, bin_y1) - max(y0, bin_y0))
                # 计算对象与该 y-bin 的重叠长度（y 方向）

                if overlap_y <= 0:
                    continue
                # y 方向没重叠就跳过

                risk_np[bx, by] += weight * (overlap_x * overlap_y) / bin_area
                # 将重叠面积占比（相对于一个 bin 的面积）乘权重，累加到该 bin 的风险值
                #固定块覆盖的面积越多、引脚越多，对该 bin 的“风险”贡献越大

    risk_tensor = torch.from_numpy(risk_np).to(device=device, dtype=dtype)
    # 把 numpy 风险图转成 torch 张量，并放到目标设备/类型上.这样就能用 PyTorch 的算子在 CPU/GPU 上继续计算

    risk_tensor = self._apply_gaussian_blur(risk_tensor, kernel_size=3)
    # 先做一次 3x3 高斯模糊（细粒度平滑），抑制离散化的锯齿/噪声

    with torch.no_grad():
        # 下面这段不需要参与反向传播，仅作为辅助场叠加

        if params.rass_route_weight > 0 and getattr(self.op_collections, "route_utilization_map_op", None) is not None:
            # 如果开启了 route 风险权重，且提供了路由利用率算子

            route_map = self.op_collections.route_utilization_map_op(data_collections.pos[0]).to(device=device, dtype=dtype)
            # 根据当前单元坐标 pos 计算路由利用率热图（RUDY 风格），并对齐设备/类型

            route_map = self._resize_risk_map(route_map, (num_bins_x, num_bins_y))
            # 将路由热图 resize 到与风险网格相同的分辨率（bin 尺寸）

            max_route = torch.max(route_map)
            if max_route > 0:
                route_map = route_map / max_route
            # 做一次最大值归一化，把 route_map 标准化到 [0,1]

            risk_tensor = risk_tensor + params.rass_route_weight * route_map
            # 按权重把路由热点叠加到风险图中

        if params.rass_pin_weight > 0 and getattr(self.op_collections, "pin_utilization_map_op", None) is not None:
            # 若开启了引脚利用率风险权重，且提供了引脚利用率算子

            pin_map = self.op_collections.pin_utilization_map_op(data_collections.pos[0]).to(device=device, dtype=dtype)
            # 基于当前 pos 计算引脚利用率热图

            pin_map = self._resize_risk_map(pin_map, (num_bins_x, num_bins_y))
            # 同样 resize 到风险网格大小

            max_pin = torch.max(pin_map)
            if max_pin > 0:
                pin_map = pin_map / max_pin
            # 最大值归一化到 [0,1]

            risk_tensor = risk_tensor + params.rass_pin_weight * pin_map
            # 将引脚热点按权重叠加进风险图

    if params.rass_multiscale_weight > 0:
        # 如果开启了多尺度融合

        coarse = self._apply_gaussian_blur(risk_tensor, kernel_size=5)
        # 再做一次更“粗”的 5x5 高斯平滑，得到长尺度的上下文（通道/走廊等）

        risk_tensor = (1.0 - params.rass_multiscale_weight) * risk_tensor + params.rass_multiscale_weight * coarse
        # 用 α 做 fine/coarse 融合：risk = (1-α)*细 + α*粗
        # 这样既保留局部热点，又注入大范围的风险趋势


        risk_tensor = risk_tensor.clamp(min=0)
        # 把风险图中所有负值截断为 0，确保风险非负
        max_val = torch.max(risk_tensor)
        # 计算整张风险图的最大值，准备用于归一化
        if max_val > 0:
            risk_tensor = risk_tensor / max_val
            # 若最大值大于 0，则将风险图按最大值归一化到 [0,1]；全 0 则跳过
        state = {
            "risk_map": risk_tensor,# 归一化后的风险热图（二维张量）
            "weight": torch.tensor(params.rass_risk_weight, dtype=dtype, device=device),
            # 风险惩罚项的权重 λ_r；用与计算一致的 dtype/device 存为张量，便于后续参与计算
            "threshold": torch.tensor(params.rass_risk_threshold, dtype=dtype, device=device),
            # 风险阈值 τ（如 0.7），用于计算 [R(x,y)-τ]_+ 之类的铰链惩罚
            "bin_size_x": torch.tensor(bin_size_x, dtype=dtype, device=device),
            "bin_size_y": torch.tensor(bin_size_y, dtype=dtype, device=device),
             # 风险网格每个 bin 的物理尺寸（宽/高），用于坐标↔网格映射、信任域步长等
            "xl": torch.tensor(placedb.xl, dtype=dtype, device=device),
            "yl": torch.tensor(placedb.yl, dtype=dtype, device=device),
             # 版图左下角坐标，用于把物理坐标定位到风险网格

            "num_movable": placedb.num_movable_nodes,
            # 可移动节点数量（整型），方便后续切片/统计
            "area_weights": data_collections.node_areas[:placedb.num_movable_nodes],
            # 可移动节点的面积权重 a_i；风险惩罚通常按面积加权

            "num_bins_x": num_bins_x,
            "num_bins_y": num_bins_y,
            # 风险网格的离散尺寸（x/y 方向的 bin 数）
            "eps": torch.tensor(1e-6, dtype=dtype, device=device),
             # 数值稳定用的小常数（如防除 0）
            "layout_diag": math.hypot(placedb.xh - placedb.xl, placedb.yh - placedb.yl),
            # 版图对角线长度 √(W^2+H^2)，用于把位移等量纲化为“相对对角线比例”
        }
        state["base_weight"] = state["weight"].clone()
        # 备份一份初始风险权重，供动态调度时回退或按相对比例调整

        state["last_refresh_iter"] = -1
        # 上一次风险图刷新的迭代号；-1 表示尚未刷新过（配合周期刷新策略）
        return state
    # 返回打包好的 RASS 状态
#*********************************************************************************
    #准备与GiFt初始化挂钩，把风险图、阈值、守卫、采样数等封装给 GiFt
    def _compose_rass_options(self, params, placedb):
        if self.rass_state is None:
            return None
        # 若尚未构建 RASS 状态，则不启用，直接返回 None

        return {
            "enabled": True,# 明确开启 RASS 功能
            "risk_map": self.rass_state["risk_map"],
            # 直接传递风险热图张量给下游（用于插值/评估）

            "bin_size_x": float(self.rass_state["bin_size_x"].item()),
            "bin_size_y": float(self.rass_state["bin_size_y"].item()),
            "xl": float(self.rass_state["xl"].item()),
            "yl": float(self.rass_state["yl"].item()),
            # 单元素张量转为 Python float，便于非张量逻辑或日志打印
            "num_bins_x": self.rass_state["num_bins_x"],
            "num_bins_y": self.rass_state["num_bins_y"],
            # 风险网格的离散尺寸（整型）
            "threshold": float(self.rass_state["threshold"].item()),
             # 风险阈值 τ（float 形式供下游使用）
            "hpwl_guard": params.rass_hpwl_guard,
            "disp_avg_guard": params.rass_disp_guard_avg,
            "disp_max_guard": params.rass_disp_guard_max,
             # 自适应候选评估的三道护栏：HPWL 上限、平均位移上限、最大位移上限。保证 RASS 初始化不会引入过大的线长或位移异常
            "num_samples": params.rass_num_samples,
             # Dirichlet 采样的候选数量（探索不同基底权重组合）
            "adapt_flag": bool(params.rass_adapt_flag),
            # 是否启用自适应混合（False 则走固定权重）
            "layout_diag": self.rass_state["layout_diag"],
            # 版图对角线长度（给位移护栏/步长比例使用）
            "risk_weight": float(self.rass_state["weight"].item()),
            # 当前风险权重（可能已被动态调度修改），float 形式给下游
            "num_movable": placedb.num_movable_nodes,
             # 可移动节点数量（下游可能按规模调整采样/评估）
        }

    def _resize_risk_map(self, tensor, target_shape):
        if tensor.shape == target_shape:
            return tensor
        # 若已是目标大小，直接返回，避免不必要的插值
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
        # 用双线性插值缩放到 target_shape=(H, W)；align_corners=False 避免边缘像素对齐偏差
        return resized.squeeze(0).squeeze(0)
        # 再降回二维张量 [H, W] 返回；dtype/device 与输入保持一致

    def _apply_gaussian_blur(self, tensor, kernel_size=3):
        if kernel_size <= 1:
            return tensor
        if kernel_size == 3:
            kernel = torch.tensor([[1.0, 2.0, 1.0],
                                   [2.0, 4.0, 2.0],
                                   [1.0, 2.0, 1.0]], dtype=tensor.dtype, device=tensor.device)
            kernel = kernel / kernel.sum()
            pad = 1
        elif kernel_size == 5:
            kernel = torch.tensor(
                [[1., 4., 6., 4., 1.],
                 [4., 16., 24., 16., 4.],
                 [6., 24., 36., 24., 6.],
                 [4., 16., 24., 16., 4.],
                 [1., 4., 6., 4., 1.]],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            kernel = kernel / kernel.sum()
            pad = 2
        else:
            kernel = torch.ones((kernel_size, kernel_size), dtype=tensor.dtype, device=tensor.device) / (kernel_size * kernel_size)
            pad = kernel_size // 2
        tensor4d = tensor.unsqueeze(0).unsqueeze(0)
        smoothed = F.conv2d(F.pad(tensor4d, (pad, pad, pad, pad), mode="replicate"),
                            kernel.unsqueeze(0).unsqueeze(0))
        return smoothed.squeeze(0).squeeze(0)
#在主目标中额外计算 λ_r Σ [r_i-τ]_+ · a_i：对可移动单元做双线性插值，风险超过阈值才计罚，并乘以单元面积.罚项加权后与原始线长/密度目标相加，使 Nesterov 主迭代也能感知风险
    def compute_rass_penalty(self, pos):
        state = self.rass_state
        if state is None or state["weight"].item() <= 0:# 若还没构建 RASS 状态，或风险权重 λ_r ≤ 0，则惩罚为 0（返回标量 0）
            return torch.zeros((), dtype=pos.dtype, device=pos.device)
        num_movable = state["num_movable"]
        if num_movable == 0:
            return torch.zeros((), dtype=pos.dtype, device=pos.device) # 没有可移动单元，则无需计算，直接返回 0
        pos_view = pos.view(2, -1)# 将 pos 视作 [2, N]（第 0 行是 x， 第 1 行是 y），不拷贝数据
        node_size_x = self.data_collections.node_size_x[:num_movable]
        node_size_y = self.data_collections.node_size_y[:num_movable]# 取可移动单元的宽高（长度 = num_movable）
        center_x = pos_view[0, :num_movable] + 0.5 * node_size_x# 用左下角坐标 + 半宽/半高，得到每个可移动单元的中心坐标 (x,y)
        center_y = pos_view[1, :num_movable] + 0.5 * node_size_y # 后面就在中心点上对风险图做采样
        bin_size_x = state["bin_size_x"]
        bin_size_y = state["bin_size_y"]
        xl = state["xl"]
        yl = state["yl"]
        eps = state["eps"]
        num_bins_x = state["num_bins_x"]
        num_bins_y = state["num_bins_y"]# 从 state 取出网格尺寸、版图左下角、数值稳定用 eps、以及网格离散大小
        
        fx = (center_x - xl) / bin_size_x# 把中心点从物理坐标映射到“bin 坐标系”（单位是 bin）
        fy = (center_y - yl) / bin_size_y# fx, fy 是实数，表示落在第几号 bin 及其小数偏移
        max_x = torch.tensor(num_bins_x - 1, dtype=fx.dtype, device=fx.device) - eps
        max_y = torch.tensor(num_bins_y - 1, dtype=fy.dtype, device=fy.device) - eps
        fx = torch.clamp(fx, min=0.0)
        fy = torch.clamp(fy, min=0.0)# 将 fx/fy 裁剪到 [0, num_bins-1-eps]
        fx = torch.minimum(fx, max_x)# 这样下一步 floor 后的 ix0 ∈ [0, num_bins-2]，ix1=ix0+1 最多到 num_bins-1，避免越界
        fy = torch.minimum(fy, max_y)

        ix0 = torch.floor(fx)
        iy0 = torch.floor(fy)
        tx = fx - ix0
        ty = fy - iy0
        ix0 = ix0.long()
        iy0 = iy0.long()
        ix1 = torch.clamp(ix0 + 1, max=num_bins_x - 1)# 计算双线性插值所需的四邻域索引与权重
        iy1 = torch.clamp(iy0 + 1, max=num_bins_y - 1)# (ix0,iy0) 是左下格；(ix1,iy0) 右下；(ix0,iy1) 左上；(ix1,iy1) 右上。tx,ty ∈ [0,1) 是在该格内的局部小数位置

        risk_map = state["risk_map"]
        r00 = risk_map[ix0, iy0]
        r10 = risk_map[ix1, iy0]
        r01 = risk_map[ix0, iy1]
        r11 = risk_map[ix1, iy1]# 从风险图取四个角点的风险值

        risk_val = (1.0 - tx) * (1.0 - ty) * r00 \
            + tx * (1.0 - ty) * r10 \
            + (1.0 - tx) * ty * r01 \
            + tx * ty * r11# 对中心点位置做“双线性插值”，得到该点的风险值 R(x,y)

        penalty = F.relu(risk_val - state["threshold"]) # 铰链惩罚 [R - τ]_+：低于阈值 τ（如 0.7）不惩罚，高于阈值的部分线性增长
        penalty = penalty * state["area_weights"]# 按单元面积 a_i 加权（大单元更影响拥塞/工艺风险，因此惩罚更大）。area_weights 长度 = num_movable，与 penalty 一一对应

        return state["weight"] * penalty.sum() # 最终把所有可移动单元的惩罚求和，并乘以风险项全局权重 λ_r，得到一个标量损失
    #refresh_rass_state 周期性重建风险图并调用 gift_init_op.update_rass
    def refresh_rass_state(self, params, placedb, iteration, force=False):
        if (
            self.rass_state is None
            or params.rass_refresh_interval <= 0
            or not params.rass_place_flag
        ):
            return False
        last_iter = self.rass_state.get("last_refresh_iter", -1)
        if not force and iteration is not None and last_iter >= 0:# 若不是强制刷新，且有当前迭代号且曾刷新过，若距离上次刷新不到设定的刷新间隔（rass_refresh_interval），就暂不刷新
            if iteration - last_iter < params.rass_refresh_interval:
                return False
        with torch.no_grad():
            preserved_weight = self.rass_state["weight"].clone()
            preserved_base = self.rass_state.get("base_weight", preserved_weight.clone())# 先把当前风险权重 λ_r（weight）和基线权重（base_weight）备份，稍后回填
            new_state = self.build_rass_state(params, placedb, self.data_collections)# 重新构建 RASS 状态：重算风险图（可能基于最新坐标/利用率），并返回新 state
            new_state["weight"] = preserved_weight
            new_state["base_weight"] = preserved_base# 用备份回填风险权重与基线，避免刷新把动态调度过的权重重置掉
            new_state["last_refresh_iter"] = iteration if iteration is not None else 0# 记录这次刷新的迭代号；若未知则记 0
            self.rass_state = new_state# 用新 state 覆盖旧的 RASS 状态
            self.data_collections.rass_risk_map = self.rass_state["risk_map"]# 把新风险图同步到 data_collections，供其它模块直接使用
            rass_options = self._compose_rass_options(params, placedb)
            if rass_options and getattr(self.op_collections, "gift_init_op", None):
                self.op_collections.gift_init_op.update_rass(rass_options) # 重新打包 RASS 选项并通知下游（如 GiFt 初始化算子）更新内部配置
        return True
    #schedule_rass_weights 使用 rass_feedback_* 参数调节风险权重
    def schedule_rass_weights(self, params, iteration, route_metrics=None, pin_metrics=None):# 按“拥塞压力”动态调节风险权重 λ_r（rass_state["weight"]）,route_metrics/pin_metrics 为可选的拥塞与引脚指标字典
        if (
            self.rass_state is None
            or not params.rass_place_flag
            or not params.rass_feedback_flag
        ):
            return False # 若未构建 RASS 状态、未开启 RASS、或未开启反馈调度，则不做任何事
        overflow_high = getattr(params, "rass_feedback_overflow_high", 0.2)# - overflow_high/low：上/下阈形成“滞回区”，避免权重在边界来回抖动
        overflow_low = getattr(params, "rass_feedback_overflow_low", 0.05)
        step_up = getattr(params, "rass_feedback_weight_step_up", 0.2)# - step_up/down：每次上调/下调的步幅（相对于 base_weight 的比例）
        step_down = getattr(params, "rass_feedback_weight_step_down", 0.1)
        clip_ratio = max(getattr(params, "rass_feedback_weight_clip", 3.0), 1.0)# - clip_ratio：相对基线的裁剪倍数（权重 ∈ [base/clip, base*clip]）
        # 上述调度参数（带默认值）
        pressure_terms = []
        if route_metrics:# 路由压力项
            pressure_terms.append(max(route_metrics.get("avg_overflow", 0.0), 0.0))# - avg_overflow：平均溢出（越大越拥塞）
            pressure_terms.append(max(route_metrics.get("max_util", 0.0) - 1.0, 0.0))# - max_util-1：最大利用率超过 100% 的超额部分（例：1.18 -> 0.18）
        if pin_metrics:# 引脚压力项
            pressure_terms.append(max(pin_metrics.get("avg_overflow", 0.0), 0.0))
            pressure_terms.append(max(pin_metrics.get("max_util", 0.0) - 1.0, 0.0))# - avg_overflow、max_util 同上
            pressure_terms.append(max(pin_metrics.get("high_risk_ratio", 0.0), 0.0))# - high_risk_ratio：高风险 bin 的占比（越大表示风险更普遍）
        pressure = max(pressure_terms) if pressure_terms else 0.0
         # 取所有压力项中的最大值作为“总压力”指标
        base_weight = self.rass_state.get("base_weight", self.rass_state["weight"])
        base_val = float(base_weight.item())
        if base_val <= 0:
            return False# 基线权重（初始 λ_r），基线 <= 0 时不调度
        current_val = float(self.rass_state["weight"].item())
        max_weight = base_val * clip_ratio
        min_weight = base_val / clip_ratio if clip_ratio > 1.0 else base_val * 0.5# 计算权重允许的上下限： [base/clip, base*clip]，若 clip_ratio == 1 则允许对称 0.5×base 的下界（保守兜底）
        updated_val = current_val
        if pressure > overflow_high:
            updated_val = min(max_weight, current_val + base_val * step_up)# 压力高于上阈：上调权重（以 base 为步长单位），并裁剪到 max_weight
        elif pressure < overflow_low:
            updated_val = max(min_weight, current_val - base_val * step_down)# 压力低于下阈：下调权重，并裁剪到 min_weight
        # 若处于滞回区 [overflow_low, overflow_high]：不变
        if abs(updated_val - current_val) < 1e-9:
            return False
        # 若变化极小（≈未变），直接返回 False
        self.rass_state["weight"].data.fill_(updated_val)# 若变化极小（≈未变），直接返回 False
        rass_options = self._compose_rass_options(params, placedb=self.placedb)
        if rass_options and getattr(self.op_collections, "gift_init_op", None):
            self.op_collections.gift_init_op.update_rass(rass_options) # 将更新后的选项推送给初始化算子（如 GiFt 初始化 op），使运行中生效
        return True # 返回 True 表示本次确实更新了风险权重
    #repair_rass_hotspots 在高风险 bin 内执行局部修复。
    def repair_rass_hotspots(self, params, placedb, pos_tensor, route_metrics=None, pin_metrics=None):    # 基于风险图，对最高风险的若干 bin 做一次轻量的局部“推离”修复（把在该 bin 内的可移动单元往 bin 外推）
        if (
            self.rass_state is None
            or not params.rass_place_flag
            or not params.rass_feedback_flag
        ):
            return False# 若未构建 RASS 状态、未开启 RASS、或未开启反馈功能，直接不做修复
        topk = int(max(getattr(params, "rass_hotspot_topk", 0), 0))
        if topk <= 0:
            return False # 取要修复的“热点 bin 个数” topk（<=0 则不修）
        threshold = getattr(params, "rass_hotspot_threshold", 0.9)# 风险阈值：只有风险值超过该阈的 bin 才视为需要修复的热点
        risk_map = self.rass_state["risk_map"]
        if risk_map is None or risk_map.numel() == 0:
            return False # 没有风险图就不修
        flat = risk_map.view(-1)# 把 2D 风险图拉平成 1D，便于做 topk
        k = min(topk, flat.numel())
        values, indices = torch.topk(flat, k)# 取风险值最高的 k 个元素：values 是风险值，indices 是对应的扁平索引
        mask = values > threshold
        if mask.sum() == 0:
            return False
        # 只保留超过阈值的那些 bin；如果都不超过，就不需要修复
        num_bins_y = self.rass_state["num_bins_y"]
        bin_size_x = float(self.rass_state["bin_size_x"].item())
        bin_size_y = float(self.rass_state["bin_size_y"].item())
        xl = float(self.rass_state["xl"].item())
        yl = float(self.rass_state["yl"].item())
        # 取出网格尺寸、每个 bin 的物理大小，以及版图左下角坐标。risk_map 的形状约定是 (num_bins_x, num_bins_y)，即先 x 后 y
        pos_view = pos_tensor.view(2, -1)# 位置张量视作 [2, N]（第 0 行是 x， 第 1 行是 y），不拷贝数据
        node_size_x = self.data_collections.node_size_x[: self.rass_state["num_movable"]]
        node_size_y = self.data_collections.node_size_y[: self.rass_state["num_movable"]]
        centers_x = pos_view[0, : self.rass_state["num_movable"]] + 0.5 * node_size_x
        centers_y = pos_view[1, : self.rass_state["num_movable"]] + 0.5 * node_size_y
         # 仅取“可移动单元”的宽高（长度 = num_movable）。# 计算可移动单元的中心坐标（x/y）
        updated = False
        with torch.no_grad():
            for flat_idx, value in zip(indices[mask], values[mask]): # 遍历所有超过阈值的热点 bin（按风险大小排序后的子集）
                bx = int(flat_idx.item() // num_bins_y)# 从扁平索引还原 2D 索引 (bx, by)
                by = int(flat_idx.item() % num_bins_y) # 这里用 num_bins_y 做除/模，与 risk_map 的 (x,y) 维度约定一致
                bin_xl = xl + bx * bin_size_x
                bin_xh = bin_xl + bin_size_x
                bin_yl = yl + by * bin_size_y
                bin_yh = bin_yl + bin_size_y# 计算该 bin 的物理边界 [bin_xl, bin_xh) × [bin_yl, bin_yh)
                in_bin = (
                    (centers_x >= bin_xl)
                    & (centers_x < bin_xh)
                    & (centers_y >= bin_yl)
                    & (centers_y < bin_yh)
                )# 找出“中心点落在该 bin 内”的可移动单元（布尔掩码）
                if in_bin.sum() == 0:
                    continue # 该热点 bin 内没有可移动单元，跳过
                cx = (bin_xl + bin_xh) * 0.5
                cy = (bin_yl + bin_yh) * 0.5# bin 的中心坐标
                move_x = centers_x[in_bin] - cx
                move_y = centers_y[in_bin] - cy# 每个单元中心到 bin 中心的向量（希望把单元往“远离”中心的方向推）
                move_x = torch.sign(move_x).masked_fill(move_x == 0, 1.0)
                move_y = torch.sign(move_y).masked_fill(move_y == 0, 1.0) # 只取方向：sign >0 表示向 +x 推，<0 表示向 -x 推；等于 0 的设为 +1，避免不动
                step_scale = 0.1 + 0.15 * float(value.item()) # 推动步长的比例系数：基础 0.1，再随热点风险值线性增大（风险越高推得稍远）。value ∈ [0,1]（前面风险图做过归一化），所以 step_scale 大致在 [0.1, 0.25]
                delta_x = move_x * bin_size_x * step_scale
                delta_y = move_y * bin_size_y * step_scale# 实际位移量：以 bin 尺寸为尺度，按方向和强度推进
                indices_in_bin = in_bin.nonzero(as_tuple=False).view(-1)# 将布尔掩码转为索引列表（这些索引都是“可移动单元”的局部索引 0..num_movable-1）
                new_x = pos_view[0, indices_in_bin] + delta_x
                new_y = pos_view[1, indices_in_bin] + delta_y # 应用位移，得到新的左下角坐标（注意 pos_view 存的是单元左下角）
                min_x = torch.full_like(new_x, float(placedb.xl))
                min_y = torch.full_like(new_y, float(placedb.yl))# 版图左下角，用于下界裁剪
                size_x = node_size_x[indices_in_bin].to(device=new_x.device, dtype=new_x.dtype)
                size_y = node_size_y[indices_in_bin].to(device=new_y.device, dtype=new_y.dtype)# 取对应单元的宽高，并对齐 dtype/device
                max_x = torch.full_like(new_x, float(placedb.xh)) - size_x
                max_y = torch.full_like(new_y, float(placedb.yh)) - size_y# 上界裁剪时要保证“单元右/上边界不越界”，所以用 (xh - 宽, yh - 高)
                new_x = torch.max(torch.min(new_x, max_x), min_x)
                new_y = torch.max(torch.min(new_y, max_y), min_y)# 将新坐标裁剪到合法范围内 [xl, xh-宽] / [yl, yh-高]
                pos_view[0, indices_in_bin] = new_x
                pos_view[1, indices_in_bin] = new_y# 回写位置（只影响可移动部分的这些索引）
                updated = True
        return updated # 返回是否进行了任何更新（True=至少推了一次；False=无事可做）

    def build_nctugr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call NCTUgr for congestion estimation
        """
        path = "%s/%s" % (params.result_dir, params.design_name())
        return nctugr_binary.NCTUgr(
            aux_input_file=os.path.realpath(params.aux_input),
            param_setting_file="%s/../thirdparty/NCTUgr.ICCAD2012/DAC12.set" %
            (os.path.dirname(os.path.realpath(__file__))),
            tmp_pl_file="%s/%s.NCTUgr.pl" %
            (os.path.realpath(path), params.design_name()),
            tmp_output_file="%s/%s.NCTUgr" %
            (os.path.realpath(path), params.design_name()),
            horizontal_routing_capacities=torch.from_numpy(
                placedb.unit_horizontal_capacities *
                placedb.routing_grid_size_y),
            vertical_routing_capacities=torch.from_numpy(
                placedb.unit_vertical_capacities *
                placedb.routing_grid_size_x),
            params=params,
            placedb=placedb)

    def build_adjust_node_area(self, params, placedb, data_collections):
        """
        @brief adjust cell area according to routing congestion and pin utilization map
        """
        total_movable_area = (
            data_collections.node_size_x[:placedb.num_movable_nodes] *
            data_collections.node_size_y[:placedb.num_movable_nodes]).sum()
        total_filler_area = (
            data_collections.node_size_x[-placedb.num_filler_nodes:] *
            data_collections.node_size_y[-placedb.num_filler_nodes:]).sum()
        total_place_area = (total_movable_area + total_filler_area
                            ) / data_collections.target_density
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin_weights=data_collections.pin_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            route_num_bins_x=placedb.num_routing_grids_x,
            route_num_bins_y=placedb.num_routing_grids_y,
            pin_num_bins_x=placedb.num_routing_grids_x,
            pin_num_bins_y=placedb.num_routing_grids_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_place_area - total_movable_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity)

        def build_adjust_node_area_op(pos, route_utilization_map,
                                      pin_utilization_map):
            return adjust_node_area_op(
                pos, data_collections.node_size_x,
                data_collections.node_size_y, data_collections.pin_offset_x,
                data_collections.pin_offset_y, data_collections.target_density,
                route_utilization_map, pin_utilization_map)

        return build_adjust_node_area_op

    def build_fence_region_density_op(self, fence_region_list, node2fence_region_map):
        assert type(fence_region_list) == list and len(fence_region_list) == 2, "Unsupported fence region list"
        self.data_collections.node2fence_region_map = torch.from_numpy(self.placedb.node2fence_region_map[:self.placedb.num_movable_nodes]).to(fence_region_list[0].device)
        self.op_collections.inner_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=self.name,
            fence_regions=fence_region_list[0],
            fence_region_mask=self.data_collections.node2fence_region_map>1e3) # density penalty for inner cells
        self.op_collections.outer_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=self.name,
            fence_regions = fence_region_list[1],
            fence_region_mask=self.data_collections.node2fence_region_map<1e3) # density penalty for outer cells

    def build_multi_fence_region_density_op(self):
        # region 0, ..., region n, non_fence_region
        self.op_collections.fence_region_density_ops = []

        for i, fence_region in enumerate(self.data_collections.virtual_macro_fence_region[:-1]):
            self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        name=self.name,
                        region_id=i,
                        fence_regions=fence_region)
            )

        self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        name=self.name,
                        region_id=len(self.placedb.regions),
                        fence_regions=self.data_collections.virtual_macro_fence_region[-1])
        )
        def merged_density_op(pos):
            ### stop mask is to stop forward of density
            ### 1 represents stop flag
            res = torch.stack([density_op(pos, mode="density") for density_op in self.op_collections.fence_region_density_ops])
            return res

        def merged_density_overflow_op(pos):
            ### stop mask is to stop forward of density
            ### 1 represents stop flag
            overflow_list, max_density_list = [], []
            for density_op in self.op_collections.fence_region_density_ops:
                overflow, max_density = density_op(pos, mode="overflow")
                overflow_list.append(overflow)
                max_density_list.append(max_density)
            overflow_list, max_density_list = torch.stack(overflow_list), torch.stack(max_density_list)

            return overflow_list, max_density_list

        self.op_collections.fence_region_density_merged_op = merged_density_op

        self.op_collections.fence_region_density_overflow_merged_op = merged_density_overflow_op
        return self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op