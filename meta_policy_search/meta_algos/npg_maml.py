from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import NGMAMLOptimizer
from meta_policy_search.optimizers.conjugate_gradient_optimizer import (
    conjugate_gradients,
    _flatten_params,
    _unflatten_params,
)

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class NGMAML(MAMLAlgo):
    """
    Natural Gradient Algorithm

    Args:
        policy (Policy): policy object
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable

    """

    def __init__(
        self, *args, name="ng_maml", learning_rate=1e-3, max_epochs=1, **kwargs
    ):
        super(NGMAML, self).__init__(*args, **kwargs)

        self.optimizer = NGMAMLOptimizer(
            learning_rate=learning_rate, max_epochs=max_epochs
        )
        self._optimization_keys = [
            "observations",
            "actions",
            "advantages",
            "agent_infos",
        ]
        self.name = name
        self.meta_op_phs_dict = OrderedDict()

        self.build_graph()

    def _adapt_objective_sym(
        self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym
    ):
        """ J^{LR} objective """
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(
                action_sym, dist_info_old_sym, dist_info_new_sym
            )
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph
        """

        with tf.variable_scope(self.name):
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = (
                self._make_input_placeholders()
            )
            self.meta_op_phs_dict.update(all_phs_dict)

            # for later use
            old_params = self.policy.get_params()
            old_params_flat = _flatten_params(old_params)

            meta_grads = []
            for i in range(self.meta_batch_size):  # for each task
                """ create fisher matrix """
                log_probs = self.policy.distribution.log_likelihood_sym(
                    action_phs[i], dist_info_old_phs[i]
                )  # the length of log_probs is the horizon of a trajectory
                fishers = []  # to store the "fisher matrix" for each (s, a) pair
                for log_prob in log_probs:
                    grad_log_prob = tf.gradients(log_prob, old_params_flat)
                    fishers.append(
                        tf.matmul(grad_log_prob, grad_log_prob, transpose_b=True)
                    )
                fisher = tf.reduce_mean(tf.stack(fishers), axis=0)

                """ create policy gradient ( \nabla J^{LR}(\theta) ) """
                dist_info_sym = self.policy.distribution_info_sym(
                    obs_phs[i], params=None
                )
                jlr_objective = self._adapt_objective_sym(
                    action_phs[i], adv_phs[i], dist_info_old_phs[i], dist_info_sym
                )
                policy_grad = tf.gradients(jlr_objective, old_params_flat)

                """ create gradient of adapted policy ( \nabla J^{LR}(\theta') ) """
                eta = 0.01

                adapt_direction = conjugate_gradients(fisher, policy_grad)  # Eq. (15)
                adapted_params = _unflatten_params(
                    flat_params=old_params_flat - eta * adapt_direction,
                    params_example=old_params,
                )
                self.policy.set_params(adapted_params)
                dist_info_adapted = self.policy.distribution_info_sym(
                    obs_phs, params=adapted_params
                )
                jrl_adapted = self._adapt_objective_sym(
                    action_phs, adv_phs, dist_info_old_phs, dist_info_adapted
                )
                grad_adapted = tf.gradients(jrl_adapted, old_params_flat)

                """ calculate meta gradient """
                hessian = tf.hessians(jlr_objective, old_params_flat)
                jacobian_Fu = tf.reduce_mean(...)  # TODO

                # Eq. (17)
                meta_grad = grad_adapted - eta * tf.transpose(
                    hessian - jacobian_Fu
                ) * conjugate_gradients(fisher, grad_adapted)

                meta_grads.append(meta_grad)

            meta_grad_mean = tf.reduce_mean(tf.stack(meta_grads), axis=0)
            grads_and_vars = list(zip(meta_grad_mean, old_params_flat))
            self.optimizer.build_graph(
                grads_and_vars, jrl_adapted, self.meta_op_phs_dict
            )

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        meta_op_input_dict = self._extract_input_dict_meta_op(
            all_samples_data, self._optimization_keys
        )

        if log:
            logger.log("Optimizing")
        loss_before = self.optimizer.optimize(meta_op_input_dict)

        if log:
            logger.log("Computing statistics")
        loss_after = self.optimizer.loss(meta_op_input_dict)

        if log:
            logger.logkv("LossBefore", loss_before)
            logger.logkv("LossAfter", loss_after)
