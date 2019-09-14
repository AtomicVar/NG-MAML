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
        name (str): tf variable scope
        learning_rate (float): learning rate for optimizing the meta-objective
        num_ppo_steps (int): number of ProMP steps (without re-sampling)
        num_minibatches (int): number of minibatches for computing the ppo gradient steps
        clip_eps (float): PPO clip range
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for annealing clip_eps. If anneal_factor < 1, clip_eps <- anneal_factor * clip_eps at each iteration
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
            # === create fisher matrix ===
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = (
                self._make_input_placeholders()
            )
            self.meta_op_phs_dict.update(all_phs_dict)

            for i in range(self.meta_batch_size):  # for each task
                log_prob = self.policy.distribution.log_likelihood_sym(
                    obs_phs[i], dist_info_old_phs[i]
                )
                fisher = tf.reduce_mean(
                    tf.matmul(log_prob, tf.transpose(log_prob))
                )  # TODO: it is wrong!

            # === create policy gradient ( \nabla J^{LR}(\theta) ) ===
            dist_info_sym = self.policy.distribution_info_sym(obs_phs, params=None)
            jlr_objective = self._adapt_objective_sym(
                action_phs, adv_phs, dist_info_old_phs, dist_info_sym
            )
            original_params = self.policy.get_params()  # for later use
            policy_grad = tf.gradients(jlr_objective, _flatten_params(original_params))

            # === create gradient of adapted policy ( \nabla J^{LR}(\theta') ) ===
            eta = 0.01

            adapt_direction = conjugate_gradients(fisher, policy_grad)  # Eq. (15)
            adapted_params = _unflatten_params(
                flat_params=_flatten_params(self.policy.get_params()) - eta * adapt_direction,
                params_example=self.policy.get_params()
            )
            self.policy.set_params(adapted_params)
            dist_info_adapted = self.policy.distribution_info_sym(
                obs_phs, params=adapted_params
            )
            jrl_adapted = self._adapt_objective_sym(
                action_phs, adv_phs, dist_info_old_phs, dist_info_adapted
            )
            grad_adapted = tf.gradients(
                jrl_adapted, _flatten_params(self.policy.get_params())
            )

            # === calculate meta gradient ===
            hessian = tf.hessians(jlr_objective, original_params)
            jacobian_Fu = tf.reduce_mean(...)  # TODO

            # Eq. (17)
            meta_grad = -(
                grad_adapted
                - eta
                * tf.transpose(hessian - jacobian_Fu)
                * conjugate_gradients(fisher, grad_adapted)
            )

            grads_and_vars = list(zip(meta_grad, self.policy.get_params()))
            self.optimizer.build_graph(grads_and_vars, self.meta_op_phs_dict)

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
