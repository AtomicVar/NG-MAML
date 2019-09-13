from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import NGMAMLOptimizer
from meta_policy_search.optimizers.conjugate_gradient_optimizer import conjugate_gradients, ConjugateGradientOptimizer, \
    _flatten_params

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
            self,
            *args,
            name="ng_maml",
            learning_rate=1e-3,
            max_epochs=1,
            num_minibatches=1,
            target_inner_step=0.01,
            init_inner_kl_penalty=1e-2,
            adaptive_inner_kl_penalty=True,
            **kwargs
    ):
        super(NGMAML, self).__init__(*args, **kwargs)

        self.optimizer = NGMAMLOptimizer(learning_rate=learning_rate, max_epochs=max_epochs,
                                         num_minibatches=num_minibatches)
        self.target_inner_step = target_inner_step
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = init_inner_kl_penalty * np.ones(self.num_inner_grad_steps)
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        """ J^{LR} objective """
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                   dist_info_old_sym, dist_info_new_sym)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph
        """

        with tf.variable_scope(self.name):
            # cg_optimizer = ConjugateGradientOptimizer()
            eta = 0.01

            adapt_direction = conjugate_gradients(fisher, grad)  # Eq. (15)
            adapted_policy_params = _flatten_params(self.policy.get_param_values()) - eta * adapt_direction  # \theta'
            grad_adapted = ...

            # Eq. (17)
            meta_grad = grad_adapted - eta * (hessian - jacobian_Fu).T * conjugate_gradients(fisher,
                                                                                             grad_adapted)

            grads_and_vars = list(zip(meta_grad, self.policy.get_params()))
            meta_objective = ...  # L(\theta)

            self.optimizer.build_graph(
                loss=meta_objective,
                grads_and_vars=grads_and_vars,
                input_ph_dict=self.meta_op_phs_dict
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
        meta_op_input_dict = self._extract_input_dict_meta_op(all_samples_data, self._optimization_keys)

        if log: logger.log("Optimizing")
        loss_before = self.optimizer.optimize(meta_op_input_dict)

        if log: logger.log("Computing statistics")
        loss_after, inner_kls, outer_kl = self.optimizer.compute_stats(meta_op_input_dict)

        if self.adaptive_inner_kl_penalty:
            if log: logger.log("Updating inner KL loss coefficients")
            self.inner_kl_coeff = self.adapt_kl_coeff(self.inner_kl_coeff, inner_kls, self.target_inner_step)

        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('KLInner', np.mean(inner_kls))
            logger.logkv('KLCoeffInner', np.mean(self.inner_kl_coeff))

    def adapt_kl_coeff(self, kl_coeff, kl_values, kl_target):
        if hasattr(kl_values, '__iter__'):
            assert len(kl_coeff) == len(kl_values)
            return np.array([_adapt_kl_coeff(kl_coeff[i], kl, kl_target) for i, kl in enumerate(kl_values)])
        else:
            return _adapt_kl_coeff(kl_coeff, kl_values, kl_target)


def _adapt_kl_coeff(kl_coeff, kl, kl_target):
    if kl < kl_target / 1.5:
        kl_coeff /= 2

    elif kl > kl_target * 1.5:
        kl_coeff *= 2
    return kl_coeff
