import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array
from ares.attack.utils import maybe_to_array, uniform_l_2_noise, uniform_l_inf_noise
from ares.loss import CrossEntropyLoss, Vods


class ODIPGDAttacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        ''' Based on ares.attack.bim.BIM '''
        self.name = 'odi-pgd'
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        # wd
        output_dim = 10 if dataset == 'cifar10' else 1000
        wd = uniform_l_inf_noise(batch_size, output_dim, tf.constant([1.]*self.batch_size), self.model.x_dtype)

        loss = CrossEntropyLoss(self.model)  # 定义loss
        loss_odi = Vods(self.model, wd)
        # random init magnitude
        self.rand_init_eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.rand_init_eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))

        # calculate init point within rand_init_eps
        d = np.prod(self.model.x_shape)
        noise = uniform_l_inf_noise(batch_size, d, self.rand_init_eps_var, self.model.x_dtype)

        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)

        # clip by (x_min, x_max)
        xs_init = tf.clip_by_value(tf.reshape(self.xs_ph, (self.batch_size, -1)) + noise,
                                   self.model.x_min, self.model.x_max)

        # flatten shape of xs_ph
        xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # odi step size
        self.alpha_ph_odi = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var_odi = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        alpha_odi = tf.expand_dims(self.alpha_var_odi, 1)
        # calculate loss' gradient with relate to the adversarial example
        # grad.shape == (batch_size, D)
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        self.loss = loss(self.xs_adv_model, self.ys_var)  # 计算loss
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]  # 得到对xs_adv的梯度
        # update the adversarial example
        xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps
        grad_sign = tf.sign(grad)
        # clip by max l_inf magnitude of adversarial noise
        xs_adv_next = tf.clip_by_value(self.xs_adv_var + alpha * grad_sign, xs_lo, xs_hi)  # 计算新值
        # clip by (x_min, x_max)
        xs_adv_next = tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max)

        # odi gradient
        self.odi_loss = loss_odi(self.xs_adv_model, self.ys_var)
        grad_odi = tf.gradients(self.odi_loss, self.xs_adv_var)[0]  # 得到对xs_adv的梯度
        # update the adversarial example
        grad_sign_odi = tf.sign(grad_odi)
        # clip by max l_inf magnitude of adversarial noise
        xs_adv_next_odi = tf.clip_by_value(self.xs_adv_var + alpha_odi * grad_sign_odi, xs_lo, xs_hi)  # 计算新值
        # clip by (x_min, x_max)
        xs_adv_next_odi = tf.clip_by_value(xs_adv_next_odi, self.model.x_min, self.model.x_max)

        # 初始化

        # clip by (x_min, x_max)
        xs_init = tf.clip_by_value(tf.reshape(self.xs_ph, (self.batch_size, -1)) + noise,
                                   self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)  # 用计算出的新值更新
        self.update_xs_adv_step_odi = self.xs_adv_var.assign(xs_adv_next_odi)  # 用计算出的新值更新
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)
        self.config_alpha_step_odi = self.alpha_var_odi.assign(self.alpha_ph_odi)
        self.config_rand_init_eps = self.rand_init_eps_var.assign(self.rand_init_eps_ph)

        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape)),
                         self.xs_adv_var.assign(xs_init)]
        self.setup_ys = self.ys_var.assign(self.ys_ph)

        self.iteration = 10
        # odi
        self.Nr = 5
        self.Nodi = 2
        # self.step_size = 1e-2

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            eps = maybe_to_array(self.eps, self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})  # 初始化
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: eps / 7})
            self._session.run(self.config_alpha_step_odi, feed_dict={self.alpha_ph_odi: eps / 7})

        # rand_init_magnitude = (1.0 / 255) * (self.model.x_max - self.model.x_min)
        rand_init_magnitude = kwargs['magnitude'] - 1e-6
        rand_init_eps = maybe_to_array(rand_init_magnitude, self.batch_size)
        self._session.run(self.config_rand_init_eps, feed_dict={self.rand_init_eps_ph: rand_init_eps})

    def batch_attack(self, xs, ys=None, ys_target=None):
        res = []
        loss_max = -1
        best_idx = None

        # odi
        for i in range(self.Nr):
            # self.__session.run() # 初始化x0和wd
            self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})  # 初始化
            self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys})

            for k in range(self.Nodi):
                # self.__session.run() # 参考pgd的update
                self._session.run(self.update_xs_adv_step_odi)

            # pgd

            for _ in range(self.iteration):  # 迭代K步
                self._session.run(self.update_xs_adv_step)
            res.append(self._session.run(self.xs_adv_model))  # 返回结果
            loss = self._session.run(self.loss).mean().item()
            # print(loss)
            if loss > loss_max:
                loss_max = loss
                best_idx = i

        return res[best_idx]
