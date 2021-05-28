import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array
from ares.attack.utils import maybe_to_array, uniform_l_2_noise, uniform_l_inf_noise
from ares.loss import CrossEntropyLoss, CWLoss
from time import time
import sys


class Vods:
    ''' calculate vods '''

    def __init__(self, model, wd):
        self.model = model
        self.wd = wd

    def __call__(self, xs, ys):
        logits = self.model.logits(xs)
        vods = tf.matmul(tf.transpose(self.wd), logits)
        vods = vods / tf.norm(vods, 2)
        return vods


class MyTimer:
    def __init__(self) -> None:
        self.last = time()

    def logtime(self):
        cur = time()
        print(f"line {str(sys._getframe(1).f_lineno)}, elapsed time:{cur-self.last} s")
        self.last = cur


class ODIAutoPGDAttacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        ''' Based on ares.attack.bim.BIM '''
        self.name = 'odi-autopgd'
        self.model, self.batch_size, self._session = model, batch_size, session

        output_dim = 10 if dataset == 'cifar10' else 1000
        wd = uniform_l_inf_noise(batch_size, output_dim, tf.constant([1.]*self.batch_size), self.model.x_dtype)

        # loss = CrossEntropyLoss(self.model)  # 定义loss
        loss = CWLoss(self.model)  # 定义loss
        # loss = CrossEntropyLoss(self.model)
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
        self.xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=self.xs_flatten_shape, dtype=self.model.x_dtype))
        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=self.xs_flatten_shape, dtype=self.model.x_dtype))
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
        self.eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        alpha_odi = tf.expand_dims(self.alpha_var_odi, 1)

        # reset alpha
        self.reset_alpha_step = self.alpha_var.assign(self.eps_var * 2)

        # calculate loss' gradient with relate to the adversarial example
        # grad.shape == (batch_size, D)
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        self.loss = loss(self.xs_adv_model, self.ys_var)  # 计算loss
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]  # 得到对xs_adv的梯度
        # update the adversarial example
        xs_lo, xs_hi = self.xs_var - self.eps, self.xs_var + self.eps
        grad_sign = tf.sign(grad)
        # clip by max l_inf magnitude of adversarial noise
        self.xs_adv_next = tf.clip_by_value(self.xs_adv_var + alpha * grad_sign, xs_lo, xs_hi)  # 计算新值
        # clip by (x_min, x_max)
        self.xs_adv_next = tf.clip_by_value(self.xs_adv_next, self.model.x_min, self.model.x_max)

        # odi gradient
        self.odi_loss = loss_odi(self.xs_adv_model, self.ys_var)
        grad_odi = tf.gradients(self.odi_loss, self.xs_adv_var)[0]  # 得到对xs_adv的梯度
        # update the adversarial example
        grad_sign_odi = tf.sign(grad_odi)
        # clip by max l_inf magnitude of adversarial noise
        self.xs_adv_next_odi = tf.clip_by_value(self.xs_adv_var + alpha_odi * grad_sign_odi, xs_lo, xs_hi)  # 计算新值
        # clip by (x_min, x_max)
        self.xs_adv_next_odi = tf.clip_by_value(self.xs_adv_next_odi, self.model.x_min, self.model.x_max)

        # 初始化

        # clip by (x_min, x_max)
        # xs_init = tf.clip_by_value(tf.reshape(self.xs_ph, (self.batch_size, -1)) + noise,
        #                            self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(self.xs_adv_next)  # 用计算出的新值更新
        self.update_xs_adv_step_odi = self.xs_adv_var.assign(self.xs_adv_next_odi)  # 用计算出的新值更新
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)
        self.config_alpha_step_odi = self.alpha_var_odi.assign(self.alpha_ph_odi)
        self.config_rand_init_eps = self.rand_init_eps_var.assign(self.rand_init_eps_ph)

        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, self.xs_flatten_shape)),
                         self.xs_adv_var.assign(xs_init)]
        self.setup_ys = self.ys_var.assign(self.ys_ph)

        self.iteration = 100
        # odi
        self.Nr = 10
        self.Nodi = 2
        # self.step_size = 1e-2

        print('iteration = ', self.iteration)

        p = [0, 0.22]
        pcur = 0.22
        plast = 0
        ws = [0, round(self.iteration * 0.22)]
        while True:
            newp = pcur + max(pcur - plast - 0.03, 0.06)
            neww = round(self.iteration * newp)
            if neww > self.iteration:
                break
            plast = pcur
            pcur = newp
            ws.append(neww)
        self.ws = ws
        print(f"ws:{self.ws}")
        self.init_auto()

    def init_auto(self):
        self.fmax_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.xmax_ph = tf.placeholder(self.model.x_dtype, self.xs_flatten_shape)
        self.xlast_ph = tf.placeholder(self.model.x_dtype, self.xs_flatten_shape)
        self.alpha_last_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.flast_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.fmax_last_ph = tf.placeholder(self.model.x_dtype, (self.batch_size, ))

        self.cond1 = tf.placeholder(tf.bool, name="cond1")

        self.fmax = tf.Variable(tf.zeros(self.batch_size, ), dtype=tf.float32)
        self.xmax = tf.Variable(self.xs_adv_var)
        self.xlast = tf.Variable(self.xs_adv_var)
        self.alpha_last = tf.Variable(self.alpha_var)
        self.flast = tf.Variable(tf.zeros(self.batch_size, ), dtype=tf.float32)
        self.fmax_last = tf.Variable(tf.zeros(self.batch_size, ), dtype=tf.float32)

        self.update_fmax = self.fmax.assign(self.fmax_ph)
        self.update_xmax = self.xmax.assign(self.xmax_ph)
        self.update_xlast = self.xlast.assign(self.xlast_ph)
        self.update_alpha_last = self.alpha_last.assign(self.alpha_last_ph)
        self.update_flast = self.flast.assign(self.flast_ph)
        self.update_fmax_last = self.fmax_last.assign(self.fmax_last_ph)

        self.fcnt = 0
        self.k = 0
        self.wlast = 0

        w = self.xs_adv_var + 0.75 * (self.xs_adv_next - self.xs_adv_var) + 0.25 * (self.xs_adv_var - self.xlast)
        # project(w)
        w_lo, w_hi = w - self.eps, w + self.eps
        w = tf.clip_by_value(w, w_lo, w_hi)
        w = tf.clip_by_value(w, self.model.x_min, self.model.x_max)

        self.update_xadv_op = [self.xs_adv_var.assign(w), self.xlast.assign(self.xs_adv_var)]

        alpha_notchange = tf.reduce_all(tf.equal(self.alpha_last, self.alpha_var))
        fmax_notchange = tf.reduce_all(tf.equal(self.fmax_last, self.fmax))

        self.cond2 = tf.logical_and(alpha_notchange, fmax_notchange)

        def cond1_or_cond2_true():
            op = [self.alpha_var.assign(self.alpha_var / 2),
                  self.xs_adv_var.assign(self.xmax), self.alpha_last.assign(self.alpha_var),
                  self.fmax_last.assign(self.fmax)]
            return op

        def cond1_or_cond2_false():
            op = [self.alpha_var,
                  self.xs_adv_var, self.alpha_last.assign(self.alpha_var),
                  self.fmax_last.assign(self.fmax)]
            return op

        self.update_step_size = tf.cond(tf.logical_or(self.cond1, self.cond2),
                                        cond1_or_cond2_true, cond1_or_cond2_false)

        def update_xmax_fmax():
            # self.fcnt += 1
            return [self.fmax.assign(self.loss), self.xmax.assign(w)]

        def dummy_f():
            return [self.fmax, self.xmax]

        def update_fcnt():
            self.fcnt += 1
            return self.flast.assign(self.loss)
        
        def dummy_update_fcnt():
            return self.flast.assign(self.loss)

        # with tf.control_dependencies([update_xadv_op]):
        self.update_xmax_fmax_step = tf.cond(tf.reduce_mean(
            self.loss) > tf.reduce_mean(self.fmax), update_xmax_fmax, dummy_f)

        self.update_fcnt_step = tf.cond(tf.reduce_mean(
            self.loss) > tf.reduce_mean(self.flast), update_fcnt, dummy_update_fcnt)

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            eps = maybe_to_array(self.eps, self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})  # 初始化
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: eps / 7})
            self._session.run(self.config_alpha_step_odi, feed_dict={self.alpha_ph_odi: eps / 7})

        rand_init_magnitude = (1.0 / 255) * (self.model.x_max - self.model.x_min)
        # rand_init_magnitude = kwargs['magnitude'] - 1e-6
        rand_init_eps = maybe_to_array(rand_init_magnitude, self.batch_size)
        self._session.run(self.config_rand_init_eps, feed_dict={self.rand_init_eps_ph: rand_init_eps})

    def batch_attack(self, xs, ys=None, ys_target=None):
        res = []
        # loss_max = -1
        best_idx = None
        succ_max = -1
        # odi
        # print("start new attack")
        for i in range(self.Nr):
            # self.__session.run() # 初始化x0和wd
            self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})  # 初始化
            self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys})

            for k in range(self.Nodi):
                # self.__session.run() # 参考pgd的update
                self._session.run(self.update_xs_adv_step_odi)

            # 复原alpha
            self._session.run(self.reset_alpha_step)
            mytimer = MyTimer()

            # 初始化
            # fmax = tf.Variable(tf.zeros(self.batch_size, ), dtype=tf.float32)
            # xmax = tf.Variable(self.xs_adv_var)
            f0, x0, _ = self._session.run([self.loss, self.xs_adv_var, self.update_xs_adv_step])
            # x0 = self._session.run(self.xs_adv_var)
            # _ = self._session.run(self.update_xs_adv_step)
            mytimer.logtime()
            # self._session.run(self.update_xs_adv_step)
            f1 = self._session.run(self.loss)
            x1 = self._session.run(self.xs_adv_var)
            mytimer.logtime()
            # fcnt = 0
            if f0.mean() >= f1.mean():
                self._session.run([self.update_fmax, self.update_xmax], feed_dict={self.fmax_ph: f0, self.xmax_ph: x0})
            else:
                self._session.run([self.update_fmax, self.update_xmax], feed_dict={self.fmax_ph: f1, self.xmax_ph: x1})
                self.fcnt += 1

            # fmax_last = tf.Variable(self.fmax)

            # self.xlast = tf.Variable(self.xs_adv_var)
            # alpha_last = tf.Variable(self.alpha_var)
            # flast = tf.Variable(self.fmax)
            # op = [self.xlast.assign(x0), alpha_last.assign(self.alpha_var), flast.assign(f1)]

            # self._session.run(op)
            self._session.run([self.update_xlast, self.update_alpha_last, self.update_flast, self.update_fmax_last], feed_dict={
                              self.xlast_ph: x0, self.alpha_last_ph: self._session.run(self.alpha_var), self.flast_ph: f1, self.fmax_last_ph: self._session.run(self.fmax)})
            # wlast = 0
            mytimer.logtime()
            # print(type(fmax))
            print(f"{i} in odi")
            # print(f"{self._session.run(self.fmax_last.initializer)}")
            for k in range(1, self.iteration):  # 迭代K-1步
                # self._session.run(self.update_xs_adv_step)
                self._session.run(self.update_xadv_op)

                if k in self.ws:
                    self.k = k

                    self._session.run(self.update_step_size, feed_dict={
                                      self.cond1: self.fcnt < 0.75 * (self.k - self.wlast)})

                    self.fcnt = 0
                    self.wlast = k

                self._session.run([self.update_xmax_fmax_step, self.update_fcnt_step])

                logits = self.model.logits(self.xs_adv_model)
                preds = tf.argmax(logits, 1)
                preds = self._session.run(preds)
                succ = (preds != ys).sum()
                print('succ = ', succ)

                if k % 20 == 0:
                    print('k = ', k)
                # mytimer.logtime()

            xs_adv = self._session.run(self.xs_adv_model).astype(np.float32)
            res.append(xs_adv)  # 返回结果
            # xs_adv = np.clip(xs_adv, xs - self.eps * (self.model.x_max - self.model.x_min), xs + self.eps * (self.model.x_max - self.model.x_min))
            # xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            logits = self.model.logits(self.xs_adv_model)
            preds = tf.argmax(logits, 1)
            preds = self._session.run(preds)
            succ = (preds != ys).sum()
            print('succ = ', succ)
            # loss = self._session.run(self.loss).mean().item()
            # print(loss)
            # if loss > loss_max:
            #     loss_max = loss
            #     best_idx = i
            if succ > succ_max:
                succ_max = succ
                best_idx = i

        return res[best_idx]
