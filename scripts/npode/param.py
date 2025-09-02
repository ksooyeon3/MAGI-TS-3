import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter
from gpflow.utilities import positive


float_type = tf.float64

class Param:
    """
    A GPflow-1.x‐style Param + transforms replacement,
    implemented purely with tf.Variable and TFP bijectors.
    """
    def __init__(self,
                 value,
                 transform=None,    # None, "positive", or a tfp.bijectors.Bijector
                 fixed=False,
                 name=None,
                 learning_rate=None,  # currently unused
                 summ=False):
        self.value = value
        self.fixed = fixed
        self.name = name or "param"

        # pick the bijector
        if transform is None:
            bijector = tfp.bijectors.Identity()
        elif transform == "positive":
            bijector = positive()   # softplus‐based
        else:
            # assume a TFP Bijector
            bijector = transform

        self._bijector = bijector

        if fixed:
            # store a constant in the _constrained_ space
            self._var = tf.constant(value, dtype=float_type, name=self.name)
        else:
            # invert the init-value into the unconstrained space
            # init_unconstrained = bijector.inverse(value)
            # self._var = tf.Variable(init_unconstrained,
            #                          dtype=float_type,
            #                          name=self.name)
            self._var = Parameter(value,
                                  transform=bijector,
                                  name=self.name)
            
        if summ:
            # histogram of the _unconstrained_ variable
            tf.summary.histogram(self.name, self._var)

    def __call__(self):
        # always return the _constrained_ value
        if self.fixed:
            return self._var
        return self._bijector.forward(self._var)

    def assign(self, new_value):
        """
        Overwrite with a new _constrained_ value:
        we push it back through inverse before assigning.
        """
        if self.fixed:
            raise RuntimeError("Param is fixed; cannot assign.")
        unconstrained = self._bijector.inverse(new_value)
        self._var.assign(unconstrained)

    @property
    def shape(self):
        # purely Python shape of the original `value`
        return self.value.shape




# import tensorflow as tf
# from gpflow import transforms
# # from gpflow.base import Parameter
# # from gpflow.utilities import positive

# float_type = tf.float64

# #class Variable(Variable):
# #    '''
# #    extend tf.Variable to have properties : learning_rate
# #    '''
# #    pass
# #
# #    def set_learning_rate(self,value):
# #        self._learning_rate = value
# #
# #    @property
# #    def learning_rate(self):
# #        if hasattr(self,'_learning_rate'):
# #            return self._learning_rate
# # 
# #        else:
# #            return 0.001

# class Param:
#     '''
#     Inheriting from GPFlow
#     TODO : add a fixed flag in which case this should return tf.tensor instead of tf.Variable
#     '''
#     def __init__(self,value,transform = None,fixed=False,name=None,learning_rate=None,summ=False):
#         self.value = value
#         self.fixed = fixed

#         if name is None:
#             self.name = "param"
#         else:
#             self.name = name

#         if transform is None:
#             self.transform=transforms.Identity()
#         else:
#             self.transform = transform

#         if self.fixed:
#             self.tf_opt_var = tf.constant(self.value,name=name,dtype=float_type)
#         else:
#             # self.tf_opt_var = Variable(self.transform.backward(self.value),name=name,dtype=float_type)
#             self.tf_opt_var = tf.Variable(self.transform.backward(self.value),name=name,dtype=float_type)

# #        if learning_rate is not None and not self.fixed:
# #            self.tf_opt_var.set_learning_rate(learning_rate)

#         if summ:
#             self.variable_summaries(self.tf_opt_var)

#     def __call__(self):
#         if self.fixed:
#             return self.tf_opt_var
#         else:
#             return self.transform.forward_tensor(self.tf_opt_var)

#     def __set__(self, instance, value):
#         self.tf_opt_var.assign(self.transform.backward(value))

#     def variable_summaries(self,var):
#         """Attach tensorBoard visualization"""
#         tf.summary.histogram(self.name, var)

#     @property
#     def shape(self):
#         return self.value.shape
