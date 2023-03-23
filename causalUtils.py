import numpy as np
import numpy.matlib as matlib
import scipy
from sklearn.metrics.pairwise import rbf_kernel as rbf
import copy
import GPy

# Calculate the maximum mean discrepancy
def mmd2(x, y):
  """
    compute MMD Square using unbiased empirical estimate
    @ In, x, numpy.array, (numSamples, numVars)
    @ In, y, numpy.array, (numSamples, numVars)
    @ Out, empiricalMMD2, float, An unbiased empirical estimate of the squared population MMD
  """
  x = np.atleast_1d(x)
  y = np.atleast_1d(y)
  xy = np.hstack((x, y)).reshape(-1, 1)
  sigma = np.median(scipy.spatial.distance.cdist(xy, xy))/2
  gamma = 1.0/(2.0*sigma**2)
  num_samples = float(np.min([len(x), len(y)]))
  if len(x.shape) == 1:
    x = x.reshape(-1, 1)
  if len(y.shape) == 1:
    y = y.reshape(-1, 1)
  K = rbf(x, x, gamma=gamma)
  L = rbf(y, y, gamma=gamma)
  KL = rbf(x, y, gamma=gamma)
  np.fill_diagonal(K, 0)
  np.fill_diagonal(L, 0)
  np.fill_diagonal(KL, 0)
  empiricalMMD2 = (1/(num_samples*(num_samples-1)))*(np.sum(K+L-KL-KL.T))
  return empiricalMMD2


def generateVariationData(initCondMax, initCondMin,inpMax, inpMin, testCausalVarIndex, rng, numExp=10, T=100,
                          initCondStd=1.e-2, nu=10):
  """
    @ In, initCondMax, list
    @ In, initCondMin, list
    @ In, inpMax, list
    @ In, inpMax, list
    @ In, testCausalVarIndex, int
    @ In, inpVars, list, list of input variables
    @ In, rng, random number generator object,
  """
  assert len(initCondMax) == len(initCondMin), f"length for 'initCondMax' {initCondMax} should be the same as 'initCondMin' {initCondMin}"
  assert len(inpMax) == len(inpMin), f"length for 'inpMax' {inpMax} should be same as 'inpMin' {inpMin}"
  dimState = len(initCondMax)
  dimInp = len(inpMax)
  initCond1 = np.zeros((numExp, dimState))
  initCond2 = np.zeros((numExp, dimState))
  inpTraj1 = np.zeros((T, dimInp))
  inpTraj2 = np.zeros((T, dimInp))
  # For the first type experiment, we investigate whether xj causes xi. We conduct two experiments with different initial
  # conditions xj_I(0) n.e. xj_II(0) while all others are kept the same
  for i in range(dimState):
    if i != testCausalVarIndex:
      # for xi not the causal variable, keep the initial condition the same, or generate them follow the same distribution
      initCond1[:, i] = rng.normal(np.mean([initCondMax[i], initCondMin[i]]), initCondStd, numExp)
      initCond2[:, i] = rng.normal(np.mean([initCondMax[i], initCondMin[i]]), initCondStd, numExp)
    else:
      # for test causal variable, generate initial conditions far away from each others
      initCond1[:, i] = rng.normal(initCondMin[i]+nu*initCondStd, initCondStd, numExp)
      initCond2[:, i] = rng.normal(initCondMax[i]-nu*initCondStd, initCondStd, numExp)
  # generate input trajectories
  # Remark: note that for the testing experiments, we design open-loop trajectories. That is, the input
  # cannot depend on the system's current state. This is essential since it create independent trajectories
  #  for which we can leverage the MMD as a similarity measure
  if testCausalVarIndex < dimState:
    inpTraj1 = rng.uniform(inpMin, inpMax, (T, dimInp))
    inpTraj2 = copy.deepcopy(inpTraj1)
  else:
    idx = testCausalVarIndex - dimState
    inpTraj1 = rng.uniform(inpMin, inpMax, (T, dimInp))
    inpTraj2 = copy.deepcopy(inpTraj1)
    inpTraj1[:, idx] = inpMin[idx]
    inpTraj2[:, idx] = inpMax[idx]

  return initCond1, initCond2, inpTraj1, inpTraj2

# Get actual MMD, LHS of eq. (9), conduct two experiment, and compute MMD for the two results
def computeMMD2ForGivenSystem(sys, testCausalVarIndex, testEffectVarsIndex, initCond1, inpTraj1,
             initCond2, inpTraj2=0, numExp=1):
  """
    Compute the MMD square for given system
    @ In, sys, a model object with self.state variable and self.dynamics method
      self.state state state values
      self.dynamics accept controls/inputs and return new state values
    @ In, testCausalVarIndex, int, index for causal var, where var that will be tested if it have influence on testEffectVars
    @ In, testEffectVarsIndex, list, list of indices for effect vars, vars that will be tested if the tested causal variable have influence on them
    @ In, initCond1, numpy.array, (numExp, numberStateVars), random perturbed initial condition
    @ In, inpTraj1, numpy.array, (inpDim, numTimeSteps), random perturbed input trajectories
    @ In, initCond2, numpy.array, (numExp, numberStateVars), random perturbed initial condition, should be distinct from initCond1 when
       for testCausalVar
    @ In, inpTraj2, numpy.array, (inpDim, numTimeSteps), random perturbed input trajectories, when test causal var is on of input, set the trajectories values
       for testCausalVar distinct for two experiment (for example, min and max for each experiment respectively)
    @ In, numExp, int, number of experiments
    @ Out,
  """
  sys1 = sys
  sys2 = copy.deepcopy(sys)
  for idx in range(numExp):
    # assign initial condition to sys state
    sys1.state = initCond1[idx, :].reshape(-1, 1)
    sys2.state = initCond2[idx, :].reshape(-1, 1)
    try:
      len(inpTraj2)
    except:
      inpTraj2 = inpTraj1
    # dims: numStateVars X (T+1)
    x_st = np.zeros((len(sys1.state), inpTraj1.shape[1]+1)) # state variable values for initCond1 and inpTraj1
    y_st = np.zeros((len(sys1.state), inpTraj1.shape[1]+1)) # state variable values for initCond2 and inpTraj2
    # store the initial values
    x_st[:, 0] = initCond1[idx, :]
    y_st[:, 0] = initCond2[idx, :]
    # loop over time
    for i in range(inpTraj1.shape[1]):
      # solve system dynamics for each time step with given input perturbation
      sys1.dynamics(inpTraj1[:, i])
      sys2.dynamics(inpTraj2[:, i])
      # collect state values into x and y
      x_st[:, i+1] = sys1.state[:, 0]
      y_st[:, i+1] = sys2.state[:, 0]
    # collect numExp times of x_st and y_st into x_mult and y_mult
    # [x_st_exp1, x_st_exp2, ..., x_st_numExp], [y_st_exp1, y_st_exp2, ..., y_st_numExp]
    # dim: numStateVars X [(T+1)*numExp]
    try:
      x_mult = np.hstack((x_mult, x_st))
      y_mult = np.hstack((y_mult, y_st))
    except:
      x_mult = copy.deepcopy(x_st)
      y_mult = copy.deepcopy(y_st)
  mmd = np.zeros(len(testEffectVarsIndex))
  lenT = inpTraj1.shape[1] + 1
  for idx, i in enumerate(testEffectVarsIndex):
    # if test effect var is same as test causal var (in this case, only state variables),
    # remove initial condition before testing causal relationship
    if i == testCausalVarIndex:
      for j in range(numExp):
        x_mult[i, j*lenT:(j+1)*lenT] -= initCond1[j, i]
        y_mult[i, j*lenT:(j+1)*lenT] -= initCond2[j, i]
    # compute mmd2 of given test effect variable for two seperate set of variations (including initial condition and input variations)
    # mmd2 reflect the effects from all input and initial condition variations
    # this will be compared with results from non-causal model
    # If this computed value is larger, which means test causal variable will actuallly cause the discrepancy in mmd2
    mmd[idx] = mmd2(x_mult[i, :], y_mult[i, :])
  return mmd


# Predict MMD based on the GP model
# construct a surrogate model with non-causal assumption, use the surrogate as the predictor, compute
#  E(MMD(non_causal)) + nu * sqrt(Var[MMD(non_causal)])
#  similar function as cause_id_linear.get_test_statistic
def predictMMDWithSurrogate(GPs, test_infl_of, test_infl_on, init_cond1,
                inp_traj1, init_cond2, inp_traj2, rng, num_exp=0, num_var=0):
    dimState = init_cond1.shape[1]
    x_st = np.zeros((num_var*num_exp, dimState*inp_traj1.shape[1]
                    + dimState))
    y_st = np.zeros((num_var*num_exp, dimState*inp_traj1.shape[1]
                    + dimState))
    x_st[:, 0:dimState] = np.matlib.repmat(init_cond1, num_var, 1)
    y_st[:, 0:dimState] = np.matlib.repmat(init_cond2, num_var, 1)
    # Simulate system and store results using Monte Carlo sampling, data is generated from check_struct method
    for i in range(inp_traj1.shape[1]):
        X = np.hstack((x_st[:, i*dimState:i*dimState
                            + dimState],
                       np.matlib.repmat(inp_traj1[:, i],
                                        num_var*num_exp, 1)))
        Y = np.hstack((y_st[:, i*dimState:i*dimState
                            + dimState],
                       np.matlib.repmat(inp_traj2[:, i],
                                        num_var*num_exp, 1)))
        for idx, GP in enumerate(GPs):
            # remove data for given testing variable
            X_tmp = np.delete(X, GP.indep_var, 1)
            Y_tmp = np.delete(Y, GP.indep_var, 1)
            x_st[:, (i+1)*dimState + idx] = (
                x_st[:, i*dimState + idx]
                + rng.normal(GP.model.predict(X_tmp)[0],
                             (GP.model.predict(X_tmp)[1]))[:, 0])
            y_st[:, (i+1)*dimState+idx] = (
                y_st[:, i*dimState+idx]
                + rng.normal(GP.model.predict(Y_tmp)[0],
                             (GP.model.predict(Y_tmp)[1]))[:, 0])
    # Calculate MMD
    mmd = np.zeros((num_var, len(test_infl_on)))
    for idx, i in enumerate(test_infl_on):
        if i == test_infl_of:
            # remove initial conditions for testing
            for j in range(num_exp):
                x_st[j*num_var:(j+1)*num_var, i::dimState] -= (
                    init_cond1[j, i])
                y_st[j*num_var:(j+1)*num_var, i::dimState] -= (
                    init_cond2[j, i])
        for j in range(num_var):
            mmd[j, idx] = mmd2(x_st[num_exp*j:num_exp*j+num_exp,
                               i::dimState].flatten(),
                               y_st[num_exp*j:num_exp*j+num_exp,
                               i::dimState].flatten())
    mmd_std = np.std(mmd, axis=0)
    mmd_mean = np.mean(mmd, axis=0)
    return mmd_mean, mmd_std

# Learn a GP model for the dynamical system
class GPmodel:
    def __init__(self, num_out, id_data, indep_var, T=1000, model=None):
        """

          num_out: index for GP model, and the index for the state varialble that will be predicted by GP model
          id_data: state data x_st (dimState X timeSteps) + sampled input data u (dimInp X timeSteps)
          indep_var: index for variable that will be treated independent, and will be removed from GP training
        """
        self.indep_var = indep_var
        self.num_out = num_out
        self.min_val = 0
        self.max_val = 0
        self.id_data = id_data
        if model is not None:
            self.model = model
        else:
            self.model = self.ident_GP_model(T)
        self.min_val = np.min(id_data[0][num_out, :])
        self.max_val = np.max(id_data[0][num_out, :])

    def ident_GP_model(self, T):
        self.input_dim = self.id_data[0].shape[0] + self.id_data[1].shape[0]
        # Reduce input dimension in case of independent variables
        try:
            for el in self.indep_var:
                self.input_dim -= 1
        except:
            pass
        # Matern kernel
        ker = GPy.kern.Matern32(input_dim=self.input_dim, ARD=True)
        ker.unconstrain()
        x_st = self.id_data[0]
        u = self.id_data[1]
        # generate data and fit a GP model: x(t+1) - x(t) = GP[x(t), u(t)]
        Y = x_st[self.num_out, 1::] - x_st[self.num_out, 0:-1]
        X = np.vstack((x_st[:, 0:-1], u[:, 0:-1]))
        # Delete independent variables from input
        try:
            for el in self.indep_var:
                X = np.delete(X, el, 0)
        except:
            pass
        # Fit and optimize GP model
        model = GPy.core.gp.GP(X.T, Y.reshape(-1, 1), ker,
                               GPy.likelihoods.Gaussian(),
                               inference_method=GPy.inference.latent_function_inference.ExactGaussianInference(),
                               normalizer=True)
        return model
