import numpy as np
import numpy.matlib as matlib
from scipy.signal import chirp
import copy
from util import mmd2, set_point_ctrl
from examples import synthetic_example


# Excite the system to get data for system identification
def create_sys_id_data(sys, num_samples=100000):
    """
        Generate perturbed inputs and the corresponding outputs (e.g., the system states)
        @ In, num_samples, number of perturbed samples to generate
        @ Out, state_st, numpy.array, (number_states, num_samples+1) states due to perturbed inputs including initial states
        @ Out, inp_traj, numpy.array, (number_inputs, num_samples) perturbed input values
    """
    # Initialize input trajectory
    inp_traj = np.zeros((sys.inp_dim, num_samples))
    # Use chirp signal as input trajectory
    t = np.linspace(0, num_samples, num_samples)
    for i in range(sys.inp_dim):
        inp_traj[i, :] = chirp(t, 0.5*i, t[-1], 1*i)
    # Array to store state trajectory
    state_st = np.zeros((3, num_samples + 1))
    # Run experiment
    for i in range(num_samples):
        state_st[:, i + 1] = sys.step(inp_traj[:, i].reshape(-1, 1)).flatten()
    sys.state = np.zeros((3, 1))
    return state_st, inp_traj


# Identify the system dynamics
# (standard system identification via least squares)
def sys_id(state, inp_traj):
    """
        Identify the control system transition and input matrix using least squares
        @ In, state, numpy.array, (num_states, num_samples), generated data for state variables using sampled inputs
        @ In, inp_traj, numpy.array, (num_inputs, num_samples), sampled input data
        @ Out, A, numpy.array, (num_states, num_states), transition matrix
        @ Out, B, numpy.array, (num_states, num_inputs), input matrix
        @ Out, noise_stddev, numpy.array, (num_states, 1), noise input
    """
    data = np.vstack((state[:, 0:-2], inp_traj[:, 0:-1]))
    res = state[:, 1:-1]@np.linalg.pinv(data)
    A = res[0:len(state), 0:len(state)]
    B = res[0:len(state), len(state)::]
    noise_stddev = np.zeros((len(A[:, 0]), 1))
    for i in range(len(A[:, 0])):
        noise_stddev[i, 0] = np.sqrt(np.sum((state[i, 1:-1]
                                     - (res[i, :]@data))**2)
                                     / (len(state[0, :])-1))
    return [A, B, noise_stddev]


#
def sys_id_loc(state, inp_traj, test_infl_of):
    """
        Identify local system dynamics under non-causality assumption
        @ In, state, numpy.array, (num_states, num_samples), state data using random perturbed inputs
        @ In, inp_traj, numpy.array, (num_inputs, num_samples), randomly generated input data
        @ In, test_infl_of, int, index for currently testing variable (one of sys state variables + sys input variables)
        @ Out, A/B
    """
    data = np.vstack((state[:, 0:-2], inp_traj[:, 0:-1]))
    # We assume that variable 'test_infl_of' does not cause x_i
    # Thus, we delete this data before estimating the model
    data_tmp = np.delete(data, test_infl_of, 0)
    res = state[:, 1:-1]@np.linalg.pinv(data_tmp)
    res = np.insert(res, test_infl_of, 0, axis=1)
    A = res[0:len(state), 0:len(state)]
    B = res[0:len(state), len(state)::]
    return [A, B]


# Monte Carlo simulation of a given system
# with given initial state and input trajectories
def simulate_system(model, x0, u, rng, num_exp=1000):
    A, B, noise = model
    sys_dim = len(A[:, 0])
    # different noise for different perturbation
    noise_arr = rng.normal(0, matlib.repmat(noise, num_exp, 1),
                           (sys_dim*num_exp, len(u[0, :])))
    # One simulation with zero noise to get the mean
    noise_arr[0:sys_dim, :] = 0
    x_st = np.zeros((sys_dim*num_exp, len(u[0, :]) + 1))
    x_st[:, 0] = matlib.repmat(x0, num_exp, 1).flatten()
    # Extend matrices for parallel computing
    A_ext = np.kron(np.eye(num_exp), A)
    B_ext = np.kron(np.eye(num_exp), B)
    u_ext = matlib.repmat(u, num_exp, 1)
    for i in range(len(u[0, :])):
        x_st[:, i + 1] = (A_ext@x_st[:, i]
                          + B_ext@u_ext[:, i] + noise_arr[:, i])
    return x_st


# Get the test statistic of a system
# with initial conditions and input trajectories
def get_test_statistic(model, x0_I, x0_II, u_I, u_II, rng, num_exp=1000, nu=1):
    """
        Statistical estimation for the RHS equation (9)
        E[MMD(data1, data2|non-causal model)] + nu * sqrt(Var[MMD(data1, data2|non-causal model)])
        @ model, [A, B, noise], an identified model with assumption that test_infl_of variable is non-causal
    """
    sys_dim = len(model[0][:, 0])
    data_I = simulate_system(model, x0_I, u_I, rng, num_exp=num_exp)
    data_II = simulate_system(model, x0_II, u_II, rng, num_exp=num_exp)
    mmd_st = np.zeros((sys_dim, num_exp))
    mmd = np.zeros(sys_dim)
    for test_infl_on in range(sys_dim):
        for i in range(num_exp):
            mmd_st[test_infl_on, i] = mmd2(data_I[sys_dim*i
                                                  + test_infl_on, 1::],
                                           data_II[sys_dim*i
                                                   + test_infl_on, 1::])
        mmd[test_infl_on] = (mmd_st[test_infl_on, 0]
                             + nu*np.std(mmd_st[test_infl_on, 1::]))
    return mmd


# Steer the system to the initial conditions for the next experiment
def go_to_init(model, sys, ctrl, x_init, tolerance=1e-2):
    """
        @ In, model, initial identified model structure [A, B, noise_stddev] using least square
        @ In, sys, synthetic_example, initial model structure
        @ In, ctrl, a controller based on the initial model, ctrl = [F, Mx, Mu]
        @ In, x_init, numpy.array, initial values for state variable
        @ tolerance, float, control the system to reach initial position of state variable
    """
    F, Mx, Mu = ctrl
    while True:
        action = (Mu - F@Mx)@x_init + F@sys.state
        sys.step(action).flatten()
        if np.linalg.norm(sys.state - x_init) < tolerance:
            print("reached initial position for experiment")
            return


# Execute a causality testing experiment
def caus_exp(model, sys, x0_I, x0_II, u_I, u_II, ctrl):
    """
        @ In, model, initial identified model structure [A, B, noise_stddev] using least square
        @ In, sys, synthetic_example, initial model structure
        @ In, x0_I, numpy.array, initial condition for state variable for first experiment
        @ In, x0_II, numpy.array, initial condition for state variable for second experiment
        @ In, u_I, numpy.array, random samples for input variables for first experiment
        @ In, u_II, numpy.array, random samples for input variables for second experiment
        @ In, ctrl, a controller based on the initial model, ctrl = [F, Mx, Mu]
        @ Out, xI_st, numpy.array, (num_states, num_samples), new state variables values for first experiment
        @ Out, xII_st, numpy.array, (num_states, num_samples), new state variables values for second experiment
    """
    num_samples = len(u_I[0, :])
    xI_st = np.zeros((len(sys.high_obs), num_samples + 1))
    xII_st = copy.deepcopy(xI_st)
    # Go to initial position for first experiment
    go_to_init(model, sys, ctrl, x0_I)
    xI_st[:, 0] = sys.state.flatten()
    # Start first experiment
    print("start first experiment")
    for i in range(num_samples):
        xI_st[:, i + 1] = sys.step(u_I[:, i].reshape(-1, 1)).flatten()
    # Go to initial position for second experiment
    go_to_init(model, sys, ctrl, x0_II)
    xII_st[:, 0] = sys.state.flatten()
    # Start second experiment
    print("start second experiment")
    for i in range(num_samples):
        xII_st[:, i + 1] = sys.step(u_II[:, i].reshape(-1, 1)).flatten()
    return xI_st, xII_st


def caus_id(rng):
    """
        @ In, rng, Random Generator
        @ Out, None
    """
    sys = synthetic_example(rng)
    sys_dim = len(sys.high_obs)
    inp_dim = sys.inp_dim
    # Null hypothesis: no state/input has a causal influence on any state
    # rows: effect, columns: cause, for example caus_i,j = 1 indicates variable j has influence on variable i
    caus = np.zeros((sys_dim, sys_dim + inp_dim))
    # Start with standard system identification, generate system state data and input data for num_samples=100000
    sys_id_state, sys_id_inp = create_sys_id_data(sys)
    # initial estimation about the linear time-invariant model with Gaussian noise
    # identified [A, B, noise_stddev]
    init_model = sys_id(sys_id_state, sys_id_inp)
    # Get a controller based on the initial model, ctrl = [F, Mx, Mu]
    ctrl = set_point_ctrl(init_model[0], init_model[1], np.diag([1, 1, 1]),
                          np.diag([0.01, 0.01, 0.01]))
    # Start causal identification
    for test_infl_of in range(sys_dim + inp_dim):
        print("new causality test")
        # system state variable, test the influence from the initial state variables
        # if state variable, choose initial conditions of state variable as far apart as possible,
        # and choose the same random inputs trajectories to excite (perturb) the system,
        # elif input variable, choose the same initial conditions for state variables, and choose
        # different random generated input trajectories to excite (perturb) the system.
        if test_infl_of < sys_dim:
            print("testing influence of state ", test_infl_of)
            # Choose initial conditions as far apart as possible
            x0_I = np.zeros((sys_dim, 1))
            x0_I[test_infl_of, 0] = sys.high_obs[test_infl_of]
            x0_II = np.zeros((sys_dim, 1))
            x0_II[test_infl_of, 0] = -sys.high_obs[test_infl_of]
            # Choose the same input trajectory that excites the system
            u_I = rng.uniform(-1, 1, (inp_dim, 100))
            u_II = u_I
        else:
            test_inp = test_infl_of - sys_dim
            print("testing influence of input ", test_inp)
            # Choose initial position in 0
            x0_I = np.zeros((sys_dim, 1))
            x0_II = x0_I
            # Choose different input trajectories that excite the system
            u_I = 10*rng.uniform(-1, 1, (inp_dim, 100))
            u_II = copy.deepcopy(u_I)
            u_I[test_inp, :] = 100*rng.uniform(-1, 1, 100)
            u_II[test_inp, :] = np.zeros(100)
        # Do causality experiment, perform controled perturbation for both initial state variables and input variables
        # generate state variables data for two seperate experiment
        exp_data_I, exp_data_II = caus_exp(init_model, sys, x0_I, x0_II,
                                           u_I, u_II, ctrl)
        # Get model assuming variables are non-causal
        caus_model = sys_id_loc(sys_id_state, sys_id_inp, test_infl_of)
        caus_model.append(init_model[2])
        # Get test statistic, RHS of Eq. (9)
        test_stat = get_test_statistic(caus_model,
                                       exp_data_I[:, 0].reshape(-1, 1),
                                       exp_data_II[:, 0].reshape(-1, 1),
                                       u_I, u_II, rng, nu=5)
        print("Obtained test statistic")
        # Compute MMD and compare with test statistic, LHS of Eq. (9)
        for test_infl_on in range(sys_dim):
            if test_infl_of == test_infl_on:
                # remove effects from initial conditions
                exp_data_I[test_infl_on, :] -= x0_I[test_infl_on, 0]
                exp_data_II[test_infl_on, :] -= x0_II[test_infl_on, 0]
            mmd_exp = mmd2(exp_data_I[test_infl_on, :],
                           exp_data_II[test_infl_on, :])
            if mmd_exp > test_stat[test_infl_on]:
                caus[test_infl_on, test_infl_of] = 1
        print("new causality matrix:")
        print(caus)
        sys.state = np.zeros((3, 1))


if __name__ == '__main__':
    # construct a new Generator with the default BitGenerator (PCG64)
    rng = np.random.default_rng(987654)
    caus_id(rng)
