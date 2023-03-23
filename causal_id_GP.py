from causalUtils import mmd2, generateVariationData, computeMMD2ForGivenSystem, GPmodel, predictMMDWithSurrogate
from examples import mult_tank_system
import numpy as np
import GPy
import copy


# Excite the system to get data for system identification
def sys_id_data(sys, rng, T=1000):
    """
        Sample input 'u' and collect states x_st
    """
    u = rng.uniform(sys.inp_bound[0, :], sys.inp_bound[1, :], (T, sys.inp_dim))
    x_st = np.zeros((len(sys.state), T))
    for idx, inp in enumerate(u):
        state = sys.dynamics(inp)
        x_st[:, idx] = state[:, 0]
    return [x_st, u.T]


# Get the causal structure of the system
def check_struct(GPs, sys1, sys2, id_data, test_infl_of, test_infl_on,
                 rng, nu=10, init_cond_std=1e-2, num_exp=10, num_var=50):
    """
        GPs: trained GP model
        sys1: multi-tank system
        sys2: multi-tank system
        id_data: initial training data
        test_infl_of: int, index of state variable + input variable
        test_infl_on: list of index of state variable in tank.state
        nu: factor for std of MMD(data_I, data_II, f_non_causal_model)
        num_exp: number of experiments/testing
    """
    dimState = len(sys1.state)
    dimInp = sys1.inp_dim
    initCondMax = np.zeros(dimState)
    initCondMin = np.zeros(dimState)
    inpMax = np.zeros(dimInp)
    inpMin = np.zeros(dimInp)
    for i in range(dimState):
        initCondMax[i] = GPs[i].max_val
        initCondMin[i] = GPs[i].min_val
    for i in range(dimInp):
        inpMax[i] = sys1.inp_bound[1, i]
        inpMin[i] = sys1.inp_bound[0, i]

    init_cond1, init_cond2, inp_traj, inp_traj2 = generateVariationData(initCondMax, initCondMin,inpMax, inpMin, test_infl_of, rng=rng, numExp=num_exp, T=100,
                          initCondStd=init_cond_std, nu=nu)

    GPs_test = GPs
    num_gps = list(np.linspace(0, dimState-1, dimState))
    num_gps = [int(x) for x in num_gps]
    # Estimate GPs with parallel computing
    # independent variable (test_infl_of)
    GPs_test = [GPmodel(gp, id_data, [test_infl_of]) for gp in num_gps]
    # RHS of eq. (9)
    e_mmd, std_mmd = predictMMDWithSurrogate(GPs_test, test_infl_of,
                                    test_infl_on, init_cond1,
                                    inp_traj.T, init_cond2, inp_traj2.T, rng,
                                    num_exp=num_exp, num_var=num_var)
    # LHS of eq. (9)
    exp_mmd = computeMMD2ForGivenSystem(sys2, test_infl_of, test_infl_on, init_cond1, inp_traj.T,
                    init_cond2, inp_traj2.T, num_exp)
    return exp_mmd > e_mmd + nu*std_mmd


def create_multi_tank_system(rng):
    # Create quadruple tank system (4 tanks, 2 inputs)
    num_tanks = 4
    connections = np.array([[2], [3], [], []], dtype=object)
    inp_dim = 2
    inp_mapping = np.array([[0], [1], [1], [0]])
    multi_tank = mult_tank_system(num_tanks, connections, inp_dim,
                                  inp_mapping)
    return multi_tank


if __name__ == '__main__':
    rng = np.random.default_rng(987654)
    sys1 = create_multi_tank_system(rng)
    sys2 = create_multi_tank_system(rng)
    # sample inputs and collect system states [x_st, u.T]
    data = sys_id_data(sys1, rng)
    num_gps = list(np.linspace(0, len(sys1.state)-1, len(sys1.state)))
    num_gps = [int(x) for x in num_gps]
    # fit the GP model using sampled data
    GPs = [GPmodel(gp, data, []) for gp in num_gps]
    model_arr = []
    # retrieve the trained GP model
    for i in range(len(sys1.state)):
        model_arr = np.append(model_arr, GPs[i].model)
    num_tests = len(sys1.state) + sys1.inp_dim
    test_infl_of = [0, 2, 4, 6, 8, 9]
    test_infl_on = [0, 2, 4, 6]

    caus = np.zeros((len(test_infl_on), len(test_infl_of)))
    for idx1, el in enumerate(test_infl_of):
        indep_el = check_struct(GPs, sys1, sys2, data, el, test_infl_on, rng)
        for idx2, ind in enumerate(indep_el):
            caus[idx2, idx1] = ind
        print("new causality matrix:")
        print(caus)
