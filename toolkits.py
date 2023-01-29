import numpy as np
import time

# Load evenly distributed weight vector
def load_weight(moead_object):
    # Load csv data
    file = moead_object.csv_name + '.csv'
    W = np.loadtxt(fname=file)
    moead_object.N = W.shape[0]
    moead_object.W = W
    return W

# Calculation initialization Pareto front
def init_pareto(moead_object):
    # print(moead_object.FV)
    for pi in range(moead_object.N):
        np = 0
        F_V_P = moead_object.FV[:, pi]
        for ppi in range(moead_object.N):
            F_V_PP = moead_object.FV[:, ppi]            
            if (pi != ppi) and (is_dominate(F_V_PP, F_V_P)):
                np += 1
        if np == 0:
            moead_object.EP_Pt.loc[pi] = F_V_P
    moead_object.EP_Pt.drop_duplicates(inplace=True)
    # print("EP_Pt is {}".format(moead_object.EP_Pt))

# Calculate T neighbors
def cal_B(moead_object):
    W = np.array(moead_object.W)
    neigh_num = moead_object.T
    for i, aw in enumerate(W):
        dis = np.linalg.norm((W-aw), axis=1)
        B_T = np.argsort(dis)
        B_T = B_T[1:neigh_num+1]
        moead_object.W_Bi_T[:, i] = B_T.reshape(neigh_num)
    moead_object.W_Bi_T.astype('int')

# Judge whether A dominates B
def is_dominate(A, B):
    return (np.all(A<=B) and np.any(A<B))

# Calculate the distance from individual X to reference point
def cal_dis_to_z(moead_object, X, Z):
    F_X = moead_object.Test_fun.obj_unc(X)
    dis = np.linalg.norm(F_X-Z)
    return dis 

# Calculate Tchebycheff distance of individual X
def cal_tche(w, x, z):
    return np.max(w*np.abs(x-z))

# Calculate Tchebycheff distance between individual X and reference point Z
def cal_tche_x_z(moead_object, id, X):
    weight_i = moead_object.W[id]
    F_X = moead_object.Test_fun.obj_func(moead_object, X)
    dis_tche = cal_tche(weight_i, F_X, moead_object.Z)
    return np.max(dis_tche)

# Update neighbor
def update_B(moead_object, P_B, Y):
    for j in P_B:
        # Take neighbor Xj
        j = int(j)
        Xj = moead_object.Pop[j, :]
        dis_xj = cal_tche_x_z(moead_object, j, Xj)
        dis_y = cal_tche_x_z(moead_object, j, Y)
        if dis_y <= dis_xj:
            moead_object.Pop[j, :] = Y
            F_Y = moead_object.Test_fun.obj_func(moead_object, Y)
            moead_object.FV[:, j] = F_Y
            # If the id exists, update the value of its corresponding target function set
            if id in moead_object.EP_Pt.index:
                moead_object.EP_Pt.loc[id] = F_Y

def update_EP_Y(moead_object, id_y):
    # Update frontier according to Y
    # Get id_ Function value of Y
    F_Y = moead_object.FV[:, id_y]
    # Collection to be deleted
    # Number of leading edge sets
    EPPT = moead_object.EP_Pt
    del_mid1 = EPPT[(EPPT.score1>F_Y[0])|(EPPT.score2>F_Y[1])].index.tolist()
    del_mid2 = EPPT[(EPPT.score1>=F_Y[0])&(EPPT.score2>=F_Y[1])].index.tolist()
    del_mid = set(del_mid1+del_mid2)
    EPPT.drop(del_mid, inplace=True)
    is_dominated1 = EPPT[(EPPT.score1<F_Y[0])|(EPPT.score2<F_Y[1])].index.tolist()
    is_dominated2 = EPPT[(EPPT.score1<=F_Y[0])&(EPPT.score2<=F_Y[1])].index.tolist()
    is_dominated = len(set(is_dominated1+is_dominated2))
    if is_dominated == 0:
        # Update EP. If the solution to be considered is not dominated by the solution in EP, add EP
        EPPT.loc[id_y]=F_Y
    moead_object.EP_Pt = EPPT
    return moead_object.EP_Pt

# Initialize reference point Z*
def init_Z(moead_object):
    moead_object.Z[0] = min(moead_object.FV[0])
    moead_object.Z[1] = min(moead_object.FV[1])
