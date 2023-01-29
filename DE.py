import toolkits as tk
import numpy as np
from sklearn.model_selection import train_test_split 

'''
Differential evolution algorithm
'''

# Create a population
def Creat_Pop(moead_object):
    Pop = np.zeros((moead_object.N, moead_object.Test_fun.var_dim), dtype='int')
    FV = np.zeros((moead_object.obj_dim, moead_object.N), dtype='float')
    bound = moead_object.Test_fun.sol_bound
    Pop += bound[:, 0]
    Pop += ((bound[:, 1] - bound[:, 0])*np.random.rand(moead_object.N, moead_object.Test_fun.var_dim)).astype('int')
    moead_object.X_train, moead_object.X_test, moead_object.y_train, moead_object.y_test = train_test_split(moead_object.data, moead_object.target, test_size=0.2)
    for i in range(moead_object.N):
        obj_value = moead_object.Test_fun.obj_func(moead_object, Pop[i, :])
        FV[:, i] = obj_value
        moead_object.tabu.append(obj_value)
    moead_object.Pop, moead_object.FV = Pop, FV
    # print("fv is {}".format(FV))
    return Pop, FV

# Crossover rate
Cross_Rate = 0.5

# mutation
def mutate(moead_object, best, p1, p2):
    bound = moead_object.Test_fun.sol_bound
    f = 5 + 10 * np.random.rand()  # Scale factor
    d = f * (p1 - p2)
    temp_p = best + d
    temp_p = np.clip(temp_p, bound[:, 0], bound[:, 1])
    return temp_p

# cross
def crossover(moead_object, p1, vi):
    var_num = moead_object.Test_fun.var_dim
    bound = moead_object.Test_fun.sol_bound
    ui = np.zeros(var_num)
    k = np.random.random_integers(0, var_num - 1)
    for j in range(0, var_num):
        if np.random.random() < Cross_Rate or j == k:
            ui[j] = vi[j]
        else:
            ui[j] = p1[j]
    ui = np.clip(ui, bound[:, 0], bound[:, 1])
    return ui

# Generate next generation
def generate_next(moead_object, wi, p0, p1, p2):
    tche_p0 = tk.cal_tche_x_z(moead_object, wi, p0)
    tche_p1 = tk.cal_tche_x_z(moead_object, wi, p1)
    tche_p2 = tk.cal_tche_x_z(moead_object, wi, p2)
    arr = [p0, p1, p2]
    tche = np.array([tche_p0, tche_p1, tche_p2])
    index = np.argsort(tche)
    best = arr[index[0]]
    bw = arr[index[2]]
    bm = arr[index[1]]

    vi = mutate(moead_object, best, bm, bw)
    ui = crossover(moead_object, p0, vi)
    return ui


def envolution(moead_object):
    for gen in range(moead_object.max_gen):
        moead_object.X_train, moead_object.X_test, moead_object.y_train, moead_object.y_test = train_test_split(moead_object.data, moead_object.target, test_size=0.2)
        # moead.gen = gen
        for pi, p in enumerate(moead_object.Pop):
            # Neighbor set of the pi-th individual
            Bi = moead_object.W_Bi_T[:, pi]
            k = np.random.randint(moead_object.T)
            l = np.random.randint(moead_object.T)
            # Randomly select 2 individuals from the neighborhood to generate new solutions
            ik = Bi[k]
            il = Bi[l]
            Xi = moead_object.Pop[pi]
            Xk = moead_object.Pop[ik]
            Xl = moead_object.Pop[il]
            # Generate next generation
            Y = generate_next(moead_object, pi, Xi, Xk, Xl)
            tche_i = tk.cal_tche_x_z(moead_object, pi, Xi)
            tche_y = tk.cal_tche_x_z(moead_object, pi, Y)
            if tche_y < tche_i:
                moead_object.Pop[pi] = np.copy(Y)
                F_Y = moead_object.Test_fun.obj_func(moead_object, Y)
                tk.update_EP_ID(moead_object, pi, F_Y)
                tk.update_Z(moead_object, F_Y)
                tk.update_EP_Y(moead_object, pi)
            tk.update_B(moead_object, Bi, Y)
        print('gen %s,EP size :%s, Z is %s' % (gen, len(moead_object.EP_Pt_ID), moead_object.Z))
    return moead_object.EP_Pt_ID
