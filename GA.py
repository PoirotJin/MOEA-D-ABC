import toolkits as tk
import numpy as np
from sklearn.model_selection import train_test_split

'''
genetic algorithm
'''
# create population
def Creat_Pop(moead_object):
    Pop = np.random.randint(low=0, high=2, size=(moead_object.N, moead_object.Test_fun.var_dim), dtype='int')
    FV = np.zeros((moead_object.obj_dim, moead_object.N), dtype='float')
    moead_object.X_train, moead_object.X_test, moead_object.y_train, moead_object.y_test = train_test_split(moead_object.data, moead_object.target.ravel(), test_size=0.1, random_state=0)
    for i in range(moead_object.N):
        obj_value = moead_object.Test_fun.obj_func(moead_object, Pop[i, :])
        FV[:, i] = obj_value
    moead_object.Pop, moead_object.FV = Pop, FV
    return Pop, FV

# Mutation mode
def mutate(moead_object, p):
    var_dim = moead_object.Test_fun.var_dim
    rand = int(var_dim * np.random.rand())
    p[rand] = bool(1-p[rand])
    return p  

# Crossing mode 1
def crossover(moead_object, pop1, pop2):
    var_dim = moead_object.Test_fun.var_dim
    rand = int(var_dim * np.random.rand())
    if np.random.rand() < 0.5:
        pop1[:rand], pop2[:rand] = pop2[:rand], pop1[:rand]
    else:
        pop1[rand:], pop2[rand:] = pop2[rand:], pop2[rand:]
    return pop1, pop2 

# Cross mutation
def cross_mutation(moead_object, p1, p2):
    y1 = np.copy(p1)
    y2 = np.copy(p2)
    c_rate = 1
    m_rate = 0.5
    if np.random.rand() < m_rate:
        y1 = mutate(moead_object, y1)
        y2 = mutate(moead_object, y2)
    if np.random.rand() < c_rate:
        y1, y2 = crossover(moead_object, y1, y2)
    # Randomly specify y1, y2
    rand = np.random.rand()
    if rand < 0.5:
        return y1 
    else:
        return y2

def generate_next(moead_object, xl, xk):
    # Cross variation
    y = cross_mutation(moead_object, xl, xk)
    # Select the individual with the smallest (best) Tchebycheff distance
    # More detailed judgment with a probability of 0.5
    FY2 = moead_object.Test_fun.obj_func(moead_object, y)
    moead_object.FY = FY2
    return y

def stop_criteria(moead_object):
    if np.any((np.array(moead_object.EP_Pt.score1)<0.1).any() and (np.array(moead_object.EP_Pt.score2)<0.1).any()):
        return False
    else:
        return True

def envolution(moead_object):
    # Evolution, start to evolve moead.max_ Gen times
    for gen in range(1):
        if stop_criteria(moead_object):
            # Each iteration generates a set of test set training sets
            moead_object.X_train, moead_object.X_test, moead_object.y_train, moead_object.y_test = train_test_split(moead_object.data, moead_object.target.ravel(), test_size=0.1)
            print("EPPT {} at the beginning of evolution".format(moead_object.EP_Pt))
            print("FV {} at the beginning of evolution".format(moead_object.FV))
            EPPT = moead_object.EP_Pt
            # Start to evolve each individual in the array moead.Pop
            for pi in range(moead_object.N-90):
                # Neighbor set of individual No. pi
                Bi = moead_object.W_Bi_T[:, pi]
                # Randomly select a number in T as the neighbor of pi
                k = np.random.randint(moead_object.T)
                l = np.random.randint(moead_object.T)
                # Randomly select 2 individuals from the neighborhood to generate new solutions
                ik = Bi[k]
                il = Bi[l]
                Xi = moead_object.Pop[pi, :]
                Xk = moead_object.Pop[ik, :]
                Xl = moead_object.Pop[il, :]
                # Evolve the next generation of individuals. Based on the two Xk randomly selected from their own Xi+neighbors, Xl also considers "gen" to evolve the next generation
                Y = generate_next(moead_object, Xk, Xl)
                # Calculate the current Xi, the Tchebycheff distance before evolution
                weight_i = moead_object.W[pi]
                F_X = moead_object.Test_fun.obj_func(moead_object, Xi)
                tche_xi = tk.cal_tche(weight_i, F_X, moead_object.Z)
                # Calculate the current Xi, the evolved Tchebycheff distance
                F_Y = moead_object.Test_fun.obj_func(moead_object, Y)
                tche_y = tk.cal_tche(weight_i, F_Y, moead_object.Z)
                # Start to compare whether a better next generation has evolved, so as to retain
                if tche_y <= tche_xi:
                    print("++++++++++++++++++++++++++")
                    print("F_Y{}".format(F_Y))
                    # Update frontier according to Y
                    moead_object.FV[:, pi] = F_Y
                    # Collection to be deleted
                    del_mid1 = EPPT[(EPPT.score1>F_Y[0])|(EPPT.score2>F_Y[1])].index.tolist()
                    del_mid2 = EPPT[(EPPT.score1>=F_Y[0])&(EPPT.score2>=F_Y[1])].index.tolist()
                    del_mid = set(del_mid1+del_mid2)
                    print("del_mid is {}".format(del_mid))
                    print("EPPT {}".format(EPPT))
                    EPPT.drop(del_mid, inplace=True)
                    print("EPPT {}".format(EPPT))
                    is_dominated1 = EPPT[(EPPT.score1<F_Y[0])|(EPPT.score2<F_Y[1])].index.tolist()
                    is_dominated2 = EPPT[(EPPT.score1<=F_Y[0])&(EPPT.score2<=F_Y[1])].index.tolist()
                    is_dominated = len(set(is_dominated1+is_dominated2))
                    if is_dominated == 0:
                        # Update EP. If the solution to be considered is not dominated by the solution in EP, add EP
                        # print("pi1 is".format(pi))
                        EPPT.loc[pi]=F_Y
                        # print("EPPT is{}".format(EPPT))
                    # 100 rows and 90 columns
                    moead_object.Pop[pi, :] = Y
                print("EPPT {} after Y pair Xi update".format(EPPT))
                for j in Bi:
                    # Take neighbor Xj
                    Xj = moead_object.Pop[j, :]
                    weight_i = moead_object.W[j]
                    F_Xj = moead_object.Test_fun.obj_func(moead_object, Xj)
                    tche_xj = tk.cal_tche(weight_i, F_Xj, moead_object.Z)
                    print("xxxxxxx{}".format(EPPT))
                    if tche_y <= tche_xj:
                        moead_object.FV[:, j] = F_Y
                        moead_object.Pop[j, :] = Y
                        # If the id exists, update the value of its corresponding target function set
                        EPPT.loc[j] = F_Y
                    print("EPPT {} after updating neighbor".format(EPPT))
        else:
            break
        moead_object.EP_Pt = EPPT
        print("FV {}, EPPT {}".format(moead_object.FV, moead_object.EP_Pt))
        print('Iteration {}, dominate the number of frontier individuals len (moead_object. EP_Pt): {}'.format(gen,moead_object.EP_Pt))
    return moead_object.EP_Pt
