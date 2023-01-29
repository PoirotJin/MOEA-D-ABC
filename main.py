import GA
import numpy as np
import pandas as pd
import toolkits as tk
import time 
import clf


class MOEAD(object):
    # test function
    Test_fun = clf 
    # use GA? or DE?
    evol_way = GA 

    # data
    X_train = None 
    y_train = None 
    X_test = None 
    y_test = None 

    # The population size depends on the number of weight vectors
    N = 101
    # Number of objective functions
    obj_dim = 2
    # Maximum Iterations
    max_gen = 30
    # Neighbor setting (only updates and crosses the neighbors)
    T = 5
    # Leading edge ID
    EP_Pt_ID = []
    # The function value of the leading edge
    EP_Pt_FV = []
    EP_Pt = pd.DataFrame(columns=['score1','score2'])

    # population
    # Dimension: N * D
    Pop = None
    # The function value calculated by each individual of the population
    # data
    data = None
    target = None
    # Dimension: obj_ dim*N
    FV = None
    FX = None
    FY = None
    # weight
    W = []
    # T neighbors of the weight
    W_Bi_T = np.zeros((T, N), dtype='int')
    # The best point that can be reached in theory
    Z = np.zeros(obj_dim)
    # Weight vector storage directory 
    # (weight vector is fixed, so it is stored in advance)
    csv_name = 'test1'

    # Run time point
    delta_time = None

    def __init__(self):
        self.init_data()

    def init_data(self):
        # load data
        clf.load_data(self)
        # load weight
        tk.load_weight(self)
        # Calculate the T neighbors of each weight Wi
        tk.cal_B(self)
        # create population
        self.evol_way.Creat_Pop(self)
        tk.init_pareto(self)

    def run(self):
        t = time.time()
        # EP_ X_ ID: the ID of the leading individual solution. The sequence number in the above array: Pop
        # Evolution
        EP_Pt = self.evol_way.envolution(self)
        print("Dominating frontier solutions:", EP_Pt)
        dt = time.time() - t
        print(dt)

if __name__ == "__main__":
    moead = MOEAD()
    moead.run()