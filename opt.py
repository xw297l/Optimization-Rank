import pandas as pd
import numpy as np
import time
from time import perf_counter
n = 192

df=pd.read_pickle(r'cvs_rcts.pkl').sort_values(by='normalized_cvr_scores', ascending=False)[:n]
cvr = df.set_index('sku').to_dict()['normalized_cvr_scores']
rcts =  df.set_index('sku').to_dict()['normalized_scores']

imp = pd.read_pickle(r'impression_weights.pkl').sort_values(by='pos', ascending=True)[:n]
new_col = 'normalized_impressions'
col = 'impressions'
imp[new_col] = (imp[col] - imp[col].min()+0.0001) / (imp[col].max() - imp[col].min())  
imp['pos'] = imp['pos']-1
impre = imp.set_index('pos').to_dict()['normalized_impressions']

cvr_bbsort = df.to_dict()['normalized_cvr_scores']
rcts_bbsort =  df.to_dict()['normalized_scores']

imp_weighted_rcts_by_bbsort = sum([rcts_bbsort[i]*impre[i] for i in range(n)]) 
imp_weighted_cvr_by_bbsort  = sum([cvr_bbsort[i]*impre[i] for i in range(n)])
alpha = 0.97
cvr_thold = alpha * imp_weighted_cvr_by_bbsort
print("cvr_thold",cvr_thold)
print("imp_weighted_rcts_by_bbsort",imp_weighted_rcts_by_bbsort)
print("imp_weighted_cvr_by_bbsort", imp_weighted_cvr_by_bbsort)

import numpy as np
from pyomo.environ import *

start = perf_counter()
model = ConcreteModel()
#Index Set I J, I for products, J for position
model.I = Set(initialize = df['sku'][:n].tolist())
model.J = Set(initialize = imp['pos'][:n].tolist())
e1 = perf_counter()

# Parameter:  cvr for ith product's CVR, rcts for ith product's RCTS, alpha for jth position's impression weight
model.cvr = Param(model.I,initialize=cvr)
model.rcts = Param(model.I,initialize=rcts)
model.impre = Param(model.J, initialize=impre)
e2 = perf_counter()

# Var: zij, whether place product i at position j
model.z = Var(model.I, model.J, within= Binary)
e3 = perf_counter()
# Objective: minimize total RCTS regarding products and positions
def obj_rule(model):
    return sum(sum(model.z[i,j]*model.rcts[i] for i in model.I)
               * model.impre[j]
               for j in model.J)
model.obj = Objective(rule = obj_rule, sense=minimize)
e4 = perf_counter()

#Constraints
    #c1: cvr sum above C

def c1_rule(model):
    return sum(sum(model.z[i,j]*model.cvr[i] for i in model.I)
               * model.impre[j]
               for j in model.J) >= cvr_thold # 1 as default, set to other threshold later
model.c1 = Constraint(rule = c1_rule)
    #c2: each position only 1 product
e5 = perf_counter()
def c2_rule(model, j):
    return sum([model.z[i,j] for i in model.I]) == 1
model.c2 = Constraint(model.J, rule = c2_rule)
e6 = perf_counter()
def c3_rule(model, i):
    return sum([model.z[i,j] for j in model.J]) == 1
model.c3 = Constraint(model.I, rule = c3_rule)
e7 = perf_counter()
#print("time before write", e1-start)
model.write('model.lp') 
e8 = perf_counter()
#print("time till model.write finished:", e2-start)
#model.pprint() 

solver = SolverFactory('gurobi') 
#solver.options["OptimalityTol"] = 0.01
#solver.options["MIPGap"] = 0.01
#solver.options["Threads"] = 96
#solver.options["Presolve"] = 1
#solver.options["TimeLimit"] = 100
print("time before solve", e9- start)

solution = solver.solve(model) 

solution.write() 

print("time till solved",e10 - start)

x_opt = np.array([value(model.z[i,j]) for i in model.I for j in model.J]).reshape((len(model.I), len(model.J))) # 提取最优解
obj_values = value(model.obj) # 提取目标函数
print("x_opt.shape",(x_opt.shape))
print("optimal objective: \n {}".format(obj_values))
a = list(cvr.values()) # bbsort val decreasing
b = list(impre.values()) # impre val 0, 1,2,...,n
c = np.dot(a, x_opt)
c1_val = np.dot(c,b)
print("c1_val at optimal objective: \n {}".format(c1_val))
print("c1_threshold: {}".format(cvr_thold))

print("optimum point: \n {} ".format(x_opt))
from numpy import savetxt
savetxt('./x_opt_alpha'+str(alpha)+'_n'+str(n)+'_cl167_geo1.csv', x_opt, delimiter=',')

print("imp_weighted_rcts_by_bbsort",imp_weighted_rcts_by_bbsort)
print("imp_weighted_cvr_by_bbsort", imp_weighted_cvr_by_bbsort)
print("imp_weighted_rcts_by_optrank: \n {}".format(obj_values))
print("imp_weighted_cvr_by_optrank: \n {}".format(c1_val))
