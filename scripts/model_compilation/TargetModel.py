import numpy as np
from numba.pycc import CC

cc = CC('model_func')

@cc.export("model_func", "float64[:](float64,float64[:],float64[:])")
def model_function(time, S, k):

    P = np.zeros(len(S))

    F_in = np.array(
    [[2.233610701,2.238605815],
     [0.150000000,0.150000000],
     [0.000000000,0.000000000],
     [0.000000000,0.000000000]]    )

    flow_time = np.array(
    [0.000000000,2.000000000]    )

    total_flow = np.array(
    [29.616805350,29.619302907]    )

    i = np.abs(flow_time - time).argmin()

    P[0] = +np.interp(time,F_in[0],flow_time)-k[0]*S[0]*S[1]-k[1]*S[0]*S[3]-S[0]*np.interp(time,total_flow,flow_time)
    P[1] = +k[1]*S[0]*S[3]-k[0]*S[0]*S[1]-k[2]*S[4]*S[1]-k[3]*S[4]*S[1]-k[4]*S[7]*S[1]-S[1]*np.interp(time,total_flow,flow_time)
    P[2] = +k[0]*S[0]*S[1]-S[2]*np.interp(time,total_flow,flow_time)
    P[3] = +k[1]*S[0]*S[3]+np.interp(time,F_in[1],flow_time)-k[1]*S[0]*S[3]-S[3]*np.interp(time,total_flow,flow_time)
    P[4] = +k[4]*S[7]*S[1]-k[2]*S[4]*S[1]-k[3]*S[4]*S[1]-S[4]*np.interp(time,total_flow,flow_time)
    P[5] = +k[2]*S[4]*S[1]-S[5]*np.interp(time,total_flow,flow_time)
    P[6] = +k[3]*S[4]*S[1]-S[6]*np.interp(time,total_flow,flow_time)
    P[7] = +k[4]*S[7]*S[1]+F_in[2,i]-k[4]*S[7]*S[1]-S[7]*np.interp(time,total_flow,flow_time)
    P *= time

    return P

def wrapper_function(time, S, k):
    return model_function(time, S, k)

species = {
'O=C(CO)CO':0,
'OC=C(O)CO':1,
'O=C[C@@](O)(CO)C(O)(CO)CO':2,
'[OH-]':3,
'O=C[C@H](O)CO':4,
'O=C(CO)[C@H](O)[C@H](O)[C@H](O)CO':5,
'O=C[C@@](O)(CO)[C@@H](O)[C@H](O)CO':6,
'O':7,
}

reactions = {
'O=C(CO)CO.OC=C(O)CO>>O=C[C@@](O)(CO)C(O)(CO)CO':0,
'O=C(CO)CO.[OH-]>>OC=C(O)CO.[OH-]':1,
'O=C[C@H](O)CO.OC=C(O)CO>>O=C(CO)[C@H](O)[C@H](O)[C@H](O)CO':2,
'O=C[C@H](O)CO.OC=C(O)CO>>O=C[C@@](O)(CO)[C@@H](O)[C@H](O)CO':3,
'O.OC=C(O)CO>>O.O=C[C@H](O)CO':4,
}

inputs = {
'O=C(CO)CO':2.0,
'[OH-]':0.12,
'O':0.0,
}

k = np.zeros(max(reactions.values())+1) # rate constants

S = np.zeros(len(species)) # initial concentrations

if __name__ == "__main__":
    cc.compile()
