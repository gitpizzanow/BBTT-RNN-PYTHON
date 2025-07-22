import numpy as np
from Structure import RNN
from functions import CLIP , Loss



def forward(rnn : RNN ,
            X : np.ndarray ,
            y : np.ndarray,
            max_timestep:int):

    for t in range(max_timestep):
        x = X[t].reshape(-1,1)
        rnn.h_row[t] = rnn.wxh @ x + rnn.whh @ rnn.h[t] + rnn.bxh
        rnn.h[t+1] = np.tanh(rnn.h_row[t])
        rnn.y_pred[t] = rnn.why @ rnn.h[t+1] + rnn.by
        
        rnn.loss[t] = Loss(y[t], rnn.y_pred[t])

        



if __name__ == '__main__':
    max_timestep : int = 2
    rnn = RNN(max_timestep)
    X = np.array([
    [1],   
    [2]    ])  

    y = np.array([
        [2],   
        [3]    ]) 
    
    
    forward(
        rnn , X,y,max_timestep
    )


    print(rnn.y_pred) # [array([[-0.00501238]]), array([[-0.0100752]])]
  
