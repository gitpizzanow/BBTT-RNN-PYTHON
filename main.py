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


def backward(rnn : RNN ,
            X : np.ndarray ,
            y : np.ndarray,
            t:int):

    # dwxh , dwhh , dwhy , dbxh , dby , dh_next
    
    dwxh    = np.zeros((rnn.hidden_size, rnn.input_size))
    dwhh    = np.zeros((rnn.hidden_size, rnn.hidden_size))
    dwhy    = np.zeros((rnn.output_size, rnn.hidden_size))
    dbxh    = np.zeros((rnn.hidden_size, 1))
    dby     = np.zeros((rnn.output_size, 1))
    dh_next = np.zeros_like(rnn.h[0]).reshape(-1,1)

    for t in range(t-1,-1,-1):
        y_pred = rnn.y_pred[t]
        dy = 2 * (y_pred - y[t].reshape(-1,1))
        
        dwhy += dy @ rnn.h[t+1].T
        dby  += dy

        dh  = rnn.why.T @ dy + dh_next
        dh_raw = dh * (1 - np.tanh(rnn.h_row[t])**2)   

        dbxh += dh_raw 
        dwhh += dh_raw @ rnn.h[t].T
        dwxh += dh_raw @ X[t].reshape(1, -1)

        dh_next = rnn.whh.T @ dh_raw

    return {
        "dwxh": CLIP(dwxh),
        "dwhh": CLIP(dwhh),
        "dwhy": CLIP(dwhy),
        "dbxh": CLIP(dbxh),
        "dby":  CLIP(dby)
    }


def apply_gradients(rnn: RNN, grads: dict, lr: float):
    
    for k in grads:
        grads[k] = CLIP(grads[k])

    
    rnn.wxh -= lr * grads["dwxh"]
    rnn.whh -= lr * grads["dwhh"]
    rnn.why -= lr * grads["dwhy"]
    rnn.bxh -= lr * grads["dbxh"]
    rnn.by  -= lr * grads["dby"]

         


        
        



if __name__ == '__main__':
   
    X = np.array([
    [1],   
    [2]    ])  

    y = np.array([
        [2],   
        [3]    ]) 
    
    
    max_timestep : int = X.size
    rnn = RNN(max_timestep)

    #Train 
    for e in range(100):
        forward(
            rnn , X,y,max_timestep
        )
        print(f'loss {e+1 }= {np.mean(rnn.loss)} \n')
        grads = backward(
            rnn , X,y,max_timestep
        )

        apply_gradients(rnn,grads,0.005)


        
