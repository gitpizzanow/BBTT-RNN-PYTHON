import numpy as np
from Structure import RNN
from functions import CLIP , Loss , mean_squared_error, mean_absolute_error, regression_accuracy



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
    for e in range(300):
        forward(
            rnn , X,y,max_timestep
        )
        
        grads = backward(
            rnn , X,y,max_timestep
        )

        apply_gradients(rnn,grads,0.005)

    # Evaluate
    mse = mean_squared_error(y, rnn.y_pred)
    mae = mean_absolute_error(y, rnn.y_pred)
    acc = regression_accuracy(y, rnn.y_pred, tolerance=0.3)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Regression Accuracy (±0.1): {acc*100:.2f}%") # tolerance margin

    # MSE: 0.0373 ->  predictions are close on average
    # MAE: 0.1927 -> average error 
    # Regression Accuracy (±0.1): 100.0%

    
    """
    X_new = np.array([
        [3],
        [4]
    ])
    y_dummy = np.zeros_like(X_new) 
    forward(rnn, X_new, y_dummy, max_timestep)
    
    for t,pred in enumerate(rnn.y_pred):
        print(f"Prediction at t={t}: {pred.flatten()[0]:.4f}")
    """
