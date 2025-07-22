import numpy as np

class RNN :
    def __init__(this, 
        max_timestep,
        input_size  = 1,
        hidden_size = 3, 
        output_size = 1
        ):
        this.max_timestep = max_timestep
        this.input_size = input_size
        this.hidden_size = hidden_size
        this.output_size = output_size

        this.wxh = np.random.uniform(-1,1,(hidden_size , input_size)) * 0.1
        this.whh = np.random.uniform(-1,1,(hidden_size , hidden_size)) * 0.1
        this.why = np.random.uniform(-1,1,(output_size , hidden_size)) * 0.1

        this.bxh = np.zeros((hidden_size, 1))
        this.by  = np.zeros((output_size, 1))

        this.h      = [np.zeros((hidden_size, 1)) for _ in range(max_timestep + 1)]
        this.h_row  = [np.zeros((hidden_size, 1)) for _ in range(max_timestep)]
        this.y_pred = [np.zeros((output_size, 1)) for _ in range(max_timestep)]
        this.loss   = np.zeros(max_timestep)


