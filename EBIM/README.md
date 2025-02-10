# Codes for the Paper "Minimum Entropy Coupling with Bottleneck"

This repository contains the code and instructions for reproducing the results and figures from the paper titled "Minimum Entropy Coupling with Bottleneck".

## Prerequisites
- Python 3.12

## Setting Up the Environment
1. Create a virtual environment:  
   ```sh
   python -m venv myenv
   ```
2. Activate the virtual environment:  
   - On Windows: 
     ```sh
     myenv\Scripts\activate
     ```
   - On macOS and Linux: 
     ```sh
     source myenv/bin/activate
     ```
3. Install the required packages from `requirements.txt`:  
   ```sh
   pip install -r requirements.txt
   ```

## Generating Figures
The following Jupyter notebooks generate the figures related to EBIM as mentioned in the paper:
- `EBIM-fig2.ipynb` – Generates Figure 2
- `EBIM-fig9.ipynb` – Generates Figure 9
- `EBIM-fig10.ipynb` – Generates Figure 10

## Markov Coding Games
A Grid environment can be created and used to learn a Maximum Entropy policy. Below is an example of how to set up and run the grid environment:

### Setting Up the Grid Environment
```python
from gridworld import make_grid_world

grid_config = {
    'shape': (8, 8),
    'default_reward': 0,
    'goal': (0, -1),
    'goal_reward': 1,
    'trap': (1, -1),
    'trap_reward': -1,
    'obstacles': [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
    'noise': 0.1
}
gw = make_grid_world(grid_config)
```

### Learning a Maximum Entropy Policy Using Soft-Q Value Iteration
```python
import numpy as np

discount = 0.95
beta = np.exp(-6)

qpolicy, q_grids = gw.run_qvalue_iterations(iterations=100, discount=discount, beta=beta)
policy = qpolicy[..., -1]
```

### Running a Markov Coding Game with Rate Limit Episode
```python
# message
message = 1
message_size = 50
message_prior = np.ones(message_size) / message_size

compression_rate = 0.9
start = (5, 4)

states, actions, reward, beliefs, mutual_infos, joints = gw.run_message_conditional_policy(
    policy=policy,
    message=message,
    message_prior=message_prior,
    start=start,
    max_steps=1_000, 
    discount=discount,
    compression_rate=compression_rate,
)
```

For a complete example, see the `gridworld.ipynb` notebook.

## Acknowledgments
The gridworld environment is forked from [kevin-hanselman/grid-world-rl](https://github.com/kevin-hanselman/grid-world-rl).
