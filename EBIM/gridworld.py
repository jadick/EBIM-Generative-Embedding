import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cv2
from scipy.special import logsumexp, softmax
from scipy.stats import entropy
from tqdm.auto import trange
from coupling import min_ent_coupling, bottlenecked_min_ent_coupling



def max_lse(arr, b, axis=None):
    if b == 0:
        return np.max(arr, axis=axis)
    else:
        return b * logsumexp(arr / b, axis=axis)
    
def sample_position(prob_array):
    # Flatten the array
    flattened_array = prob_array.flatten()

    # Sample an index from the flattened array
    sampled_index = np.random.choice(len(flattened_array), p=flattened_array)

    # Map the index back to 2D coordinates
    row, col = np.unravel_index(sampled_index, prob_array.shape)

    return row, col


class GridWorldMDP:

    # up, right, down, left
    _direction_deltas = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]
    _num_actions = len(_direction_deltas)

    def __init__(self,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 action_probabilities,
                 no_action_probability):

        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._obstacle_mask = obstacle_mask
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability,
            obstacle_mask
        )

    @property
    def shape(self):
        return self._reward_grid.shape

    @property
    def size(self):
        return self._reward_grid.size

    @property
    def reward_grid(self):
        return self._reward_grid

    def run_value_iterations(self, discount=1.0,
                             iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

        utility_grid = np.zeros_like(self._reward_grid)
        for i in range(iterations):
            utility_grid = self._value_iteration(
                utility_grid=utility_grid,
                discount=discount
            )
            policy_grids[:, :, i] = self.best_policy(utility_grid)
            utility_grids[:, :, i] = utility_grid
        return policy_grids, utility_grids

    def run_qvalue_iterations(self, discount=1.0, iterations=10, beta=0):
        q_grids = np.zeros((*self.shape, self._num_actions, iterations))
        policy_grids = np.zeros_like(q_grids)
        
        q_grid = np.zeros_like(q_grids[..., 0])
        
        for i in trange(iterations):
            q_grid = self._qvalue_iteration(
                q_grid=q_grid,
                discount=discount,
                beta=beta
            )
            policy_grids[..., i] = self.best_soft_policy(q_grid, beta)
            q_grids[..., i] = q_grid
            
        return policy_grids, q_grids
            
    def _qvalue_iteration(self, q_grid, discount=1.0, beta=0):
        out = np.zeros_like(q_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                for a in range(self._num_actions):
                    out[i, j, a] = self._calculate_qval(
                        (i, j), a, discount, q_grid, beta
                    )
        return out
    
    def _calculate_qval(self, loc, action, discount, q_grid, beta):
        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        
        row, col = loc
        return ( 
            discount * np.sum(np.sum(
                self._T[row, col, action, :, :] * max_lse(q_grid, axis=-1, b=beta),
            axis=-1), axis=-1)
        ) + self._reward_grid[loc]
        
    def run_policy_iterations(self, discount=1.0,
                              iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

        policy_grid = np.random.randint(0, self._num_actions,
                                        self.shape)
        utility_grid = self._reward_grid.copy()

        for i in range(iterations):
            policy_grid, utility_grid = self._policy_iteration(
                policy_grid=policy_grid,
                utility_grid=utility_grid,
                discount=discount
            )
            policy_grids[:, :, i] = policy_grid
            utility_grids[:, :, i] = utility_grid
        return policy_grids, utility_grids

    def generate_experience(self, current_state_idx, action_idx):
        sr, sc = self.grid_indices_to_coordinates(current_state_idx)
        next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()

        next_state_idx = np.random.choice(np.arange(next_state_probs.size),
                                          p=next_state_probs)

        return (next_state_idx,
                self._reward_grid.flatten()[next_state_idx],
                self._terminal_mask.flatten()[next_state_idx])

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        # Annoyingly, this doesn't work for negative indices.
        # The mode='wrap' parameter only works on positive indices.
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)

    def best_policy(self, utility_grid):
        M, N = self.shape
        return np.argmax((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                         .sum(axis=-1).sum(axis=-1), axis=2)
        
    def best_soft_policy(self, q_grid, beta=0):
        beta = beta or 1e-10
        return softmax(q_grid / (beta), axis=2)

    def _init_utility_policy_storage(self, depth):
        M, N = self.shape
        utility_grids = np.zeros((M, N, depth))
        policy_grids = np.zeros_like(utility_grids)
        return utility_grids, policy_grids

    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability,
                                  obstacle_mask):
        M, N = self.shape

        T = np.zeros((M, N, self._num_actions, M, N))

        r0, c0 = self.grid_indices_to_coordinates()

        T[r0, c0, :, r0, c0] += no_action_probability

        for action in range(self._num_actions):
            for offset, P in action_probabilities:
                direction = (action + offset) % self._num_actions

                dr, dc = self._direction_deltas[direction]
                r1 = np.clip(r0 + dr, 0, M - 1)
                c1 = np.clip(c0 + dc, 0, N - 1)

                temp_mask = obstacle_mask[r1, c1].flatten()
                r1[temp_mask] = r0[temp_mask]
                c1[temp_mask] = c0[temp_mask]

                T[r0, c0, action, r1, c1] += P

        terminal_locs = np.where(self._terminal_mask.flatten())[0]
        T[r0[terminal_locs], c0[terminal_locs], :, :, :] = 0
        return T

    def _value_iteration(self, utility_grid, discount=1.0):
        out = np.zeros_like(utility_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calculate_utility((i, j),
                                                    discount,
                                                    utility_grid)
        return out

    def _policy_iteration(self, *, utility_grid,
                          policy_grid, discount=1.0):
        r, c = self.grid_indices_to_coordinates()

        M, N = self.shape

        utility_grid = (
            self._reward_grid +
            discount * ((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                        .sum(axis=-1).sum(axis=-1))[r, c, policy_grid.flatten()]
            .reshape(self.shape)
        )

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]

        return self.best_policy(utility_grid), utility_grid

    def _calculate_utility(self, loc, discount, utility_grid):
        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * utility_grid,
                       axis=-1),
                axis=-1)
        ) + self._reward_grid[loc]

    def plot_policy(self, utility_grid, policy_grid=None):
        if policy_grid is None:
            policy_grid = self.best_policy(utility_grid)
        markers = "^>v<"
        marker_size = 200 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'w'

        no_action_mask = self._terminal_mask | self._obstacle_mask

        utility_normalized = (utility_grid - utility_grid.min()) / \
                             (utility_grid.max() - utility_grid.min())

        utility_normalized = (255*utility_normalized).astype(np.uint8)

        utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
        for i in range(3):
            channel = utility_rgb[:, :, i]
            channel[self._obstacle_mask] = 0

        plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')

        for i, marker in enumerate(markers):
            y, x = np.where((policy_grid == i) & np.logical_not(no_action_mask))
            plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

        y, x = np.where(self._terminal_mask)
        plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                 color=marker_fill_color)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy_grid.shape)/8
        best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[best_option]
        plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy_grid.shape[0]-0.5])
        plt.xlim([-0.5, policy_grid.shape[1]-0.5])

    def plot_policy2(self, policy, states=None, actions=None, start=None, ax=None):
        if ax is None:
            figsize = np.array(self.shape)[::-1] * 2
            fig, ax = plt.subplots(figsize=figsize)
            # ax.axis('off')
        else:
            fig = ax.get_figure()
        
        grid = self._reward_grid.copy()
        grid[self._obstacle_mask] = np.nan
        
        from matplotlib.colors import LinearSegmentedColormap
        # Define the colors for specific values
        colors = ["darkred", "white", 'darkgreen']  # Colors at specified values
        values = [0, 0.5, 1]  # The values corresponding to the colors

        # Create a colormap
        # cmap = LinearSegmentedColormap.from_list("custom_map", list(zip(values, colors)))
        cmap = mpl.colormaps['PiYG']
        # cmap = mpl.colormaps['RdBu']
        cmap.set_bad(color='darkgrey')
        
        sns.heatmap(grid, ax=ax, cbar=False, square=True, linewidth=1, linecolor='k', cmap=cmap)
        
        # "^>v<"
        offsets = np.array([
            [0, -1],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])
        offsets_adjusted = offsets * .25 + .5
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if grid[i, j] == 0:
                    probs = policy[i, j]
                    
                    if probs.ndim == 0:
                        p = np.zeros((self._num_actions))
                        p[int(probs)] = 1
                        probs = p
                        
                    for a, prob in enumerate(probs):
                        pos = np.array([j, i]) + offsets_adjusted[a]
                        prob_str = f'{prob:.2f}'
                        # prob_str = '0' if float(prob_str) == 0 else prob_str
                        t = ax.text(*pos, prob_str, ha='center', va='center')
                        c = cmap(prob/2+0.5)
                        t.set_bbox(dict(facecolor=c, alpha=.6, edgecolor='gray', boxstyle='round,pad=0.1'))

        
        # plot trajectory
        if states is not None:
            states = np.array(states) + 0.5
            ax.plot(states[:, 1], states[:, 0], 'ko-', lw=2)
        
            if actions is not None:
                # labels = '⬆⇒⬇⬅'
                for s, a in zip(states, actions):
                    dx, dy = offsets[a] * .5
                    ax.arrow(x=s[1], y=s[0], dx=dx, dy=dy, head_width=.08, 
                             overhang=0.3, color='k', length_includes_head=True)
                    # ax.text(s[1], s[0], labels[a], size=20)
        
        # plot starting cell
        if start is not None:
            start = np.array(start) + .5
            # ax.plot(start[1]+.35, start[0]+.35, 'ro', ms=20, zorder=100)
            t = ax.text(start[1]+.35, start[0]+.35, 'S', color='w', ha='center', va='center', zorder=100)
            t.set_bbox(dict(facecolor='r', edgecolor='w', boxstyle='circle,pad=0.2'))

        
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        return fig, ax
        
                    
    def run_policy(self, policy, start, discount=1, max_steps=10_000):
        states, actions = [start], []
        
        state = start
        reward = 0
        
        for i in range(max_steps):
            # sample action from policy
            a = np.random.choice(self._num_actions, p=policy[state])
            
            # sample next state from transition prob
            state = sample_position(self._T[(*state, a)])
            
            # get reward of the state
            reward += discount**(i+1) * self.reward_grid[state]
            
            actions.append(a)
            states.append(state)
            
            if self._reward_grid[state]:
                break
        
        return states, actions, reward
    
    
    def run_message_conditional_policy(self, policy, message, message_prior, start,
                                       max_steps=10_000, discount=1, 
                                       compression_rate=1, comp_method='search_function',
                                       ):

        states, actions, believes, mutual_infos, joints = [start], [], [message_prior], [], []
        
        state = start
        belief = message_prior
        reward = 0
        
        # compress actions instead
        if comp_method.endswith('_reverse'):
            reverse = True
            comp_method = comp_method.split('_reverse')[0]
        else:
            reverse = False
            
        
        if compression_rate == 1:
            coupling_fn = min_ent_coupling
        else:
            H_M = entropy(belief, base=2)
            R = H_M * compression_rate
            
            if reverse:
                # compress Px (policy)
                coupling_fn = lambda Px, Py: bottlenecked_min_ent_coupling(Px, Py, R, comp_method)
            else:
                # compress Py (message)
                coupling_fn = lambda Px, Py: bottlenecked_min_ent_coupling(Py, Px, R, comp_method).T
                
        
        
        for i in range(max_steps):
            
            # find message conditional policy
            joint_policy_message = coupling_fn(policy[state], belief)
            joints.append(joint_policy_message)
            
            message_conditional_policy = joint_policy_message[:, message]
            message_conditional_policy = message_conditional_policy / message_conditional_policy.sum()
            
            mi = entropy(policy[state], base=2) + entropy(belief, base=2) - entropy(joint_policy_message.ravel(), base=2)
            mutual_infos.append(mi)
            
            # sample action from message conditional policy
            a = np.random.choice(self._num_actions, p=message_conditional_policy)
            actions.append(a)
            # reward += beta * entropy(message_conditional_policy)
            
            # sample next state from transition prob
            state = sample_position(self._T[(*state, a)])
            states.append(state)
            
            # update belief
            belief = joint_policy_message[a, :]
            belief = belief / belief.sum()
            believes.append(belief)
            
            # get reward of the state
            reward += discount**(i+1) * self.reward_grid[state]
            
            if self._terminal_mask[state]:
                break
        return states, actions, reward, np.array(believes), mutual_infos, joints
        

    def run_blockwise_message_conditional_policy(self, policy, messages, messages_prior,
                                                 start, max_steps=10_000, discount=1,
                                                 compression_rate=1):
        
        if compression_rate == 1:
            coupling_fn = min_ent_coupling
        else:
            H_M = entropy(messages_prior, base=2)
            R = H_M * compression_rate
            coupling_fn = lambda Px, Py: bottlenecked_min_ent_coupling(Px, Py, R)
        
        
        states, actions, believes = [start], [], [messages_prior]
        
        state = start
        belief = messages_prior
        reward = 0
        
        
        for i in (pbar:=trange(max_steps)):
            # find message block with highest entropy
            block = np.argmax(entropy(belief, axis=1))
            pbar.set_description(f'active block = {block}')
            
            # find message conditional policy
            joint_policy_message = coupling_fn(policy[state], belief[block])
            message_conditional_policy = joint_policy_message[:, messages[block]]
            message_conditional_policy = message_conditional_policy / message_conditional_policy.sum()
            
            # sample action from message conditional policy
            a = np.random.choice(self._num_actions, p=message_conditional_policy)
            # reward += beta * entropy(message_conditional_policy)
            
            # sample next state from transition prob
            state = sample_position(self._T[(*state, a)])
            
            # update belief
            belief[block] = joint_policy_message[a, :]
            belief[block] = belief[block] / belief[block].sum()
            believes.append(belief)
            
            # get reward of the state
            reward += discount**(i+1) * self.reward_grid[state]
            
            actions.append(a)
            states.append(state)
            
            if self._terminal_mask[state]:
                break
        
        return states, actions, reward, np.array(believes)
    
    
def make_grid_world(config):
    # reward grid
    reward_grid = np.zeros(config['shape']) + config['default_reward']
    reward_grid[config['goal']] = config['goal_reward']
    reward_grid[config['trap']] = config['trap_reward']
    for obs in config['obstacles']:
        reward_grid[obs] = 0

    # terminal mask
    terminal_mask = np.zeros_like(reward_grid, dtype=bool)
    terminal_mask[config['goal']] = True
    terminal_mask[config['trap']] = True

    # obstacle mask
    obstacle_mask = np.zeros_like(reward_grid, dtype=bool)
    for obs in config['obstacles']:
        obstacle_mask[obs] = True

    # make grid
    return GridWorldMDP(
        reward_grid=reward_grid,
        obstacle_mask=obstacle_mask,
        terminal_mask=terminal_mask,
        action_probabilities=[
            (-1, config['noise']),
            (0, 1 - 2 * config['noise']),
            (1, config['noise']),
        ],
        no_action_probability=0.0
    )