import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Définition de l'environnement CUBE
class CubeEnvironment:
    # Initialisation
    # Start = position de départ de l'agent = origine du cube
    # Goal: position de l'objectif, extrémité de du cube
    def __init__(self, size):
        self.size = size
        self.obstacles = set()
        for i in range(20): # Génération de 20 obstacles
            obstacle = np.random.randint(1, size, 3)
            self.obstacles.add(tuple(obstacle))
        self.start_position_3d = (0, 0, 0)
        self.goal_position_3d = (size-1, size-1, size-1)

    # Fonction Reward
    # 100 si objectif atteint, -100 si dans obstacle, -1 si case normale
    def get_reward(self, position_3d):
        if position_3d == self.goal_position_3d:
            return 100
        elif position_3d in self.obstacles:
            return -100
        else:
            return -1

# Définititon de la Q-Table adaptée à l'environnement 3D
class CubeQTable:
    def __init__(self, size):
        self.size = size
        self.q_table = np.zeros((size, size, size, 6))

    # Choix de l'action (aléatoire dans 10% des cas sinon choix de la meilleure action)
    def get_action(self, state):
        if np.random.uniform(0, 1) < 0.1:
            return np.random.randint(0, 6)
        else:
            return np.argmax(self.q_table[state])

    # Position suivante en fonction de la position actuelle et de l'action choisie
    def get_next_position(self, position_3d, action):
        next_position_3d = list(position_3d)
        if action == 0:
            next_position_3d[0] += 1
        elif action == 1:
            next_position_3d[0] -= 1
        elif action == 2:
            next_position_3d[1] += 1
        elif action == 3:
            next_position_3d[1] -= 1
        elif action == 4:
            next_position_3d[2] += 1
        elif action == 5:
            next_position_3d[2] -= 1
        next_position_3d = tuple(next_position_3d)
        next_position_3d = self.clip_position_to_boundaries(next_position_3d)
        return next_position_3d

    # Vérifie que l'action choisie ne mène pas à un mouvement interdit (sortie du cube)
    def clip_position_to_boundaries(self, position_3d):
        x, y, z = position_3d
        x = max(min(x, self.size-1), 0)
        y = max(min(y, self.size-1), 0)
        z = max(min(z, self.size-1), 0)
        return (x, y, z)

    # Met à jour la Q-Table en fonction de l'action effectuée et du reward obtenu
    def update(self, state, action, reward, next_state):
        current_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - LEARNING_RATE) * current_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        self.q_table[state][action] = new_value

    # Permet d'obtenir la trajectoire optimale (choix de la meilleure action à chaque itération)
    def get_optimal_trajectory(self, start_position_3d, goal_position_3d):
        current_position_3d = start_position_3d
        trajectory_3d = [start_position_3d]
        while current_position_3d != goal_position_3d:
            x, y, z = current_position_3d
            state = (x, y, z)
            action = np.argmax(q_table.q_table[state])
            next_position_3d = q_table.get_next_position(current_position_3d, action)
            trajectory_3d.append(next_position_3d)
            current_position_3d = next_position_3d
        return trajectory_3d

# Définir les paramètres d'apprentissage
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 1000
STEPS_PER_EPISODE = 100

# Créer l'environnement et la Q-table
size = 5
env = CubeEnvironment(size)
q_table = CubeQTable(size)

# Entraîner la Q-table
for episode in range(EPISODES):
    current_position_3d = env.start_position_3d
    x, y, z = current_position_3d
    state = (x, y, z)
    for step in range(STEPS_PER_EPISODE):
        action = q_table.get_action(state)
        next_position_3d = q_table.get_next_position(current_position_3d, action)
        reward = env.get_reward(next_position_3d)
        x, y, z = next_position_3d
        next_state = (x, y, z)
        q_table.update(state, action, reward, next_state)
        if next_position_3d == env.goal_position_3d:
            break
        current_position_3d = next_position_3d
        state = next_state

# Récupérer la trajectoire optimale
optimal_trajectory = q_table.get_optimal_trajectory(env.start_position_3d, env.goal_position_3d)


import plotly.graph_objects as go

# Créer la figure Plotly pour la visualisation 3D
fig = go.Figure(data=[go.Scatter3d(
    x=[env.start_position_3d[0]],
    y=[env.start_position_3d[1]],
    z=[env.start_position_3d[2]],
    mode='markers',
    marker=dict(size=8, color='red')
)])

# Ajouter les positions à la figure Plotly
for position in env.obstacles:
    fig.add_trace(go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode='markers',
        marker=dict(size=8, color='blue')
    ))

fig.add_trace(go.Scatter3d(
    x=[env.goal_position_3d[0]],
    y=[env.goal_position_3d[1]],
    z=[env.goal_position_3d[2]],
    mode='markers',
    marker=dict(size=8, color='green')
))

# Ajouter la trajectoire optimale à la figure Plotly
x_traj, y_traj, z_traj = zip(*optimal_trajectory)
fig.add_trace(go.Scatter3d(
    x=x_traj,
    y=y_traj,
    z=z_traj,
    line=dict(color='red', width=4),
    mode='lines'
))

# Mettre en place les paramètres de la caméra pour la visualisation 3D
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=size/2, y=size/2, z=size/2),
    eye=dict(x=size*1.5, y=size*1.5, z=size/2)
)
fig.update_layout(scene_camera=camera)

# Définir la mise en page finale et afficher la figure
fig.update_layout(scene=dict(
    xaxis=dict(range=[-1, size], autorange=False),
    yaxis=dict(range=[-1, size], autorange=False),
    zaxis=dict(range=[-1, size], autorange=False)),
    width=800,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10)
)

import plotly.io as pio

pio.write_html(fig, file='viz_3d.html', auto_open=True)
