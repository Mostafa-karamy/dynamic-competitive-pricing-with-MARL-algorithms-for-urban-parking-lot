import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# the market base demand
base_total_demand = np.array([
    50, 50, 20, 20, 50, 200, 300, 300, 400, 400, 400, 500,
    1000, 1200, 1500, 1800, 2000, 1500, 1000, 700, 500, 300, 100, 50
])
# base_total_demand = base_total_demand * 2

# the parking lot attributes and functions
class ParkingLot:
    def __init__(self, name, capacity, cost_price, quality, distance_km):
        self.name = name
        self.capacity = capacity
        self.cost_price = cost_price
        self.quality = quality
        self.distance_km = distance_km
        self.occupancy = 0.0
        self.num_serviced = []
        self.price = cost_price * 1.1
        self.current_demand = 0

    def record_service(self, arrivals, rejected):
        self.num_serviced.append(max(0, arrivals - rejected))
        if len(self.num_serviced) > 6:
            self.num_serviced.pop(0)

    def compute_departures(self):
        if len(self.num_serviced) <= 3:
            return 0.0
        return (
            0.25 * self.num_serviced[-2]
            + 0.5 * self.num_serviced[-3]
            + 0.25 * self.num_serviced[-4]
        )

    def calculate_lot_condition(self, last_occupancy, arrivals, departure):
        inside = (last_occupancy * self.capacity) + arrivals - departure
        occupancy = inside / max(1e-8, self.capacity)
        real_occupancy = float(np.clip(occupancy, 0.0, 1.0))

        if occupancy > 1.0:
            rejected = int(round(max(0, inside - self.capacity)))
            full = True
        else:
            rejected = 0
            full = False
        return real_occupancy, rejected, full

class CompetitiveParkingEnv:
    def __init__(self, lots, sensitivities, day, wether):
        self.wether_type = wether
        self.day_type = day
        self.lots = lots
        self.sens = sensitivities
        self.time_slot = 0
        starting_mean_price = np.mean(1.1 * (np.array([lot.price for lot in self.lots])))
        self.mean_price_list = [float(starting_mean_price)]
        self.final_demand_list = []
        self.lot_capacities = np.array([lot.capacity for lot in self.lots], dtype=float)
        self.reward_list = []
        self.cum_reward = 0.0

    def normalize_attributes(self):
        prices = np.array([lot.price for lot in self.lots], dtype=float)
        costs = np.array([lot.cost_price for lot in self.lots], dtype=float)
        
        def fixed_norm(x, min_val=0, max_val=400):
            rng = (max_val - min_val) + 1e-6
            return np.clip((x - min_val) / rng, 0.0, 1.0)
        
        qualities = np.array([lot.quality for lot in self.lots], dtype=float)
        distances = np.array([lot.distance_km for lot in self.lots], dtype=float)
        
        def dyn_norm(x):
            x = np.array(x, dtype=float)
            xmin = np.min(x)
            xmax = np.max(x)
            if np.isclose(xmax, xmin):
                return np.zeros_like(x)
            return (x - xmin) / (xmax - xmin)

        norm_price = fixed_norm(prices)
        norm_quality = dyn_norm(qualities)
        norm_distance = dyn_norm(distances)

        return norm_price, norm_quality, norm_distance

    def compute_utilities(self):
        norm_price, norm_quality, norm_distance = self.normalize_attributes()
        utilities = (
            - self.sens['price'] * norm_price
            + self.sens['quality'] * norm_quality
            - self.sens['distance'] * norm_distance
        )
        return utilities

    def add_avg_price(self):
        avg_price = float(np.mean([lot.price for lot in self.lots]))
        self.mean_price_list.append(avg_price)

    def get_ped(self, time_slot, day_type, weather_type):
        ped = -0.4
        if 0 <= time_slot < 13:
            time_modifier = 0.7
        elif 13 <= time_slot < 17:
            time_modifier = 1.27
        else:
            time_modifier = 1.0
        ped *= time_modifier
        if day_type == 1:
            ped *= 1.1
        if weather_type == 1:
            ped *= 0.7
        return float(ped)

    def demand_change(self, PED):
        desired_price = float(np.mean([lot.cost_price for lot in self.lots])) * 1.50
        new_price = self.mean_price_list[-1]
        demand_factor = PED * ((new_price - desired_price) / max(1e-8, desired_price))
        return float(demand_factor)

    def total_demand(self, hour, demand_factor):
        base_demand = max(0.0, base_total_demand[hour] * (1.0 + demand_factor))
        final_demand = int(np.random.poisson(max(0.0, base_demand)))
        return int(max(0, final_demand))

    def mnl_allocation(self, final_demand):
        utilities = self.compute_utilities()
        exp_u = np.exp(utilities - np.max(utilities))
        denom = np.sum(exp_u) + 1e-12
        shares = exp_u / denom
        return shares * float(final_demand)

    def step(self, multipliers, hour):
        for lot, mult in zip(self.lots, multipliers):
            lot.prev_price = lot.price
            lot.price = lot.cost_price * mult


        self.add_avg_price()
        market_avg_price = float(np.mean([lot.price for lot in self.lots]))

        PED = self.get_ped(hour, self.day_type, self.wether_type)
        demand_factor = self.demand_change(PED=PED)
        final_demand = self.total_demand(hour, demand_factor)
        lot_demands = self.mnl_allocation(final_demand)

        step_data = []
        for lot_index, (lot, demand) in enumerate(zip(self.lots, lot_demands)):

            departs = lot.compute_departures()
            arrivals = max(0, int(round(demand)))  # raw inbound demand
            lot.occupancy, rejected, full = lot.calculate_lot_condition(lot.occupancy, arrivals, departs)
            lot.record_service(arrivals, rejected)

            profit = lot.price - lot.cost_price
            served = max(0, arrivals - rejected)
            profit_gained = profit * served

            total_capacity = np.sum(self.lot_capacities)
            capacity_share = lot.capacity / max(1e-8, total_capacity)
            target_arrivals = capacity_share * final_demand

            arrival_bonus = (arrivals - target_arrivals) * profit * 1.1

            reject_penalty = rejected * profit * 1.5

            is_peak = hour >= 14 and hour <= 19
            x = 0.7 if is_peak else 0.3
            raw_reward = (
                (x * profit_gained)
                + ((1-x) * arrival_bonus)
                - reject_penalty
            )

            reward = raw_reward

            rec = {
                'Hour': hour,
                'Lot': lot.name,
                'Price': round(float(lot.price), 2),
                'Arrivals': int(arrivals),
                'Departures': float(departs),
                'Rejected': int(rejected),
                'Occupancy_%': round(float(lot.occupancy) * 100.0, 2),
                'Profit_Gained': round(float(profit_gained), 2),
                'Arrival_Bonus': round(float(arrival_bonus), 2),
                'Reject_Penalty': round(float(reject_penalty), 2),
                'Reward': round(float(reward), 2),
                'day_type': int(self.day_type),
                'weather': int(self.wether_type),
            }
            step_data.append(rec)
            self.cum_reward += float(reward)
            self.reward_list.append(float(reward))

        done = (hour == 23)
        return step_data, market_avg_price, done




def get_state(obs, market_avg_price, lots, lot_index, btd, current_hour):
    future_window = 2 
    future_demand = [
        btd[(current_hour + i) % 24]
        for i in range(1, future_window + 1)
    ]
    avg_future_demand = sum(future_demand) / future_window

    occ_frac = float(obs['Occupancy_%']) / 100.0
    hour_frac = float(obs['Hour']) / 23.0
    day_type = float(obs['day_type'])
    weather_type = float(obs['weather'])
    rel_price = (float(obs['Price']) - float(market_avg_price)) / max(1e-8, float(market_avg_price))

    # Competitor-aware features
    own_price = float(obs['Price'])
    comp_prices = [lot.price for i, lot in enumerate(lots) if i != lot_index]
    min_comp_price = min(comp_prices) if comp_prices else own_price

    own_occ = occ_frac
    comp_occs = [lot.occupancy for i, lot in enumerate(lots) if i != lot_index]
    avg_comp_occ = float(np.mean(comp_occs)) if comp_occs else own_occ
    occ_gap_to_avg = own_occ - avg_comp_occ

    return np.array([
        occ_frac,              
        hour_frac,             
        day_type,               
        weather_type,           
        rel_price,              
        occ_gap_to_avg,         
        avg_future_demand       
    ], dtype=np.float32)




# DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.layers(x)


# Agent functions
class Agent:
    def __init__(self, state_dim, actions, gamma=0.95, lr= 0.002, buffer_size=100000, batch_size=128):
        self.actions = actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.decay = 0.999
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, len(actions))
        self.target_model = DQN(state_dim, len(actions))
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, greedy=False):
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        with torch.no_grad():
            q_values = self.model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q_values).item())

    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)
        s = torch.as_tensor(np.array(s, dtype=np.float32))
        a = torch.as_tensor(a, dtype=torch.long)
        r = torch.as_tensor(r, dtype=torch.float32)
        s_next = torch.as_tensor(np.array(s_next, dtype=np.float32))
        done = torch.as_tensor(done, dtype=torch.float32)
        q_pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_model(s_next).max(1)[0]
            q_target = r + self.gamma * q_next * (1.0 - done)
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


# get enviroment
def reset_env():
    lots = [
        ParkingLot("Lot A", 1000, 120, 0.7, 0.7),
        ParkingLot("Lot B", 1500, 95, 0.5, 1.4),
        ParkingLot("Lot C", 1250, 70, 0.3, 1.8),
    ]
    return CompetitiveParkingEnv(
        lots,
        sensitivities={'price': 1.5, 'quality': 0.1, 'distance': 0.2},
        day=np.random.randint(0, 2), wether=np.random.randint(0, 2)
    )

# initials training parameter
ACTIONS = np.linspace(0.8, 3.5, 20)
state_dim = 7

agent_A_eval = Agent(state_dim, ACTIONS)
agent_B_eval = Agent(state_dim, ACTIONS)
agent_C_eval = Agent(state_dim, ACTIONS)

MODEL_DIR = r"E:\codes and programming\sthochastic processes\codes\competative pricing code"
agent_A_eval.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dqn_agent_A_version_4.pth")))
agent_B_eval.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dqn_agent_B_version_4.pth")))
agent_C_eval.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dqn_agent_C_version_4.pth")))


for ag in (agent_A_eval, agent_B_eval, agent_C_eval):
    ag.epsilon = 0.0
    ag.model.eval() 

eval_agents = [agent_A_eval, agent_B_eval, agent_C_eval]

env = reset_env()
lots = env.lots
eval_steps = []

# evaluate policy and agents for a day
for hour in range(24):
    states, actions_idx = [], []
    market_avg_price = float(np.mean([lot.price for lot in lots]))

    for i, lot in enumerate(lots):
        obs_stub = {
            'Occupancy_%': lot.occupancy * 100.0,
            'Hour': hour,
            'day_type': env.day_type,
            'weather': env.wether_type,
            'Price': lot.price
        }
        s = get_state(obs_stub, market_avg_price, lots, i, base_total_demand, hour)
        a_idx = eval_agents[i].select_action(s, greedy=True)
        states.append(s)
        actions_idx.append(a_idx)

    multipliers = [ACTIONS[a] for a in actions_idx]
    step_data, market_avg_price, done = env.step(multipliers, hour)
    eval_steps.extend(step_data)


df_eval = pd.DataFrame(eval_steps)



# Make sure save path exists
save_path = r"D:\university files\university of tehran-masters\semester 2\stochastic processes\report photos"
os.makedirs(save_path, exist_ok=True)

# Create one figure with 4 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
lots_names = df_eval['Lot'].unique()
# 1) Prices
for lot in lots_names:
    lot_data = df_eval[df_eval['Lot'] == lot]
    axes[0].plot(lot_data['Hour'], lot_data['Price'], marker='o', label=lot)
axes[0].set_title("Price")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Price")
axes[0].grid(True)

# 2) Profit
for lot in lots_names:
    lot_data = df_eval[df_eval['Lot'] == lot]
    axes[1].plot(lot_data['Hour'], lot_data['Profit_Gained'], marker='o', label=lot)
axes[1].set_title("Profit Gained")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Profit")
axes[1].grid(True)

# 3) Occupancy
for lot in lots_names:
    lot_data = df_eval[df_eval['Lot'] == lot]
    axes[2].plot(lot_data['Hour'], lot_data['Occupancy_%'], marker='o', label=lot)
axes[2].set_title("Occupancy %")
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("Occupancy %")
axes[2].grid(True)


# Add one legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(lots_names))

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the whole row of graphs as one image
plt.savefig(os.path.join(save_path, "lots_results_row.png"), dpi=300, bbox_inches="tight")

plt.show()

# # Plot per-lot results
lots_names = df_eval['Lot'].unique()
# plt.figure(figsize=(8, 6))

# for lot in lots_names:
#     lot_data = df_eval[df_eval['Lot'] == lot]
#     plt.plot(lot_data['Hour'], lot_data['Occupancy_%'], marker='o', label=f"{lot}")

# plt.xlabel("Hour")
# plt.ylabel("Occupancy")
# plt.title("Occupancy of Lots in a Day")
# plt.legend()
# plt.grid(True)

# # Save the combined figure
# save_path = r"D:\university files\university of tehran-masters\semester 2\stochastic processes\report photos"
# os.makedirs(save_path, exist_ok=True)
# plt.savefig(os.path.join(save_path, "all_lots_prices.png"), dpi=300, bbox_inches="tight")

# plt.show()
# for i, lot in enumerate(lots_names):
#     lot_data = df_eval[df_eval['Lot'] == lot]

#      axes[i, 0].plot(lot_data['Hour'], lot_data['Price'], marker='o')
#      axes[i, 0].set_ylabel('Price')
#      axes[i, 0].set_title(f'{lot} - Price')
#      plt.savefig(f'Price in a day {lot}.png')

#      axes[i, 1].plot(lot_data['Hour'], lot_data['Profit_Gained'], marker='o', color='green')
#      axes[i, 1].set_ylabel('Profit')
#      axes[i, 1].set_title(f'{lot} - Profit Gained')
#      plt.savefig(f'profit in a day {lot}.png')

#      axes[i, 2].plot(lot_data['Hour'], lot_data['Occupancy_%'], marker='o', color='red')
#      axes[i, 2].set_ylabel('Occupancy %')
#      axes[i, 2].set_title(f'{lot} - Occupancy %')
#      plt.savefig(f'occupancy in a day {lot}.png')


#  plt.tight_layout()
#  plt.show()

