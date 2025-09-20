import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

class ParkingLot:
    def __init__(self):
        self.base_ped = -0.4  # Price elasticity of demand
        self.CAPACITY = 100
        self.COST_PRICE = 100
        self.FIXED_PRICE = 150
        self.BASE_ARRIVAL_RATES = np.array([
            0.5, 0.5, 0.2, 0.2, 0.5, 2, 3, 3, 4, 4, 4, 5,
            10, 12, 15, 18, 20, 15, 10, 7, 5, 3, 1, 0.5
        ])

    def get_ped(self, time_slot, day_type, weather_type):
        ped = self.base_ped
        
        # Time modifier
        if 0 <= time_slot < 13: 
            time_modifier = 0.7
        elif 13 <= time_slot < 17: 
            time_modifier = 1.27
        else:
            time_modifier = 1.0
        ped *= time_modifier

        # Day type modifier (1 for weekend/holiday)
        if day_type == 1:
            day_modifier = 1.1
            ped *= day_modifier

        # Weather modifier (1 for bad weather)
        if weather_type == 1:
            weather_modifier = 0.7
            ped *= weather_modifier

        return ped
    
    def calculate_occupancy(self, dynamic_price, time_slot, occupancy, day_type, weather_type):
        PED = self.get_ped(time_slot, day_type, weather_type)
        delta_price = ((dynamic_price - self.FIXED_PRICE) / self.FIXED_PRICE) *100
        delta_demand = delta_price * PED
        
        # Calculate available spaces
        qt = (1 - occupancy) * self.CAPACITY
        
        # Adjust arrival rate based on price change
        adjusted_lambda = max(0, self.BASE_ARRIVAL_RATES[time_slot] * (1 + delta_demand / 100))
        
        # Simulate new arrivals
        new_cars = np.random.poisson(lam=adjusted_lambda)
        
        # Calculate actual parked cars (limited by available spaces)
        actual_parked = min(new_cars, qt)
        departure_rate = 0.1 * occupancy*self.CAPACITY
        # Update occupancy
        new_occupancy = ((occupancy * self.CAPACITY) + actual_parked - departure_rate) / self.CAPACITY
            
        return new_occupancy, actual_parked, delta_demand
    
    def calculate_reward(self, occupancy, delta_demand, actual_parked, dynamic_price):
        is_peak = occupancy >= 0.6
        x = 0.99 if is_peak else 0.01
        
        # Calculate profit from sold spots (only the ones that actually parked)
        profit_from_sold = actual_parked * (dynamic_price - self.COST_PRICE)
        
        # Calculate potential revenue from demand change
        revenue_from_demand = delta_demand * dynamic_price / 100  # Scale by 100 to match percentage
        
        reward = x * profit_from_sold + (1 - x) * revenue_from_demand
        return reward

class QLearningAgent:
    def __init__(self):
        self.LEARNING_RATE = 0.2
        self.DISCOUNT_FACTOR = 0.8
        self.EPSILON = 0.5
        self.FINAL_EPSILON = 0.01
        self.TOTAL_EPISODES = 50000  # Reduced for faster testing
        self.TIME_SLOTS = 24    
        self.OCC_BUCKETS = 10  # 0-10%, 10-20%, ..., 90-100%
        self.DAY_TYPES = 2
        self.WEATHER_TYPES = 2
        self.PRICE_MULTIPLIERS = np.round(np.linspace(1.0, 2.0, 11), decimals=1)
        self.ACTION_SPACE_SIZE = len(self.PRICE_MULTIPLIERS)
        self.state_space_shape = (self.TIME_SLOTS, self.OCC_BUCKETS, self.DAY_TYPES, self.WEATHER_TYPES)
        
        self.alpha = self.LEARNING_RATE
        self.gamma = self.DISCOUNT_FACTOR
        self.epsilon = self.EPSILON
        
        # Initialize Q-table with small random values to encourage exploration
        self.q_table = np.random.uniform(low=-1, high=1, size=self.state_space_shape + (self.ACTION_SPACE_SIZE,))
    
    def get_state_index(self, state):
        time_slot, occupancy, day_type, weather_type = state
        # Convert occupancy percentage to bucket index (0-9)
        occupancy_bucket = min(self.OCC_BUCKETS - 1, int(occupancy * 10))
        return (time_slot, occupancy_bucket, day_type, weather_type)
    
    def choose_action(self, state):
        state_idx = self.get_state_index(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.ACTION_SPACE_SIZE)
        else:
            return np.argmax(self.q_table[state_idx])
        
    def update_q_value(self, state, action, reward, next_state):
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        old_value = self.q_table[state_idx + (action,)]
        next_max = np.max(self.q_table[next_state_idx])
        
        # Q-learning update formula
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_idx + (action,)] = new_value

    def decay_epsilon(self, episode):
        # Linear epsilon decay
        self.epsilon = max(self.FINAL_EPSILON, 
                          self.EPSILON - (self.EPSILON - self.FINAL_EPSILON) * episode / self.TOTAL_EPISODES)

def run_training():
    env = ParkingLot()
    agent = QLearningAgent()
    rewards_log = []
    
    for episode in range(agent.TOTAL_EPISODES):
        day_type = np.random.randint(0, agent.DAY_TYPES)
        weather_type = np.random.randint(0, agent.WEATHER_TYPES)
        occupancy = np.random.uniform(0, 0.2)  # Start with low occupancy
        episode_reward = 0
        
        agent.decay_epsilon(episode)
        
        for time_slot in range(agent.TIME_SLOTS):
            current_state = (time_slot, occupancy, day_type, weather_type)
            action_idx = agent.choose_action(current_state)
            dynamic_price = env.COST_PRICE * agent.PRICE_MULTIPLIERS[action_idx]
            
            # Calculate new occupancy
            occupancy, actual_parked, delta_demand = env.calculate_occupancy(
                dynamic_price, time_slot, occupancy, day_type, weather_type
            )
            
            # Calculate reward
            reward = env.calculate_reward(occupancy, delta_demand, actual_parked, dynamic_price)
            
            # Next state (next time slot)
            next_time_slot = (time_slot + 1) % agent.TIME_SLOTS
            next_state = (next_time_slot, occupancy, day_type, weather_type)
            
            # Update Q-value
            agent.update_q_value(current_state, action_idx, reward, next_state)
            
            episode_reward += reward
            
        rewards_log.append(episode_reward)
        
        if episode % 1000 == 0:  # Print less frequently to reduce output
            avg_reward = np.mean(rewards_log[-1000:]) if episode > 1000 else np.mean(rewards_log)
            print(f"Episode: {episode:05d} | Avg Reward: {avg_reward:08.2f} | Epsilon: {agent.epsilon:.3f}")
    
    print("Training complete!")
    return agent, env, rewards_log

def test_policy(agent, env, day_type, weather_type, initial_occupancy=0.1):
    time_slots = list(range(agent.TIME_SLOTS))
    occupancies, prices, revenues = [], [], []
    
    # Set epsilon to 0 for deterministic policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    occupancy = initial_occupancy
    total_revenue = 0
    for time_slot in range(agent.TIME_SLOTS):
        current_state = (time_slot, occupancy, day_type, weather_type)
        action_idx = agent.choose_action(current_state)
        dynamic_price = env.COST_PRICE * agent.PRICE_MULTIPLIERS[action_idx]
        
        occupancy, actual_parked, delta_demand = env.calculate_occupancy(
            dynamic_price, time_slot, occupancy, day_type, weather_type
        )
        
        # Calculate revenue for this time slot
        revenue = actual_parked * dynamic_price
        total_revenue += revenue
        
        occupancies.append(occupancy)
        prices.append(dynamic_price)
        revenues.append(revenue)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return time_slots, occupancies, prices, revenues, total_revenue

# Run training
print("Starting training...")
trained_agent, parking_env, rewards = run_training()

