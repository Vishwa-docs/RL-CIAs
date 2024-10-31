import numpy as np
import pandas as pd

n_arms = 10  # Arms
n_events = 1000  # Plays
epsilon = 0.1  # Exploration probability

def simulate_user_engagement(item_conversion_rate):
    return np.random.rand() < item_conversion_rate


####################

# epsilon-greedy
def epsilon_greedy(true_rewards, arms, num_iterations, epsilon):
    num_items = len(true_rewards)
    q_values = np.zeros(num_items)
    n_pulls = np.zeros(num_items)
    total_rewards = []

    for _ in range(num_iterations):
        if np.random.rand() < epsilon: # Exploration
            selected_item = np.random.choice(num_items)
        else: # Exploitation
            selected_item = np.argmax(q_values) # Arm with highest value

        # Update rewards
        reward = simulate_user_engagement(true_rewards[selected_item])
        total_rewards.append(reward)

        n_pulls[selected_item] += 1
        q_values[selected_item] += (reward - q_values[selected_item]) / n_pulls[selected_item]

    return total_rewards, q_values, n_pulls


####################

# Generate dataset
np.random.seed(0)
true_rewards = np.random.rand(n_arms)  # True rewards probability
arms = np.random.randint(0, n_arms, size=n_events)
rewards = np.array([np.random.binomial(1, true_rewards[arm]) for arm in arms])  # Simulate rewards


####################
total_rewards, q_values, n_pulls = epsilon_greedy(true_rewards, arms, n_events, epsilon)

####################
# Final estimated conversion rates
estimated_conversion_rates = q_values

####################
print("\nEstimated Conversion Rates:")
for i, (rate, count) in enumerate(zip(estimated_conversion_rates, n_pulls)):
    print(f"Item {i + 1}: Estimated Rate = {rate:.2f}, Selections = {count}")