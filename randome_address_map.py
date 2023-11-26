def map_users_to_addresses(user_ids, addresses):
    """
    Randomly map each user to an address.

    :param user_ids: A list of user IDs.
    :param addresses: A list of addresses.
    :return: A dictionary with user IDs as keys and randomly assigned addresses as values.
    """
    if len(addresses) < len(user_ids):
        raise ValueError("There are more users than addresses available.")

    # Randomly sample unique addresses for each user
    selected_addresses = random.sample(addresses, len(user_ids))

    # Map each user to an address
    user_address_map = {user_id: address for user_id, address in zip(user_ids, selected_addresses)}

    return user_address_map


import json
import random

with open('adresses.json', 'r') as file:
    addresses = json.load(file)



total_user_cnt = len(addresses);
user_ids = [];
for each in range(total_user_cnt):
    if each==0:
        continue;
    user_ids.append(each)

print(user_ids)
user_address_map = map_users_to_addresses(user_ids, addresses)


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('merged_data.csv')

# Example of the map
print(user_address_map)
# Map the addresses to the DataFrame
df['address'] = df['userId'].map(user_address_map)

# Save the DataFrame to a new CSV file
df.to_csv('modified_merged_data.csv', index=False)
