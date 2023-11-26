import json

file = open('yelp_academic_dataset_business/convert_data_address.txt', 'r')

content = file.read()
lines = content.split("\n");

addresses = []
for each in lines:
    if each=="":
        continue;
    ascii_string = ''.join(char for char in each if ord(char) < 128)
    user = json.loads(ascii_string)
    address = user.get('address', {})  # Replace 'address' with the actual key in your JSON
    state = user.get('state', '')
    city = user.get('city', '')
    postal_code = user.get('postal_code', '')
    latitude = user.get('latitude', '')  # Replace 'latitude' with the actual key
    longitude = user.get('longitude', '')  # Replace 'longitude' with the actual key

    full_address = f"{address} | {state} | {city} | {postal_code} | {latitude} | {longitude}"
    addresses.append(full_address)

unique_addresses = list(set(addresses))

print(len(unique_addresses))

# Open a file in write mode
with open('adresses.json', 'w') as file:
    # Dump the list into the file
    json.dump(unique_addresses, file)


