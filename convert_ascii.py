with open('yelp_academic_dataset_business/data_address.txt', 'rb') as file:
    content = file.read()

ascii_content = content.decode('ascii', errors='ignore')


#Or write to a new file
with open('yelp_academic_dataset_business/convert_data_address.txt', 'w') as file:
    file.write(ascii_content)

