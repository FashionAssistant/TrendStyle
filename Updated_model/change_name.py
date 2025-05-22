import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('fashion.csv')

# Function to simplify category names
def modify_category_name(name):
    # Convert to lowercase to handle case insensitivity
    name = name.lower()
    print(name)

    # Replace the variations with their simplified versions
    if 't-shirt' in name:
        print("yes-t-shirt")
        name = 't-shirt'
    elif 'shirt' in name:
        print("yes-shirt")
        name = 'shirt'
    elif 'short' in name:
        print("yes-short")
        name = 'short'
    elif 'hoodies' in name:
        print("yes-hoodies")
        name = 'hoodie'
    elif 'jackets' in name:
        print("yes-jackets")
        name = 'jacket'

    # Return the name in proper title case
    return name.title()

# Apply the function to the 'category_name' column
df['category_name'] = df['category_name'].apply(modify_category_name)

# Save the updated DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)