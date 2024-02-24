import pandas as pd

# Creating a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)

# Displaying the DataFrame
print("Original DataFrame:")
print(df)

# Accessing columns
print("\nAccessing 'Name' column:")
print(df['Name'])

# Adding a new column
df['Salary'] = [50000, 60000, 70000, 80000]
print("\nDataFrame after adding 'Salary' column:")
print(df)

# Filtering data
print("\nFiltering rows where Age is greater than 30:")
filtered_df = df[df['Age'] > 30]
print(filtered_df)

# Reading from a CSV file
# Assuming you have a CSV file named 'example.csv' with columns 'Name', 'Age', 'City'
# df_csv = pd.read_csv('example.csv')

# Writing to a CSV file
# df.to_csv('output.csv', index=False)
