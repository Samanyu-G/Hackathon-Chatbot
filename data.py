import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('chatbot_dataset.csv')

# Group the DataFrame by the "Personality" column
grouped = df.groupby('Personality')

# Loop through each group and save it into a separate file
for name, group in grouped:
    # Save each group to a new file, with the file name being the personality
    group.to_csv(f'{str(name).lower()}_data.csv', index=False)
    print(f'Saved data for {name} to {str(name).lower()}_data.csv')
