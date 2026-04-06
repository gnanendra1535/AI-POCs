import pandas as pd
import os
input_path = os.path.join(os.getcwd(), "EmployeeDetails.csv")  # Adjust path if needed
df = pd.read_csv(input_path)

df.rename(columns=lambda x: x.strip(), inplace=True)

df[['First Name', 'Last Name']] = df['Name'].str.split(' ', 1, expand=True)

df.drop(columns=['Name'], inplace=True)

df['Salary'] = df['Salary'] * 1.10

output_path = os.path.join(os.getcwd(), "Employee_Data.csv")
df.to_csv(output_path, index=False)

print("Updated Employee Data saved successfully as:", output_path)
print(df.head())