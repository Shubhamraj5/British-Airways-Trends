#!/usr/bin/env python
# coding: utf-8

# In[14]:


import re
import csv

# Read the text file
file_path = r"C:\Users\91882\Downloads\britsh airways data.txt"

with open(file_path, "r", encoding="utf-8") as file:
    data = file.read()

# Regular expressions to extract relevant fields
reviews = re.findall(r'(\d{1,2}/10)\n"(.*?)"\n(.*?)\((.*?)\) (\d{1,2}\w+ \w+ \d{4})\n(.*?)\nType Of Traveller\s+(.*?)\nSeat Type\s+(.*?)\nRoute\s+(.*?)\nDate Flown\s+(.*?)\n.*?Recommended\s+(yes|no)', data, re.DOTALL)

# Prepare CSV file
csv_filename = "eva_air_reviews.csv"

with open(csv_filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Rating", "Review Title", "Reviewer", "Country", "Review Date", "Review Content", "Type of Traveller", "Seat Type", "Route", "Date Flown", "Recommended"])
    writer.writerows(reviews)

print(f"Data successfully extracted and saved to {csv_filename}")


# In[16]:


import pandas as pd

# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Display the first few rows
print(df.head(100))


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Display basic information
print(df.info())
print(df.head(5))


# In[18]:


plt.figure(figsize=(8, 5))
sns.countplot(x="Rating", data=df, palette="coolwarm")
plt.title("Distribution of Ratings")
plt.xlabel("Rating (out of 10)")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()


# In[19]:


plt.figure(figsize=(6, 6))
df["Recommended"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightgreen", "red"])
plt.title("Percentage of Recommended Reviews")
plt.ylabel("")  # Hide y-label
plt.show()


# In[22]:


plt.figure(figsize=(12, 5))
df["Route"].value_counts().head(10).plot(kind="bar", color="skyblue")
plt.title("Top 10 Most Reviewed Routes")
plt.xlabel("Flight Route")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=75)
plt.show()


# In[24]:


pip install pandas numpy scikit-learn matplotlib seaborn


# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Convert 'Rating' column to numeric
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Convert categorical variables into numerical using Label Encoding
categorical_columns = ["Seat Type", "Type of Traveller", "Route", "Recommended"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define target variable (booking likelihood)
df["Booking"] = df["Recommended_yes"]  # Assuming customers who recommend are more likely to book again

# Drop unnecessary columns
df.drop(["Recommended_yes"], axis=1, inplace=True)

# Check the processed data
print(df.head())



# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Check for non-numeric columns
print(df.dtypes)

# Convert 'Rating' column to numeric (if not already)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Drop irrelevant text columns
df.drop(["Name", "Review Title", "Review Content"], axis=1, inplace=True, errors="ignore")

# Identify categorical columns
categorical_columns = ["Seat Type", "Type of Traveller", "Route", "Recommended"]

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define the target variable (Booking Prediction)
df["Booking"] = df["Recommended"]  # Assuming "Recommended" indicates booking likelihood

# Drop the original "Recommended" column
df.drop(["Recommended"], axis=1, inplace=True)

# Ensure all remaining columns are numeric
print(df.dtypes)

# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Check for non-numeric columns
print(df.dtypes)

# Convert 'Rating' column to numeric (if not already)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Drop irrelevant text columns
df.drop(["Name", "Review Title", "Review Content"], axis=1, inplace=True, errors="ignore")

# Identify categorical columns
categorical_columns = ["Seat Type", "Type of Traveller", "Route", "Recommended"]

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define the target variable (Booking Prediction)
df["Booking"] = df["Recommended"]  # Assuming "Recommended" indicates booking likelihood

# Drop the original "Recommended" column
df.drop(["Recommended"], axis=1, inplace=True)

# Ensure all remaining columns are numeric
print(df.dtypes)# Load the dataset
df = pd.read_csv("eva_air_reviews.csv")

# Check for non-numeric columns
print(df.dtypes)

# Convert 'Rating' column to numeric (if not already)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Drop irrelevant text columns
df.drop(["Name", "Review Title", "Review Content"], axis=1, inplace=True, errors="ignore")

# Identify categorical columns
categorical_columns = ["Seat Type", "Type of Traveller", "Route", "Recommended"]

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define the target variable (Booking Prediction)
df["Booking"] = df["Recommended"]  # Assuming "Recommended" indicates booking likelihood

# Drop the original "Recommended" column
df.drop(["Recommended"], axis=1, inplace=True)

# Ensure all remaining columns are numeric
print(df.dtypes)


# In[31]:


# Get feature importance from the model
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar", figsize=(12, 5), color="royalblue")
plt.title("Feature Importance in Predicting Booking")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




