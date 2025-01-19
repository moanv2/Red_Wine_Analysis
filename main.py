import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

data = pd.read_csv("wine+quality/winequality-red.csv", delimiter=";")
print(data)

# 1. Data Loading and Exploration
# 	1.	Load the Wine Quality Dataset (red wine) into a pandas DataFrame. Display the first 5 rows.
# 	2.	What is the shape of the dataset? How many features and rows does it have?
# 	3.	Check for missing values in the dataset. How would you handle them if they exist?
# 	4.	What are the basic statistics (mean, median, standard deviation) of each column?

print(data.head(5))
print(data.shape)
print(data.isna().count())
print(data.describe())

# 2. Data Manipulation
# 	5.	Rename the columns for better readability. For example, change residual sugar to residual_sugar.
# 	6.	Create a new feature: acidity_ratio = fixed acidity / volatile acidity. Add it to the dataset.
# 	7.	Normalize the alcohol column using Min-Max scaling. Which row now has the highest value for alcohol?
# 	8.	Group the dataset by quality and calculate the mean for each feature. What insights can you derive?

print(data.columns)
data.columns = [column.replace(" ", "_") for column in data.columns]
print(data.columns)

data["acidity_ratio"] = (data["fixed_acidity"] / data["volatile_acidity"]).round(3)
print(data)

print(data["alcohol"])

# Turn column into an array first
two_dim_alc = np.array(data["alcohol"]).reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(two_dim_alc)
max_index = np.argmax(scaled_data)

print(f"Highest value for alcohol is in column: {max_index}")

quality_grouping = data.groupby("quality").mean()
print(quality_grouping)
print("We can see that the quality increases as acidity_ratio decreases alongside volatile_acidity")

# 3. Data Visualization
# 	9.	Plot a histogram for the quality column. Is the dataset balanced in terms of wine quality?
# 	10.	Create a boxplot for alcohol against quality. What does the trend show?
# 	11.	Generate a heatmap of the correlation matrix. Which features are most correlated with wine quality?
# 	12.	Use a scatter plot to visualize the relationship between citric acid and pH. Is there any noticeable pattern?


def hist():
    plt.hist(data["quality"])
    plt.xlabel("Quality Score")
    plt.ylabel("Quality Count")
    plt.title("Red Wine Quality")


def boxplot():
    # Group the 'alcohol' values by 'quality'
    grouped_alcohol = [data[data["quality"] == q]["alcohol"].values for q in sorted(data["quality"].unique())]
    plt.boxplot(grouped_alcohol, tick_labels=sorted(data["quality"].unique()))
    plt.xlabel("Quality")
    plt.ylabel("Alcohol")
    plt.title("Boxplot of Alcohol Content by Wine Quality")

    plt.show()

# Interpretation:


def heatmap_Corr():
    Corr_matrix = data.corr()
    sns.heatmap(Corr_matrix,  annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    # Volatile acidity, Citric acid


def scatter_plot():
    plt.plot(data["citric_acid"].values, data["pH"])
    plt.title("Relationship between citric acid and pH")

    plt.show()




# 4. Feature Selection
# 	13.	Use SelectKBest from sklearn.feature_selection to select the top 5 features most correlated with quality. What are they?
# 	14.	Perform PCA (Principal Component Analysis) to reduce the dataset to 2 dimensions. Plot the data points in a scatter plot with colors based on quality.


def Feature_Selection():

    top_5_features = SelectKBest(data["quality"].head(5))
    print(top_5_features)





def Principal_Component_Analysis():
    X = data.drop(columns=["quality"])  # Drop 'quality' for PCA
    y = data["quality"]  # Use 'quality' for coloring in the plot

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)

    # Add a colorbar
    plt.colorbar(scatter, label="Wine Quality")

    # Add labels and title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Wine Quality Dataset")

    # Show the plot
    plt.show()



# 5. Model Building
# 	15.	Split the dataset into training and testing sets (80-20 split). What is the distribution of quality in the training and testing sets?
# 	16.	Train a Linear Regression model to predict quality. What is the R² score on the test set?
# 	17.	Train a Random Forest Regressor on the same dataset. How does its performance compare to Linear Regression?
# 	18.	Use GridSearchCV to optimize the hyperparameters of the Random Forest model. Which parameters lead to the best performance?


