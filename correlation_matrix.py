import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from PSV files in a directory
def load_psv_data(path: str) -> list[pd.DataFrame]:
    """
    Load and return a list of DataFrames from PSV files in the specified directory.
    """
    # List all PSV files in the directory
    training_setA = sorted(Path(path).glob("*.psv"))

    if not training_setA:
        raise ValueError(f"No PSV files found in the directory: {path}")
    
    dataframes = []
    for file in training_setA:
        df = pd.read_csv(file, delimiter="|")  # Read the PSV file
        dataframes.append(df)
    
    return dataframes

# Function to clean data: exclude specified columns and drop NaN values
def clean_data(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Clean the DataFrame by excluding specified columns and dropping NaN values.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Exclude specified columns (e.g., age group columns)
    df_numeric = df_numeric.drop(columns=[col for col in df_numeric.columns if col in exclude_columns])
    
    # Drop rows with NaN values
    df_numeric = df_numeric.dropna()
    
    return df_numeric

# Function to generate a correlation heatmap
def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (8, 6), title: str = "Correlation Matrix of Features") -> None:
    """
    Generate and display a heatmap of the correlation matrix for the given DataFrame.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

# Define the path to the dataset
dataset_path = '/content/temp_data/cleaned_dataset'

# List of age group columns to exclude
exclude_columns = [
    "Age_18-44", "Age_45-59", "Age_60-64", 
    "Age_65-74", "Age_75-79", "Age_80-89"
]

# Main function to load, clean, and plot the data
def main():
    try:
        # Load the data from the PSV files
        df_list = load_psv_data(dataset_path)

        # Combine all dataframes into one
        df = pd.concat(df_list, ignore_index=True)

        # Clean the data (exclude age groups and drop NaN values)
        df_cleaned = clean_data(df, exclude_columns)

        # Generate and plot the correlation heatmap
        plot_correlation_heatmap(df_cleaned)

    except ValueError as e:
        print(str(e))

# Run the main function
if __name__ == "__main__":
    main()
