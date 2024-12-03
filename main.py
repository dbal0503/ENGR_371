import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def calculate_var(df):
    non_mean_columns = [col for col in df.columns if 'mean' not in col]
    for column in non_mean_columns:
        variance = df[column].var()
        new_column = f"{column}_var"
        df[new_column] = variance   

    
def calculate_std(df):
    non_mean_and_var_columns = [col for col in df.columns if 'mean' not in col and not "var" in col]
    for column in non_mean_and_var_columns:
        std = df[column].std()
        new_column = f"{column}_std"
        df[new_column] = std

def calculate_mean(df):
    for column in df.columns:
        mean_value = df[column].mean()
        new_column = f"{column}_mean"
        df[new_column] = mean_value

def calculate_median(df):
    non_mean_variance_std_columns = [col for col in df.columns if 'mean' not in col and 'variance' not in col and 'std' not in col]
    for column in non_mean_variance_std_columns:
        median = df[column].median()
        new_column = f"{column}_median"
        df[new_column] = median

def LinearRegressionfunction(df):
    X = df.drop(columns="acquisition_rate") ##Acquisition rate is offers/# of interviews
    y = df["acquisition_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    model = LinearRegression()
    model.fit(X_train, y_train)

    coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
    coefficients.to_csv("coefficients.csv")
    coefficients_max = coefficients.loc[coefficients["Coefficient"].abs().idxmax()]
    print(coefficients_max)


def main():
    
    PATH ="processed_data.csv"
    df = pd.read_csv(PATH)

    LinearRegressionfunction(df)
    
    calculate_mean(df)
    calculate_var(df)
    calculate_std(df)
    calculate_median(df)

    df.to_csv("data_with_statistics.csv")

    

    
main()
