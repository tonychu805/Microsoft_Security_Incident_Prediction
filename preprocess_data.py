import pandas as pd

def drop_column_with_over_50_percent_missing(df):
    missing_values = df.isnull().sum()
    missing_percentages = 100 * missing_values / len(df)

    # Display columns with missing values
    missing_data = pd.concat([missing_values, missing_percentages], axis=1, keys=['Total', 'Percent'])
    list = missing_data[missing_data.Percent > 50].index.tolist()
    df_drop = df.drop(columns = list)
    
    return df_drop


def fill_missing_value(df, column):
    df[column].fillna(df[column].mode()[0], inplace = True)


def timestamp_transform(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['DayOfWeek'] = df['Timestamp'].dt.day_name() 
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)





