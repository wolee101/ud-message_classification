import argparse
import pandas as pd
import sqlite3
import time


'''
Run the script like this:
python3 process_data.py 'messages.csv' 'categories.csv' 'message_cleaned.db' 'message_cleaned'
'''

###########################
#### CLEANUP FUNCTIONS ####
###########################

def clean_df(df):
    '''
    This function creates new columns for message categories and clean the data
    so each column has a value of 1 or 0 for the correspoding message category.
    except for the related column where the values range from 1 to 3.

    Input:
    df - dataframe to be cleaned

    Output:
    df2 - cleaned dataframe

    '''

    # Split the catgories column by ';' and create new columns
    df_new = pd.concat([df[df.columns[0]], df[df.columns[1]].str.split(';', expand = True)], axis = 1)

    for (colname, coldata) in df_new.iteritems():
        # Exclude the id column
        if colname != 'id':
            # Change the column names using the corresponding value in columns
            new_col = str(df_new[colname].iloc[0]).split('-')[0]
            df_new.rename(columns={colname: new_col}, inplace = True)
            # Separate the numeric value from the string value
            df_new[new_col] = df_new[new_col].apply(lambda x: x.split('-')[1])
            # Convert the data type from string to numeric
            df_new[new_col] = pd.to_numeric(df_new[new_col])
    return df_new


###################################
##### DATA LOADING FUNCTION ######
###################################

def load_data(data1, data2):
    '''
    This function load the two csv data files, clean them, and stores it in the SQLitr database.

    Input:
    data1 - messages data
    data2 - message categories data

    Output:
    X - message features
    y - 36 categories
    '''

    # Read in file
    df1 = pd.read_csv(data1)
    df2 = pd.read_csv(data2)

    # Clean and merge data
    df2_new = clean_df(df2)
    df_merged = pd.merge(df1, df2_new, on = 'id', how = 'left')

    # Drop duplicates
    df_merged = df_merged.drop_duplicates(subset=['id'], keep='last')

    return df_merged

#########################################
############# Main Function #############
#########################################

def main(data1_path, data2_path, sql_file_path, sql_table_name):
    # Load data files
    print("Loading csv files...")
    start = time.time()
    df_merged = load_data(data1_path, data2_path)
    print(f'\nFinished tokenizing data. Took {(time.time() - start)/60} minutes')

    # Store it to SQLite database
    print("\nSaving a sql file...")
    conn = sqlite3.connect(sql_file_path)
    df_merged.to_sql(sql_table_name, con = conn, if_exists='replace', index = False)
    print("\nSQL file saved.")

#########################################
############ Set Up Objects #############
#########################################
## Parse in Arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('data1_path', help = "path to the messages file")
parser.add_argument('data2_path', help = "path to the message catgories file")
parser.add_argument('sql_file_path', help = "name of the SQLite file to store the cleaned data ")
parser.add_argument('sql_table_name', help = "table name to save the cleaned data")
args = parser.parse_args()

########################################
################ Starter ################
#########################################
if __name__ == "__main__":
    main(args.data1_path, args.data2_path, args.sql_file_path, args.sql_table_name)
