################### Imports ###################

# Assorted Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Required for the API
import requests
from pandas.core.frame import dataclasses_to_dicts

# Required to save data
from google.colab import files

# Used in build_df()
import warnings

## Hardcoded Variables ##
base_url = "https://4yjz4qhd61.execute-api.us-east-2.amazonaws.com/dev"
key_attributes = ["bank_id", "year", "quarter"]
API_BATCH = 1000

################################################

# Gets the length of the dictionary containing the available codes and their definitions
def get_data_dict_length():
  #Requests the length of the data dictionary
  data_dict_length = requests.get(base_url+"/length/data_dict").json()
  # Returns the length component of the data dictionary length request
  return data_dict_length['length']

# Gets the dictionary containing the available codes and their definitions
def get_data_dictionary():  
  
  # Gets the data dictionary length
  length = get_data_dict_length()

  # Initializes an empty dictionary to store the variables in
  data_dict = []

  # Loops for the length of the dictionary, incrementing by 1000 (the API retrieves 1000 rows of data at a time)
  for start_row in range(0, length + 1, API_BATCH):
    # Requests the next 1000 rows of data
    current_data_dict = requests.get(base_url + "/single/data_dict?start=" + str(start_row)).json()
    # Appends the request to the overall data dictionary
    data_dict.extend(current_data_dict)

  # Converts the overall dictionary into a dataframe
  data_dict_data_frame = pd.DataFrame.from_records(data_dict)

  # This command does nothing, but was used to verify we weren't double counting records
  #data_dict_data_frame.drop_duplicates(inplace = True)

  # Returns a dataframe consisting of the overall dictionary of data code definitions
  return data_dict_data_frame

# Retrieves the codes for n attributes from the api 
# Parameters:
#   - Required: n (int) - the number of codes to retrieve
#   - Optional: start (int) - the starting point in the list/df to start retriving from, defaults to 0
def get_n_attributes(n, start=0):

  # Gets the length of the data dictionary, i.e. the total number of codes available
  length = get_data_dict_length()
  # Gets a copy of the dictionary of attribute codes
  data_dict_data_frame = get_data_dictionary()

  # Extract just the codes column from the codes+definitions df
  data_codes = data_dict_data_frame["item_code"]
  # Turn the codes column into a list for ease of handling
  complete_list_codes = data_codes.tolist()

  # Checks to make sure the number of codes requested does not exceed the number of codes available
  # If it doesn't, grab the codes as normal, if it does warn the user and grab whatever codes are available.
  if(start+n <= length):
    # Returns the codes between the start point and n-observations beyond the start point
    return complete_list_codes[start:start+n]
  else:
    # Calculate how many codes can actually be retrived
    m = length-start
    # Warn the user that they asked for more codes than were available
    warning_msg = "Warning: " + n + " code(s) were requested, but only " + m + " code(s) were available. Returning " + length-start + " code(s)."
    warnings.warn(warning_msg)
    # Return all remaining codes from the start points
    return complete_list_codes[start:]

# Retrieves the codes matching  a passed search criteria. 
# Parameters: 
#   - Required: search_term (str)
def search_codes(search_term):
  # Creates a filter for the desired search term
  my_filter = {"meaning":search_term}

  # Request the length of the data codes found matching the specified search term
  # (The filter makes this request different from get_data_dict_length(), and thus they are not interchangable)
  data_dict_asset_length = requests.post(base_url+"/length/data_dict",json = my_filter).json()
  # Get the length component of the request for length
  data_dict_numeric_length = data_dict_asset_length['length']

  # specifies the specific extention we need to attach to the base url
  specific_url = "/single/data_dict"

  # Initalize an empty overall dictionary to hold the data
  data_dict_asset = []

  # Iterates for the length of the the data, iterating by the API Batch size
  for i in (0, data_dict_numeric_length+1, API_BATCH):
    # Requests the next API Batch size worth of codes from the API given the filter term
    # (The filter makes this request different from get_data_dictionary(), and thus they are not interchangable)
    current_data = requests.post(base_url + specific_url + "?start=" + str(i), json = my_filter).json()
    # Appends the request to the overall data dictionary
    data_dict_asset.extend(current_data)

  # Convert the dictionary to a dataframe
  data_dict_assets_frame = pd.DataFrame.from_records(data_dict_asset)
  # Drop any duplicates from the df (this error might have been corrected? But might as well leave it in just to be safe for now)
  data_dict_assets_frame.drop_duplicates(inplace=True)
  # Return the dataframe
  return data_dict_assets_frame


# Gets the item codes in list form so they can be passed to build_df()
# *This should probably be expanded to be more than just a helper function for specifically search_codes() but currently that's all it is
# Parameters: 
#   - Required: search_term (str)
def query_codes(search_term):
  term_dict_df = search_codes(search_term)
  term_codes = term_dict_df["item_code"].tolist()
  return term_codes

# Gets data pertaining to a passed item code from the API
# Parameters: 
#   - Required: var_name (str)
def retrieve_data(var_name):
  str_name = var_name
  specific_url = "/single/" + str_name + "?start="
  length_url = "/length/" + str_name

  try:
    data_retrieve_length = requests.get(base_url+length_url).json()
    data_numeric_length = data_retrieve_length['length']
  except ValueError:
    warning_msg = "Warning! - " + str_name + " returned 0 values."
    warnings.warn(warning_msg)
    return 

  data_dict = []
  for i in (0, (data_numeric_length // 1000)+1, API_BATCH):
    current_data = requests.get(base_url + specific_url + str(i)).json()
    data_dict.extend(current_data)
      

  retrieval_df = pd.DataFrame.from_records(data_dict)
  return retrieval_df

# Merged a new attribute onto a passed dataframe
# Parameters: 
#   - Required: og_df (pandas dataframe), var_name (str)
def add_attribute(og_df, var_name):
  # Creates a dataframe containing only the variable to be merged (and the standard identifying columns)
  joinable_df = retrieve_data(var_name)

  # Attempt to merge the new 1 variable dataframe with the original. This can return errors if either:
  # a) the variable being added had no data associated with it and thus returned NULL, OR
  # b) the variable lacks one of the fields used to merge on.
  # The latter is more rare so we raise a verbal warning to the user. The former is relatively common so we do not. In either case we return the
  # original dataframe upon failure.
  try:
    # Merge the new dataframe with the old, using an outer join so observations aren't removed if they don't contain every single variable.
    joined_df = og_df.merge(joinable_df, on=key_attributes,how="outer")
    return joined_df
  # In the event 'the variable being added had no data associated with it and thus returned NULL' return the original df
  except TypeError:
    return og_df
  # In the event 'the variable lacks one of the fields used to merge on' return the original df and warn the user.
  except KeyError:
    print("Warning! - ", var_name, " lacks a key attribute field")
    return(og_df)

# Transforms the codes used for accessing data in the API to human readable titles
# *This function could almost certainly be more computationally efficient
# Parameters: 
#   - Required: df (pandas dataframe)
def readable_headers(df):
  # Get a copy of the code definitions
  data_dict_data_frame = get_data_dictionary()
  # Get the column headers (i.e. the codes) in the base dataframe
  column_headers = df.columns
  # Create a new dict for the original header to new header mappings 
  header_mappings = dict.fromkeys(column_headers)
  # Loop through all the headers in the df
  for header in column_headers:
    # Search the overall dictionary for a match for the original header (which should be a code) and get its value (a readable definition)
    result = data_dict_data_frame.loc[data_dict_data_frame['item_code'] == header]['meaning'].values
    # If result is null, map the original header used in the df to itself
    if not result:
      header_mappings[header] = header
    # If the result is not null, map the original header to its definition in the 
    else:
      header_mappings[header] = result[0]
  # Apply the header map
  readable_df = df.rename(columns=header_mappings)
  # Return the df, now with readable headers
  return readable_df

# Assembles a dataframe from the API using a passed list of attribute codes
# Parameters: 
#   - Required: attributes (list)
def build_df(attributes):
  # Initalize df as a none type so its type can be checked in a while loop
  df = None
  # Initialize a counter var to 0 (this counter is used control for if the attempt to initalize a base df fails)
  counter = 0
  # While the df is not initalized (ie its type is None) and the counter hasn't exceed the number of attributes, attempt to initalize a base df
  while type(df) == type(None) and counter < len(attributes):
    # Run retrieve_data() to attempt to initalize the df. On failure this will return as None.
    df = retrieve_data(attributes[counter])
    # Increment the counter
    counter += 1


  # If the loop ended because the passed list of attributes was exceeded before a df could be initalized, warn the user and return nothing
  if(counter >= len(attributes)):
    print("Warning! No id in the passed list of attributes contained data (i.e. the dataframe is empty and there is nothing to return)")
    return
  # If only the last code in the list was successful in initalizing the df, return the df
  elif(counter == len(attributes)):
    # Convert the headers to a human readable format
    df = readable_headers(df)
    return df
  # If the loop ended and there were elements left in the list to try adding, attempt to add them and return that df
  else:
    # Loop for all atributes occuring in the list after the one that succeeded in initalizing the df
    for attribute in attributes[counter+1:]:
      # Attempt the add the current attribute to the df
      df = add_attribute(df, attribute)
    # Convert the headers to a human readable format
    df = readable_headers(df)
  # Return the completed dataframe
  return df