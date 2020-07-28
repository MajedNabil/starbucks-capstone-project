import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import datetime

def clear_portfolio_df(portfolio):
    '''
    Input: 
        - portfolio: A dataframe that represents the portfolio dataframe from Starbucks 
        
    Output: 
        - df: A cleaned version of the portfolio dataframe 
    '''
    
    # copy the content of the old dataframe to a new one
    df = portfolio.copy()
    # rename from `id` to `offer_id`
    df.rename(columns={'id':'offer_id'},inplace=True)
    # convet the duration from days to hours 
    df['duration'] = df['duration']*24
    df.rename(columns={'duration' : 'duration_h'}, inplace=True)
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['difficulty','reward']
    #features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    df[numerical] = scaler.fit_transform(df[numerical])
    # Let's decondense the `channel` column, and create 4 additional columns 
    df['channel_email'] = df['channels'].apply(lambda x: 1 if 'email' in x else 0)
    df['channel_mobile'] = df['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
    df['channel_social'] = df['channels'].apply(lambda x: 1 if 'social' in x else 0)
    df['channel_web'] = df['channels'].apply(lambda x: 1 if 'web' in x else 0)
    # There's no need in keepin the old `channel` column 
    df.drop('channels', axis=1, inplace=True)
    # replacing the 'offer_id' by more easy ids
    offer_ids = df['offer_id'].astype('category').cat.categories.tolist()
    temp = {'offer_id' : {k: v for k,v in zip(offer_ids,list(range(1,len(offer_ids)+1)))}}
    # replacing the categorical values in the 'offer_id' column by numberical values
    df.replace(temp, inplace=True)
    
    labels_offer_type = df['offer_type'].astype('category').cat.categories.tolist()
    replace_map_comp_offer_type = {'offer_type' : {k: v for k,v in zip(labels_offer_type,list(range(1,len(labels_offer_type)+1)))}}
    df.replace(replace_map_comp_offer_type, inplace=True)
    return df 


def clear_profile_df(profile):
    '''
    Input: 
        - profile: A dataframe that represents the profile dataframe from Starbucks 
    
    Output: 
        - df: A cleaned version of the profile dataframe 
    '''
    df = profile.copy()
    # rename the column from `id` to `customer_id`
    df.rename(columns={'id':'customer_id'},inplace=True)
    # replacing the 'customer_id' string values  with easier values 
    customer_ids = df['customer_id'].astype('category').cat.categories.tolist()
    temp = {'customer_id' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}
    
    # replacing the  categorical labels in 'customer_id' column with the newly numerical labels
    df.replace(temp, inplace=True)
    # replacing the age = 118 by NaN value
    df['age'] = df['age'].apply(lambda x: np.nan if x == 118 else x)
    # drop NaNs 
    df.dropna(inplace=True)
    # changing the datatype of 'age' and 'income' columns to 'int' for later processing
    df[['age','income']] = df[['age','income']].astype(int)
    # creating a new column representing the age group (to conver it later on to numerical values)
    df['age_group'] = pd.cut(df['age'], bins=[17, 22, 35, 60, 103],labels=['teenager', 'young-adult', 'adult', 'elderly'])
    # replacing the 'age_group' categorical labels by numerical labels
    age_group = df['age_group'].astype('category').cat.categories.tolist()
    temp_age_group = {'age_group' : {k: v for k,v in zip(age_group,list(range(1,len(age_group)+1)))}}
    df.replace(temp_age_group, inplace=True)
    # As we did with the `group_age`, we'll do the same withe income 
    df['income_range'] = pd.cut(df['income'], bins=[29999, 60000, 90000, 120001],labels=['average', 'above-average', 'high'])
    # replacing the 'income_range' categorical labels by numerical labels
    income_range = df['income_range'].astype('category').cat.categories.tolist()
    temp_income_range = {'income_range' : {k: v for k,v in zip(income_range,list(range(1,len(income_range)+1)))}}
    df.replace(temp_income_range, inplace=True)
    # Do the same thing with the `gender` 
    labels_gender = df['gender'].astype('category').cat.categories.tolist()
    temp_gender = {'gender' : {k: v for k,v in zip(labels_gender,list(range(1,len(labels_gender)+1)))}}
    df.replace(temp_gender, inplace=True)
    # Finally, let's process the `became_member_on`
    df['became_member_on'] = pd.to_datetime(df['became_member_on'], format = '%Y%m%d')
    df['membership_year'] = df['became_member_on'].dt.year
    # compute the number of membership days, and sotre them in a brand-new column 
    df['membership_days'] = datetime.datetime.today().date() - df['became_member_on'].dt.date
    # removing the 'days' unit
    df['membership_days'] = df['membership_days'].dt.days
    # creating a new column 'member_type' representing the type of the member
    df['member_type'] = pd.cut(df['membership_days'], bins=[390, 1000, 1600, 2500],labels=['new', 'regular', 'loyal'])
    member_type = df['member_type'].astype('category').cat.categories.tolist()
    temp_member_type = {'member_type' : {k: v for k,v in zip(member_type,list(range(1,len(member_type)+1)))}}
    # replacing categorical labels in 'member_type' column with numerical labels
    df.replace(temp_member_type, inplace=True)
    # drop the unneeded columns 
    df.drop(columns = ['age','income','became_member_on', 'membership_days'], axis=1, inplace=True)
    
    return df 



def clear_transcript_df(transcript, clean_profile, clean_portfolio):
    '''
    Input: 
        - transcript: A dataframe that represents the transcript dataframe from Starbucks 
        - clean_profile: A cleaned and wrangled version of the profile dataframe 
        - clean_portfolio: A cleaned and wrangled version of the portfolio dataframe 
    
    Output: 
        - df: A cleaned version of the transcript dataframe 
    '''
    df = transcript.copy()
    # rename the `time` column to a more representative name 
    df.rename(columns={'time':'time_h'},inplace=True)
    # rename the `person` column to a more representatieve name
    df.rename(columns={'person':'customer_id'},inplace=True)
    # adjust the `customer_id` accordingly 
    customer_ids = clean_profile['id'].astype('category').cat.categories.tolist()
    temp_customer_id = {'customer_id' : {k: v for k,v in zip(customer_ids,list(range(1,len(customer_ids)+1)))}}
    df.replace(temp_customer_id, inplace=True)
    
    # create new columns (the `value` column previously)
    df['offer_id'] = '' # datatype : string
    df['amount'] = 0  # datatype : integer
    df['reward'] = 0  
    
    #extract the values from this `value` column, and then put it in the new columns accordingly 
    for idx, row in df.iterrows():
        for k in row['value']:
            if k == 'offer_id' or k == 'offer id': # b/c 'offer_id' and 'offer id' are representing the same thing 
                df.at[idx, 'offer_id'] = row['value'][k]
            if k == 'amount':
                df.at[idx, 'amount'] = row['value'][k]
            if k == 'reward':
                df.at[idx, 'reward'] = row['value'][k]
    # convert from NaN to N/A 
    df['offer_id'] = df['offer_id'].apply(lambda x: 'N/A' if x == '' else x)
    
    # dropping the 'value' column, since it's useless now 
    df.drop('value', axis=1, inplace=True)
    # excluding all events of 'transaction' 
    df = df[df['event'] != 'transaction']
    # excluding all events of 'offer received' 
    df = df[df['event'] != 'offer received']
    # After I removed the irrelevant values, it's time to convert the exsiting to numerical categories 
    labels_event = df['event'].astype('category').cat.categories.tolist()
    comp_event = {'event' : {k: v for k,v in zip(labels_event,list(range(1,len(labels_event)+1)))}}
    df.replace(comp_event, inplace=True)
    
    # It's time to relpace the `offer_id`, as I have done with the `portfolio` dataframe 
    
    offer_ids = {'offer_id': {'0b1e1539f2cc45b7b9fa7c272da2e1d7': 1,
      '2298d6c36e964ae4a3e7e9706d1fb8c2': 2,
      '2906b810c7d4411798c6938adc9daaa5': 3,
      '3f207df678b143eea3cee63160fa8bed': 4,
      '4d5c57ea9a6940dd891ad53e9dbe8da0': 5,
      '5a8bc65990b245e5a138643cd4eb9837': 6,
      '9b98b8c7a33c4b65b9aebfe6a799e6d9': 7,
      'ae264e3637204a6fb9bb56bc8210ddfd': 8,
      'f19421c1d4aa40978ebb69ca19b0e20d': 9,
      'fafdcd668e3743c1bb461111dcafc2a4': 10}}

    df.replace(offer_ids, inplace=True)
    
    return df






    
    
    
