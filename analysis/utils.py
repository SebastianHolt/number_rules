

import pandas as pd
import numpy as np
import math
import re
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats import chisquare

import matplotlib
from matplotlib import pylab, mlab, pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from IPython.core.pylabtools import figsize, getfigs
from IPython.display import display, HTML


# BOTH Experiments 1 AND 2

def get_confint(df, sortbythis, measurethis):
    """Calculates 95% confidence intervals for the mean of a specified column within a dataframe grouped by specified columns."""
    if type(sortbythis) == list:
        grouped = df.groupby(sortbythis)[measurethis].agg(['mean', 'count', 'std'])
    elif type(sortbythis) == str:
        grouped = pd.DataFrame(df.groupby([sortbythis])[measurethis].agg(['mean', 'count', 'std']))
    ci95_hi = []
    ci95_lo = []
    for _, row in grouped.iterrows():
        m, c, s = row['mean'], row['count'], row['std']
        ci95_hi.append(m + 1.96 * s / math.sqrt(c))
        ci95_lo.append(m - 1.96 * s / math.sqrt(c))
    grouped['ci95_hi'] = ci95_hi
    grouped['ci95_lo'] = ci95_lo
    grouped['yerr'] = grouped['ci95_hi'] - grouped['mean']
    return grouped

def get_err(df,by,dv,form='df',hilo=False):
    """    df = the dataframe
           by = group by this factor
           dv = the column you want to plot """
    aggDF = df.groupby([by])[[dv]].agg(['mean','count','std'])    # first aggregate the descriptive stats we need
    Xs,Ys = list(aggDF.index) , list(aggDF[dv,'mean'])            # get lists of groups and means
    yerrs = []                                                    # initialize what will be CI-95% for each group
    for i in aggDF.index:                                         # for each group...
        m, c, s = aggDF.loc[i]                                    # mean, count, sd for that group
        yerrs.append(1.96*s/math.sqrt(c))                         # add CI-95% to the list
    data = {by:Xs, dv:Ys, 'yerr':yerrs}                           # wrap it up into a dictionary
    data = pd.DataFrame(data) if form=='df' else data             # put that into a dataframe if we need to
    if (hilo) & (form=='df'):                                         # if we want the upper and lower bounds of CI-95
        data['hi'] = data[dv] + data['yerr']
        data['lo'] = data[dv] - data['yerr']
    return data                                                   # return a datastructure with X, Y, and y-error to plot

def newFontSize(ax,fs=24):
    """ Takes a list of axes, or one ax, and changes fontsize of all the important labels"""
    if type(ax) == list:
        for a in ax:
            for item in ([a.title, a.xaxis.label, a.yaxis.label] +
                         a.get_xticklabels() + a.get_yticklabels()):
                item.set_fontsize(fs)
    else:
        for item in ([a.title, a.xaxis.label, a.yaxis.label] +
                     a.get_xticklabels() + a.get_yticklabels()):
            item.set_fontsize(fs)

























# Experiment 1 ONLY
def convert_value(value):
    try:
        # If the value is already a number, return it as-is
        return int(value)
    except ValueError:
        # Handle strings based on the rules you described
        if isinstance(value, str):
            # Rule 1: If string contains only letters or ends with '+' sign
            if re.match(r'^[a-zA-Z]+$|^\d+\+[a-zA-Z]*$', value):
                return 99
            # Rule 2: If string contains 'g' or 'b' followed by some number
            elif re.match(r'[gb](\d+)', value):
                match = re.search(r'\d+', value)
                return int(match.group(0)) if match else 99
            # Rule 3: If string contains 'a' followed by some number
            elif re.match(r'a\d+', value):
                return 99
        # If it can't be converted, print the value and return it unchanged
#         print(f"Cannot convert value: {value}")
        elif pd.isna(value):
            return 0
        return value

def process_dataframe(df, dv_cols):
    # Apply the convert_value function to the specified columns
    df[dv_cols] = df[dv_cols].applymap(convert_value)
    return df


def verticalize(hdf, id_cols, dv_cols):
    # First pivot the horizontal into a vertical format
    df = pd.melt(hdf, id_vars=id_cols, value_vars=dv_cols, var_name='trial', value_name='response')
    
    # Now add new columns that will be helpful. First is to help index trials
    df['trialType'] = df.apply(lambda x:  x['trial'][0] ,axis=1)
    df['sectionTrial'] = df.apply(lambda x:  x['trial'][1:] ,axis=1)
    
    # Then we'll look at accuracy. What counts as a correct response for each question?
    acc_dict = {"s1":2,"s2":4,"s3":3,"s4":2,"s5":4,"s6":3,"s7":2,"s8":4,"s9":3,
                "l1":11,"l2":12,"l3":13,"l4":14,"l5":15,"l6":16,"l7":17,"l8":18,"l9":19,
                "m1":2,"m2":4,"m3":2,"m4":4,"m5":2,"m6":4,"m7":2,"m8":4}
    
    # Use above dictionary to add the expected response, and whether child's response matched it
    df['expected'] = df.apply(lambda x:  str(acc_dict[x['trial']]) ,axis=1)
    df['correct'] = df.apply(lambda x:  str(acc_dict[x['trial']]) == str(x['response']) ,axis=1)
    
    # Convert column types as needed
    df = df.astype({'sectionTrial':'float','expected':'float','response':'Int64'}).astype({'sectionTrial':'Int64','expected':'Int64'})
    
    # For subset-knowers, was the target quantity within their knower level?
    df['suf'] = df[['CP','hc','kl_numeric','expected']].apply(lambda x: (x.hc >= x.expected if type(x.hc) == int else np.nan) if 
                                                              x['CP']==True else x['kl_numeric']>=x['expected'], axis=1)
    
    # Lump all the big numbers into one category. Code non-responses as 0, then code big numbers as 99
    df['resp'] = df['response'].astype(float).fillna(0).astype(int)
    df['resp'] = df['resp'    ].apply(lambda x: 99 if x > 6 else x)
    
    df['response_trunc'] = df['response'].astype(float).fillna(0).astype(int).astype(str)

    df['response_trunc'] = df.apply(lambda x: int(x['response_trunc']) if ((int(x['response_trunc']) < 7) & (x['trialType']!='l')) else (int(x['response_trunc']) if (x['trialType']=='l') else 0), axis=1)
    
    # Lastly, drop meaningless Large Sets rows for subset knowers
    df = df.drop(df.loc[(df['CP']==False)&(df['trialType']=='l')].index)
    
    
    
    return df


























# Experiment 2 ONLY

def read_csv_correcting_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the first line to get the header
        header = file.readline().strip().split(',')
        
        # Assuming "comments" is the last column, determine its index
        if 'comments' in header:
            comments_index = header.index('comments')
        else:
            comments_index = header.index('utterances')
            
        # Read the rest of the file
        data_lines = file.readlines()
        
        # Correct the lines
        corrected_lines = []
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) > len(header):
                comments = ','.join(parts[comments_index:])
                corrected_line = parts[:comments_index] + [comments]
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(parts)
    
    # Creating a DataFrame
    df = pd.DataFrame(corrected_lines, columns=header)
    
    if 'utterances' in header:
        df = df.rename(columns={'utterances':'comments'})
    return df

# making request text less wordy
req_abbr_dict = {
    'Can you put one fish on the plate?' :                         'put_one',
    'Can you put two fish on the plate?' :                         'put_two',
    'Can you put three fish on the plate?' :                       'put_three',
    'Can you put four fish on the plate?' :                        'put_four',
    'Can you put five fish on the plate?' :                        'put_five',
    'Can you put six fish on the plate?' :                         'put_six',
    'Can you put seven fish on the plate?' :                       'put_seven',
    'Now its your turn to make... zo bananas?' :                   'make_zo',
    'Now its your turn to make... gi banana?' :                    'make_gi',
    'How fast can you make... zo bananas?' :                       'fast_zo',
    'How fast can you make... gi banana?' :                        'fast_gi',
    'Another Monkey number is... zozo bananas.' :                  'zozo1', #'new_zozo',
    'Another Monkey number is... zogi bananas.' :                  'zogi1', #'new_zogi',
    'Can you show me zogi bananas again?' :                        'zogi2', #'old_zogi',
    'Can you show me zozo bananas again?' :                        'zozo2', #'old_zozo',
    'Giraffe: "Whats the number gi banana look like?"' :           'G_gi',
    'Giraffe: "Whats the number zo bananas look like?"' :          'G_zo',
    'Giraffe: "Whats the number zogi bananas look like?"' :        'zogi3', #'G_zogi',
    'Giraffe: "Whats the number zozo bananas look like?"' :        'zozo3', #'G_zozo',
    'Another Giraffe number is... three-two bananas.' :            'three-two',
    'Another Giraffe number is... three-one bananas.' :            'three-one',
    'Another Giraffe number is... three-three bananas.' :          'three-three',
    }

giraffe_dict = {'the number gi bana':'G_gi', 'the number zo bana':'G_zo', 'the number zogi bana':'zogi3', 'the number zozo bana':'zozo3'}
def abbreviateRequest(longthing, silence=False):
    if longthing in req_abbr_dict.keys():
        abbreviated = req_abbr_dict[longthing]
    
    elif ('the number zozo' in longthing):
        abbreviated = giraffe_dict['the number zozo bana']
    elif ('the number zogi' in longthing):
        abbreviated = giraffe_dict['the number zogi bana']
    elif ('the number zo' in longthing):
        abbreviated = giraffe_dict['the number zo bana']
    elif ('the number gi' in longthing):
        abbreviated = giraffe_dict['the number gi bana']
    elif longthing in req_abbr_dict.values():
        abbreviated = longthing
    else:
        print("Questioned: ", longthing )
        abbreviated = 'questionable'
    return abbreviated


def getProgress(gameid, trialset, complete_trial_set):
    # a few different scenarios that we expect to encounter
    failed_demo = {'make_zo', 'make_gi'}
    failed_training_noGiraffe = {'make_zo', 'fast_zo', 'make_gi', 'fast_gi'}
    failed_training_yesGiraffe = {'three-one', 'fast_gi', 'fast_zo', 'three-two', 'make_zo', 'three-three', 'make_gi'}
    
    if (failed_demo.difference(trialset) == set()) & (trialset.difference(failed_demo) == set()):
        gametype = "Demo"
    elif (failed_training_noGiraffe.difference(trialset) == set()) & (trialset.difference(failed_training_noGiraffe) == set()):
        gametype = "Training Only"
    elif (failed_training_yesGiraffe.difference(trialset) == set()) & (trialset.difference(failed_training_yesGiraffe) == set()):
        gametype = "Training + Giraffe"
    elif (complete_trial_set.difference(trialset) == set()) & (trialset.difference(complete_trial_set) == set()):
        gametype = "Complete"
    else:
        gametype = complete_trial_set.difference(trialset)
        print(gameid, "missing:  ", gametype)


        
def getStrategy(DF):
    df = DF.copy()
    
    # Clean up the columns in this df to be easy to analyze
    df['request'] = df['request'].apply(lambda x: abbreviateRequest(x, silence=True))
    df['grab'] = df['grab'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['onebyone'] = df['onebyone'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['count'] = df['count'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['response'] = df['response'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    
    df['group1'] = df['group1'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['group2'] = df['group2'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['group3'] = df['group3'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    df['more'] = df['more'].apply(pd.to_numeric, errors='coerce').astype('Int64').fillna(0, downcast='infer')
    
    for word in ['one', 'two', 'three', 'four', 'five', 'six', 'gi', 'zo', 'AND', 'bignumber', 'distracted', 'question', 'parent', 'prompt', 'other']:
        df[word] = df[word].astype(bool)
    
    # Define some new columns we will add
    df['action_sequence'] = ""    # a coded summary of all buttons pressed and their meaning
    df['ngroups'] = 0              # the state of the board
    df['strat_sequence'] = ""     # records the sequence of different strategies (capital letters: G B C), with iconic dots for each banana it applies to (.)
    df['comments_list'] = ""      # records comments made during that trial
    df['kid_words'] = ""          # words that the kid said
    df['temporal_groups'] = ""      # how many times either (a) strategy changed or (b) strategy applied to multiple objects before being applied again
    # df['game_trial'] = ""        # we now have trial_ind in the webpage code, but this one is both retroactive and also looks only at chronology. Must write it
    
    # and any data structures we're going to put in those columns
    act_seq = ""
    strat_seq = ""
    comment_seq = ""
    temporal_groups = 0
    kid_words = ""
    
    prev_row = None
    prev_strat = None
    
    # classification dictionary (define functions for the more complex ones)
    actions = {'grab':       {'tally':0,'code':'g'},
               'onebyone':   {'tally':0,'code':'b'},
               'count':      {'tally':0,'code':'c'},
               
               'group1':     {'tally':0,'code':'γ'},
               'group2':     {'tally':0,'code':'δ'},
               'group3':     {'tally':0,'code':'ε'},
               'more':       {'tally':0,'code':'η'},
               
               'one':     {'tally':False,'code':'1'},
               'two':     {'tally':False,'code':'2'},
               'three':     {'tally':False,'code':'3'},
               'four':       {'tally':False,'code':'4'},
               'five':     {'tally':False,'code':'5'},
               'six':       {'tally':False,'code':'6'},
               'gi':     {'tally':False,'code':'gi'},
               'zo':       {'tally':False,'code':'zo'},
               'AND':       {'tally':False,'code':'&'},
               'bignumber':       {'tally':False,'code':'7'},
               
               'distracted':       {'tally':False,'code':'d'},
               'question':       {'tally':False,'code':'q'},
               'parent':       {'tally':False,'code':'i'},
               'prompt':       {'tally':False,'code':'p'},
               'other':       {'tally':False,'code':'o'},
               
               'kidaction':       {'tally':False,'code':'k'},
               'eyes':       {'tally':False,'code':'e'},
               'flagged':       {'tally':False,'code':'f'},
               'comments':       {'tally':False,'code':''},
               
               'spontaneousCounting':       {'tally':False,'code':'S'},
               'right':       {'tally':0,'code':'r'},
               'logic':       {'tally':0,'code':'l'},
               'words':       {'tally':0,'code':'w'},
              }
    
    for i,row in df.iterrows():
        # First, check for INCREASED values
        for action in ['grab','onebyone','count','group1','group2','group3','more','one','two','three','four','five','six','gi','zo','AND','bignumber','distracted','question','parent','prompt','other']:
            if row[action] > actions[action]['tally']:
                actions[action]['tally'] = row[action]
                act_seq = act_seq + " " + actions[action]['code']
                
                # If the kid said a word, then add it to the list of words
                if action in ['one','two','three','four','five','six','gi','zo','AND','bignumber']:
                    print("Kid said word")  # This is not currently working, must figure out why
                    kid_words = kid_words + "_" + actions[action]['code']
            
                # If the increased value is a strategy, ask whether it's a change, then add a . to the strat_seq
                if action in ['grab','onebyone','count']:
                    if (action != prev_strat):
                        strat_code = actions[action]['code'].upper()
                        strat_seq = strat_seq + strat_code
                        prev_strat = action  # update previous strategy using the dataframe column value
                        temporal_groups += 1
                        
                    strat_seq = strat_seq + "."

        # Then, check for DECREASED values
        for action in ['group1','group2','group3','more', 'grab','onebyone','count']:
            if row[action] < actions[action]['tally']:
                actions[action]['tally'] = row[action]
                act_seq = act_seq + " -" + actions[action]['code']
                
                # If the decreased value is a strategy, then remove the last . from the string, as well as any strat code that immediately precedes it
                if action in ['grab','onebyone','count']:
                    strat_seq = strat_seq[:-1]
                    if strat_seq[-1] in ['G','B','C']:
                        strat_seq = strat_seq[:-1]
                        temporal_groups -= 1
                        
        # Have we detected too many changes at once?
        if ((row['grab'] > actions['grab']['tally']) + (row['onebyone'] > actions['onebyone']['tally']) + (row['count'] > actions['count']['tally'])) >=2:
            print("Two or more things changed at once!")
        
        prev_row = row    # update our memory of the previous row
        # Process groupings here
        groups = np.count_nonzero([actions['group1']['tally'], actions['group2']['tally'], actions['group3']['tally'], actions['more']['tally']]) 
        grouping = ""
        
        last_resp = row['response']
        
        if row['eventType'] == 'trial':
            
            # store derived measures
            df.loc[i,'action_sequence'] = act_seq
            df.loc[i,'strat_sequence'] = strat_seq
            df.loc[i,'ngroups'] = groups
            df.loc[i,'comments_list'] = comment_seq
            df.loc[i,'kid_words'] = kid_words
            df.loc[i,'temporal_groups'] = temporal_groups
            
            
            # reset things for next trial
            act_seq = ""
            strat_seq = ""
            prev_row = None
            comment_seq = ""
            kid_words = ""
            prev_strat = None
            temporal_groups = 0
            for action in actions.keys():
                actions[action]['tally'] = False
            
    return df.loc[df['eventType'].isin(['trial','active'])]

