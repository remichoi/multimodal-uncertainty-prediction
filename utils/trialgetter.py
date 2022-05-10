import pandas as pd

def get_timestamps(id):
    """Gets list of timestamps in form (start, stop) for given participant id
    
    Args:
        id (str): response_uuid of participant
    """
    
    # Read in participant's timestamps
    path = '../data/all_ANS_trials_timing.xlsx'
    df = pd.read_excel(path)
    df = df[(df['response_uuid'] == id) & (df['trial_start'] != 'leftb')] # & (df['Task'] == 'ANS')
    
    trial_starts = [float(x) for x in list(df['trial_start'])]
    trial_stops = [float(x) for x in list(df['trial_end'])]
    
    # Adjust by start_time
    start_adjustment = min(trial_starts)
    trial_starts = list(map(lambda x: x - start_adjustment, trial_starts))
    trial_stops = list(map(lambda x: x - start_adjustment, trial_stops))
    
    # Save timestamps as list of tuples
    timestamps = list(zip(trial_starts, trial_stops))

    return timestamps

def get_audio_for_trial(audio_data, trial):
    # TODO: implement
    pass

def get_video_for_trial(video_data, trial):
    return video_data[trial]

def get_text_for_trial(text_data, trial):
    return text_data[trial]

def get_trial_info(id, trial, tag):
    # import all annotation data
    path = '../data/master_annotations.csv'
    df = pd.read_csv(path)
    
    return df[(df['uuid'] == id) & (df['trial'] == trial)][tag]


if __name__ == '__main__':
    test_id = '8326e083-b165-43d0-8037-d9fbec8115d1'
    res = get_timestamps(test_id)
    print(res)
