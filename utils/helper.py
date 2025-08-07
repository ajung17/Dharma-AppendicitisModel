from sklearn.model_selection import train_test_split

def split_data(df, random_state=None):
    if random_state is None:
        random_state = 88
    temp_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_df, val_df = train_test_split(temp_df, test_size=0.25, random_state=random_state)
    return train_df, val_df, test_df