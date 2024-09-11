import pandas as pd

def read_label(label_path):
    df = pd.read_csv(label_path, header=None)
    keys = df.iloc[0, 2:] 
    mapped_values = {}
    for index, row in df.iterrows():
        if index == 0:
            continue  # Skip the first row
        img_name = row[0]
        for i in range(len(keys)):
            if float(row[i+2]) == 1:
                mapped_values[img_name] = keys[i + 2] 
    
       
    return mapped_values


if __name__ == "__main__":
    label_path = "data/Sekiro/session_1/label.csv"
    df = read_label(label_path)
    print(df.head())    
