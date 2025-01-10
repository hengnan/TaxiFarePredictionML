import pandas as pd


def awscli_to_df():
    json_df = pd.read_json('/tmp/awscli.json')
    df = pd.DataFrame(json_df[0].tolist(), index=json_df.index,
                      columns=json_df[0].tolist()[0]).drop(0, axis=0)
    return df


if __name__ == "__main__":
    print(awscli_to_df())
