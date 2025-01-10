import os
import pandas as pd
import fsspec
fsspec.core.DEFAULT_EXPAND = True

base_url = (f"s3://dc-taxi-"
            f"{os.environ['BUCKET_ID']}-{os.environ['AWS_DEFAULT_REGION']}/parquet/vacuum/.meta/stats/*")

with fsspec.open(base_url, expand=True) as f:
    df = pd.read_csv(f)
    summary_df = df.set_index('summary')
    print(summary_df)
    ds_size = summary_df.loc['count'].astype(int).max()
    print(ds_size)
    mu = summary_df.loc['mean']
    print(mu)
    sigma = summary_df.loc['stddev']
    print(sigma)
    from math import log, floor

    fractions = [.3, .15, .1, .01, .005]
    ranges = [floor(log(ds_size * fraction, 2)) for fraction in fractions]
    print(ranges)
    sample_size_upper, sample_size_lower = max(ranges) + 1, min(ranges) - 1
    print(sample_size_upper, sample_size_lower)
    sizes = [2 ** i for i in range(sample_size_lower, sample_size_upper)]
    original_sizes = sizes
    fracs = [size / ds_size for size in sizes]
    print(*[(idx, sample_size_lower + idx, frac, size) \
            for idx, (frac, size) in enumerate(zip(fracs, sizes))], sep='\n')
    import numpy as np


    def sem_over_range(lower, upper, mu, sigma):
        sizes_series = pd.Series([2 ** i \
                                  for i in range(lower, upper + 1)])
        est_sem_df = \
            pd.DataFrame(np.outer((1 / np.sqrt(sizes_series)), sigma.values),
                         columns=sigma.index,
                         index=sizes_series.values)
        return est_sem_df


    sem_df = sem_over_range(sample_size_lower, sample_size_upper, mu, sigma)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 9))
    plt.plot(sem_df.index, sem_df.mean(axis=1))
    plt.xticks(sem_df.index,
               labels=list(map(lambda i: f"2^{i}",
                               np.log2(sem_df.index.values).astype(int))),
               rotation=90);
    plt.savefig("plot.png")
    agg_change = sem_df.cumsum().mean(axis=1)
    import numpy as np


    def marginal(x):
        coor = np.vstack([x.index.values,
                          x.values]).transpose()

        return pd.Series(index=x.index,
                         data=np.cross(coor[-1] - coor[0], coor[-1] - coor)
                              / np.linalg.norm(coor[-1] - coor[0])).idxmin()


    SAMPLE_SIZE = marginal(agg_change).astype(int)
    print(SAMPLE_SIZE, SAMPLE_SIZE / ds_size)
    pd.options.display.float_format = '{:,.2f}'.format

    test_stats_df = pd.read_csv(f"s3://dc-taxi-{os.environ['BUCKET_ID']}-{os.environ['AWS_DEFAULT_REGION']}/csv/test/.meta/stats/*.csv")

    test_stats_df = test_stats_df.set_index('summary')
    print(test_stats_df)

    dev_shards_df = pd.read_csv(f"s3://dc-taxi-{os.environ['BUCKET_ID']}-{os.environ['AWS_DEFAULT_REGION']}/csv/dev/.meta/shards/*")
    print(dev_shards_df.sort_values(by='id'))
