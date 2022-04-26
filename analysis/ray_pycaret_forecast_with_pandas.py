import time
import pandas as pd
import ray
from parallel_funcs import forecast_single, forecast_single_pmd


# %%
data_subset = pd.read_csv("data/sample_data.csv")


# %% GROUP DATA
grouped_data = data_subset.groupby("Item")

# %% PYTHON SERIAL RUN
start = time.time()
# results = grouped_data.apply(forecast_single)
results = grouped_data.apply(forecast_single_pmd)
# Item is already in dataframe, hence dropping before resetting index
results = results.droplevel(level=0)
results.reset_index(inplace=True)
results.rename(columns={"index": "YYYYMM"}, inplace=True)
end = time.time()
time_python_serial = end - start


# %% RAY RUN
start = time.time()
ray.init()
results_ray = []
for group in grouped_data.groups.keys():
    # result_ray = ray.remote(forecast_single).remote(data=grouped_data.get_group(group))
    result_ray = ray.remote(forecast_single_pmd).remote(
        data=grouped_data.get_group(group)
    )
    results_ray.append(result_ray)
results_ray = ray.get(results_ray)

# Combine all results into 1 dataframe
results_ray = pd.concat(results_ray)
results_ray.reset_index(inplace=True)
results_ray.rename(columns={"index": "YYYYMM"}, inplace=True)

end = time.time()
time_ray = end - start

# %% CHECKS
keys = ["Item", "YYYYMM"]
all_cols = ["Item", "YYYYMM", "y_pred"]
results = results.sort_values(keys)[all_cols]
results_ray = results_ray.sort_values(keys)[all_cols]

all_match = results_ray.equals(results)
if not all_match:
    match = (results_ray == results).all(axis=1)
    python_mismatch = results[~match]
    ray_mismatch = results_ray[~match]
    combined_mismatch = pd.merge(
        left=python_mismatch,
        right=ray_mismatch,
        how="outer",
        on=keys,
    )
    print(
        f"Ray Results that did not match Serial Python Run for following\n{combined_mismatch}"
    )
else:
    print("Ray Results match with Serial Python Run")

# %%
print(f"\n\nTime Python Serial: {time_python_serial}")
print(f"\n\nTime Ray: {time_ray}")

# %%
print("DONE")
