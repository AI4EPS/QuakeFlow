# %%
import sky
from tqdm import tqdm

# %%
status = sky.status()

# %%
for cluster in tqdm(sky.status()[::-1]):
    try:
        print(f"Refreshing {cluster['name']}...")
        sky.status(cluster_names=[cluster["name"]], refresh=True)
        if not cluster["to_down"]:
            sky.autostop(cluster["name"], idle_minutes=10, down=True)
    except Exception as e:
        print(e)

# %%
