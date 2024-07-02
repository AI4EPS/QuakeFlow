# %%
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# %%
cmds = [
    "python submit_phasenet.py --region NC --branch ncedc --year 2022",
    "python submit_phasenet.py --region SC --branch scedc --year 2022",
]

while True:
    for cmd in cmds:
        logging.info(f"Running: {cmd}")
        os.system(cmd)
        logging.info("Sleeping for 1 minutes...")
        time.sleep(60)
