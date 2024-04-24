# %%
import io
import multiprocessing as mp
import os

import fsspec
from obspy import read
from tqdm import tqdm


# %%
def process_url(url):
    with fsspec.open(url, mode="rb") as fp:
        content = fp.read()

        file_like_object = io.BytesIO(content)

        st = read(file_like_object)


# %%
if __name__ == "__main__":
    # %%
    # https://constantine.seis.utah.edu/datasets.html
    os.system("curl -o urls.txt https://constantine.seis.utah.edu/files/get_all_slb.sh")

    # %%
    urls = []
    with open("urls.txt") as f:
        for line in f:
            if line.startswith("wget"):
                urls.append(line.split()[-1])

    # # %%
    # for url in tqdm(urls):
    #     with fsspec.open(url, mode="rb") as fp:
    #         content = fp.read()

    #         file_like_object = io.BytesIO(content)

    #         st = read(file_like_object)
    #         # print(st)
    #         # raise

    # %%
    ncpu = mp.cpu_count() * 2
    print(f"Number of CPUs: {ncpu}")
    pbar = tqdm(total=len(urls))
    with mp.Pool(ncpu) as pool:
        for url in urls:
            pool.apply_async(
                func=process_url,
                args=(url,),
                callback=lambda _: pbar.update(1),
                error_callback=lambda e: print(e),
            )
        pool.close()
        pool.join()

# %%
