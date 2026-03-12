import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}")
import torch
import numpy as np
import fire


def main(
    file1="work_dirs/test_video_submit_fvd/fvd_feats_ori.npy",
    file2="work_dirs/test_video_submit_fvd/fvd_feats_gen.npy",
    method="videogpt"
):
    d1 = np.load(file1)
    d2 = np.load(file2)

    if method == 'styleganv':
        from fvd.styleganv.fvd import frechet_distance
    elif method == 'videogpt':
        from fvd.videogpt.fvd import frechet_distance

    fvd = frechet_distance(torch.from_numpy(d1), torch.from_numpy(d2))
    print(f"FVD score: {fvd}")


if __name__ == "__main__":
    fire.Fire(main)
