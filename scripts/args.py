import argparse

## Where each CCTorch mode works: (folder, prefix for the template tables it writes there).
## The convention across the repo is that a folder is named for the stage and its <stage>_*
## files are that stage's OUTPUT -- adloc/adloc_events.csv, gamma/gamma_events.csv -- so a
## stage's inputs are template_*, leaving <stage>_* free for what it produces:
##     {region}/qtm/template_{picks,events,stations}.csv   in
##     {region}/qtm/qtm_{events,picks}.csv                 out
## CC is the exception, kept as-is because the relocation chain (run_cctorch, run_adloc_cc,
## run_growclust_cc, run_hypodd_cc, convert_dtcc) reads cctorch_*.csv from that path.
## Adding a mode is one entry; unlisted modes get ({mode.lower()}, "template").
MODE_LAYOUT = {
    "CC": ("cctorch", "cctorch"),  # differential travel times for relocation (legacy names)
    "TM": ("qtm", "template"),  # template matching detection
}


def mode_layout(mode):
    """(folder, template-table prefix) for a CCTorch mode"""
    return MODE_LAYOUT.get(mode, (mode.lower(), "template"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="demo", help="region")
    parser.add_argument("--root_path", type=str, default="local", help="root path")

    ## Cloud
    parser.add_argument("--protocol", type=str, default="file", help="protocol (file, gs, s3)")
    parser.add_argument("--bucket", type=str, default=None, help="bucket name")
    parser.add_argument("--token", type=str, default=None, help="token")

    # parser.add_argument("--bucket", type=str, default="quakeflow_catalog", help="bucket name")
    # parser.add_argument("--protocol", type=str, default="gs", help="protocol (file, gs, s3)")
    # parser.add_argument("--token", type=str, default="application_default_credentials.json", help="token")

    ## Parallel
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="node rank")

    ## Model
    parser.add_argument("--model", type=str, default="phasenet", help="model")

    ## PhaseNet
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")

    ## ADLOC
    parser.add_argument("--iter", type=int, default=0, help="iteration")

    ## CCTorch
    parser.add_argument("--dtct_pair", action="store_true", help="run convert_dtcc.py")
    parser.add_argument(
        "--mode",
        type=str,
        default="CC",
        choices=sorted(MODE_LAYOUT),
        help="what the templates are for: CC (differential times) or TM (template matching)",
    )
    parser.add_argument("--min_cc_score", type=float, default=0.5,
                        help="absolute cc floor for a detection")
    parser.add_argument("--nmad", type=float, default=None,
                        help="also require cc > nmad * median(|cc|) of the trace, the adaptive threshold of "
                             "Shelly et al. (2007), who use 8-9. Defaults to 9 for --mode TM and 0 (off) for "
                             "--mode CC, since relocation reads the correlations directly. 0 disables it.")

    return parser.parse_args()
