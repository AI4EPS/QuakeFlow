"""Reading a channel file the way the workflow does."""

import obspy


def read_trace(path, format="MSEED"):
    """The whole day as one trace, not just its first segment.

    obspy.read(...)[0] returns the first *segment*. That is harmless on an archive of
    contiguous day files, which is why it survived here, but 41% of the Redwood Valley
    channel files are fragmented -- up to 24 segments -- and on those it silently keeps as
    little as 14% of the day. Merging first mirrors what CCTorch's read_mseed does, so the
    benchmark sees the same samples the pipeline does.
    """
    return obspy.read(path, format=format).merge(fill_value="latest")[0]
