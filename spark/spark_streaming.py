# write spark structured streaming to take input data from kafka, and explain step by step
# the input data schema is {"key": str, value: {"timestamp": str; "vec": float [nt], "dt": float}}. The format of key is "network.station.location.channel", the format for timestamp is "YYYY-MM-DDTHH:MM:SS.mmm", and "dt" is in seconds
# step 1: rename "timestamp" to "begin_timestamp", and calculate the "end_timestamp" of the window based on ("begin_timestamp", length of "vec", "dt");
# step 2: use a sliding window to aggregate the data every 1 second of a 30 second window;
# step 3: aggregate the data by key, sort by "begin_timestamp", fill vec with zeros based on gaps in "begin_timestamp", "end_timestamp", and "dt", concatenate the vec data into a single array; keep the smallest begin timestamp and the largest end timestamp
# step 4: aggregate the data by "network" and "station" from "key", sort by "key", stack key in a list, stack vec data in a list, stack timestamp data in a list, calculate the smallest "begin_timestamp", largest "end_timestamp", check if "dt" are the same and select the first "dt", align the vec data to the same length by padding with zeros based on "begin_timestamp", largest "end_timestamp", and "dt"
# step 5: send the data to the phasenet api
