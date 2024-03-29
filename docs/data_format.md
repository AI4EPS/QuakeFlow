# Standard Data Formats of QuakeFlow

- Raw data: 
	- Waveform (MSEED): 
		- Year/Jday/Hour/Network.Station.Location.Channel.mseed
	- Station (xml):
		- Network.Station.xml
	- Events (CSV):
		- colums: time, latitude, longitude, depth_km, magnitude, event_id
	- Picks (CSV)
		- columns: station_id (network.station.location.channel) phase_time, phase_type, phase_score, event_id
- Phase picking:
	- Picks (CSV):
		- columns: station_id (network.station.location.channel) phase_time, phase_type, phase_score, phase_polarity
- Phase association:
	- Events (CSV):
		- colums: time, latitude, longitude, depth_km, magnitude, event_id
	- Picks (CSV):
		- columns: station_id (network.station.location.channel), phase_time, phase_type, phase_score, phase_polarity, event_id
- Earthquake location:
	- Events (CSV):
		- colums: time, latitude, longitude, depth_km, magnitude, event_id
- Earthquake relocation:
	- Events (CSV):
		- colums: time, latitude, longitude, depth_km, magnitude, event_id
- Focal mechanism:
	- Focal mechanism (CSV):
		- columns: strike1, dip1, rake1, strike2, dip2, rake2, event_id