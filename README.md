# Taxi-Trip-Data-Analysis-Based-on-Spark

• Job: Predicted the busiest time and area for taxis in Chengdu, China

• Technology: Scala, Spark, Spark Notebook, Linux

Data set: 
Record the location and time when a taxi picks up passengers.

Data type:
TID, Lat, Lon, Time
1,30.624806,104.136604,211846

TID: A unique taxi id
Lat: The latitude of taxi at this time
Lon: The longitude of taxi at this time
Time: the record time. Type: hhmmss, eg. 211846  21:18:46

I. Do cluster by K-means based on latitude and longitude. 

II. Predict the busiest time and area for taxis 

2.1 Count the number for each hour and each group (for <hour, group> pair).

2.2 Sort result by counting number in descending order for each <hour, group> pair

Then we can find at what time and at what area taxis pick up most passengers. Also, we can predict at what time taxis pick up most passengers by counting hour and sort it, and predict at what area taxis pick up most passengers by counting clustered location and sort it.





