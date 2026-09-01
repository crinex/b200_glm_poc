# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|10.25|97.60|307|38|345|2455|307|
|8200|1024|B200 (8)|FP8|8|||8|12.70|78.71|602|75|677|4816|601|
|8197|1024|B200 (8)|FP8|8|||16|19.04|52.53|789|99|887|6308|788|
|8197|1024|B200 (8)|FP8|8|||32|22.93|43.60|1327|166|1493|10620|1327|
|8197|1024|B200 (8)|FP8|8|||64|35.90|27.86|1713|214|1927|13703|1712|
|8198|1024|B200 (8)|FP8|8|||128|60.23|16.60|2015|252|2267|16119|2013|
|8198|1024|B200 (8)|FP8|8|||256|74.87|13.36|1996|249|2245|15965|1994|