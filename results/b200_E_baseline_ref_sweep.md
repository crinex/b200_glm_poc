# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|8.19|122.11|397|50|447|3177|397|
|8200|1024|B200 (8)|FP8|8|||8|9.66|103.51|772|96|868|6174|771|
|8197|1024|B200 (8)|FP8|8|||16|12.18|82.10|1250|156|1407|10004|1250|
|8197|1024|B200 (8)|FP8|8|||32|16.10|62.13|1878|235|2113|15026|1877|
|8197|1024|B200 (8)|FP8|8|||64|30.91|32.35|1949|244|2193|15595|1948|
|8198|1024|B200 (8)|FP8|8|||128|54.89|18.22|2195|274|2469|17558|2193|
|8198|1024|B200 (8)|FP8|8|||256|72.85|13.73|2176|272|2448|17408|2174|