# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|8.13|122.95|427|53|481|3420|427|
|8200|1024|B200 (8)|FP8|8|||8|9.69|103.19|781|97|878|6244|780|
|8197|1024|B200 (8)|FP8|8|||16|12.00|83.36|1247|156|1402|9972|1246|
|8197|1024|B200 (8)|FP8|8|||32|16.05|62.31|1875|234|2110|15002|1874|
|8197|1024|B200 (8)|FP8|8|||64|29.71|33.65|2046|256|2302|16370|2045|
|8198|1024|B200 (8)|FP8|8|||128|52.91|18.90|2252|281|2534|18019|2251|
|8198|1024|B200 (8)|FP8|8|||256|74.13|13.49|2161|270|2431|17286|2159|