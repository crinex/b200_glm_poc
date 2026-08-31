# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8168|512|B200 (8)|FP8|8|||1|9.46|105.70|197|12|209|1574|99|
|8188|512|B200 (8)|FP8|8|||4|12.00|83.36|613|38|651|4903|307|
|8190|512|B200 (8)|FP8|8|||8|13.34|74.97|1048|66|1114|8387|524|
|8194|512|B200 (8)|FP8|8|||16|17.14|58.34|1631|102|1733|13050|815|
|8198|512|B200 (8)|FP8|8|||32|24.08|41.54|2403|150|2553|19223|1201|
|8197|512|B200 (8)|FP8|8|||64|46.83|21.35|2475|155|2629|19797|1237|
|8198|512|B200 (8)|FP8|8|||128|98.49|10.15|2499|156|2655|19990|1248|
|8199|512|B200 (8)|FP8|8|||256|136.87|7.31|2532|158|2691|20259|1265|