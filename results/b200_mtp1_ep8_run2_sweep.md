# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|12.23|81.79|298|37|335|2381|297|
|8200|1024|B200 (8)|FP8|8|||8|13.50|74.08|560|70|630|4480|559|
|8197|1024|B200 (8)|FP8|8|||16|18.29|54.67|846|106|952|6767|845|
|8197|1024|B200 (8)|FP8|8|||32|22.64|44.17|1329|166|1495|10634|1328|
|8197|1024|B200 (8)|FP8|8|||64|38.20|26.18|1588|198|1787|12706|1587|
|8198|1024|B200 (8)|FP8|8|||128|66.84|14.96|1784|223|2007|14272|1783|
|8198|1024|B200 (8)|FP8|8|||256|87.97|11.37|1730|216|1946|13843|1729|