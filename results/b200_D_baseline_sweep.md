# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|11.22|89.15|309|39|347|2471|309|
|8200|1024|B200 (8)|FP8|8|||8|13.05|76.66|572|71|643|4575|571|
|8197|1024|B200 (8)|FP8|8|||16|15.79|63.35|927|116|1043|7419|927|
|8197|1024|B200 (8)|FP8|8|||32|20.45|48.91|1411|176|1587|11288|1410|
|8197|1024|B200 (8)|FP8|8|||64|32.33|30.93|1768|221|1989|14147|1767|
|8198|1024|B200 (8)|FP8|8|||128|61.02|16.39|2004|250|2255|16034|2003|
|8198|1024|B200 (8)|FP8|8|||256|83.88|11.92|2001|250|2251|16006|1999|