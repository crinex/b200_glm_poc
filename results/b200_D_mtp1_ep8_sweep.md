# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|9.07|110.23|403|50|454|3226|403|
|8200|1024|B200 (8)|FP8|8|||8|10.23|97.77|736|92|827|5885|735|
|8197|1024|B200 (8)|FP8|8|||16|12.68|78.89|1184|148|1332|9473|1183|
|8197|1024|B200 (8)|FP8|8|||32|17.10|58.49|1779|222|2001|14231|1778|
|8197|1024|B200 (8)|FP8|8|||64|31.89|31.36|1916|239|2155|15327|1915|
|8198|1024|B200 (8)|FP8|8|||128|55.20|18.12|2135|267|2401|17078|2133|
|8198|1024|B200 (8)|FP8|8|||256|74.98|13.34|2079|260|2339|16633|2078|