# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|14.08|71.05|259|32|291|2069|259|
|8200|1024|B200 (8)|FP8|8|||8|16.25|61.54|463|58|521|3707|463|
|8197|1024|B200 (8)|FP8|8|||16|20.43|48.94|725|91|816|5802|725|
|8197|1024|B200 (8)|FP8|8|||32|30.54|32.74|986|123|1109|7887|985|
|8197|1024|B200 (8)|FP8|8|||64|45.17|22.14|1387|173|1561|11099|1387|
|8198|1024|B200 (8)|FP8|8|||128|65.55|15.26|1855|232|2087|14843|1854|
|8198|1024|B200 (8)|FP8|8|||256|86.91|11.51|1844|230|2074|14752|1843|