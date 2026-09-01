# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|9.07|110.21|372|47|419|2979|372|
|8200|1024|B200 (8)|FP8|8|||8|10.52|95.02|722|90|812|5776|721|
|8197|1024|B200 (8)|FP8|8|||16|12.95|77.20|1174|147|1320|9391|1173|
|8197|1024|B200 (8)|FP8|8|||32|17.02|58.75|1765|220|1985|14118|1764|
|8197|1024|B200 (8)|FP8|8|||64|32.49|30.78|1832|229|2061|14660|1831|
|8198|1024|B200 (8)|FP8|8|||128|56.16|17.81|2119|265|2384|16956|2118|
|8198|1024|B200 (8)|FP8|8|||256|75.24|13.29|2067|258|2325|16534|2065|