# B200 × GLM-5.2 FP8 벤치마크 결과

|ISL|OSL|GPU|Precision|TP|p|d|Concurrency|TPOT(ms)|Interactivity (Token/sec/user)|Input Token Throughput per GPU (Token/sec/gpu)|Output Token Throughput per GPU (Token/sec/gpu)|Total Token Throughput per GPU (Token/sec/gpu)|Input Token Throughput per server (Token/sec/server)|Output Token Throughput per server (Token/sec/server)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8197|1024|B200 (8)|FP8|8|||4|13.59|73.59|266|33|299|2129|266|
|8200|1024|B200 (8)|FP8|8|||8|15.65|63.89|490|61|551|3917|489|
|8197|1024|B200 (8)|FP8|8|||16|18.38|54.40|799|100|899|6390|798|
|8197|1024|B200 (8)|FP8|8|||32|24.13|41.44|1250|156|1406|9997|1249|
|8197|1024|B200 (8)|FP8|8|||64|32.51|30.76|1884|235|2120|15076|1883|
|8198|1024|B200 (8)|FP8|8|||128|48.85|20.47|2496|312|2807|19964|2494|
|8198|1024|B200 (8)|FP8|8|||256|75.26|13.29|3190|399|3589|25523|3188|