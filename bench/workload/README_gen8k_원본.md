#README.md

# gen8k 워크로드 만들기 + sweep 돌리기

## 1. gen8k 이 무엇인가

**8k1k 벤치에 쓰는 실제 텍스트 프롬프트 모음.** LG U+ 원본과 같은 방식으로 만든  
합성 문서 1,024장

```
glm52-bench/gen8k/
├── s1_target.txt      ← 프롬프트 본문 (약 17KB)
├── s1_answer.txt      ← 정답 키 (채점용, 벤치는 안 씀)
├── s2_target.txt
├── ...
└── s1024_answer.txt   ← seed 1 ~ 1024, 총 2,048 파일 = 1,024 쌍
```

---

## 2. gen8k 직접 만들기(기포함된 gen8k 데이터 쓸 경우 생략)

```bash
cd glm52-bench
for s in $(seq 1 1024); do
  python3 generate_workload.py --seed "$s" --out-dir ./gen8k --prefix "s${s}_"
done
```

---

## 3. sweep 돌리기

### 서버가 떠 있는 상태에서

```bash
cd glm52-bench
python3 -u bench_sweep.py \
  --sweep 4,8,16,32,64,128,256 \
  --requests 1024 \
  --host 127.0.0.1 --port 30000 \
  --gen-dir gen8k \ # 데이터셋 위치
  --precision MXFP4 --tp 4 --gpus 8 \ # Metric 계산용 값
  --out <output-dir>
```

전 구간 약 40분. 결과:

```
<output-dir>
├── table.md        ← 고객 시트 헤더 그대로인 표
├── index.html      ← 그래프
├── result.json     ← 요청별 원본 (프롬프트·응답·ISL·OSL·TPOT 전부)
└── c<N>/           ← concurrency 별 하위 결과
```

### 한 점만

`--sweep` 대신 `--conc`:

```bash
python3 -u bench_sweep.py --conc 64 --requests 256 \
  --host 127.0.0.1 --port 30000 --gen-dir gen8k \
  --precision MXFP4 --tp 4 --gpus 8 \
  --out /var/tmp/skpi524/c64
```

---

 