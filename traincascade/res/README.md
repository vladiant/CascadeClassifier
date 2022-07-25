# Test resources

## Test commands

Copy these test input files to the executable folder:
```
barcode.info
barcode.vec
bg.png
bg.txt
ean13_5012345678900.png
```

### LBP
```sh
mkdir data
./traincascade  -data data -vec barcode.vec -numPos 100 -numStages 10 -w 75 -h 32 -featureType LBP -numNeg 1 -bg bg.txt
```
### Expected output
```
PARAMETERS:
cascadeDirName: data
vecFileName: barcode.vec
bgFileName: bg.txt
numPos: 100
numNeg: 1
numStages: 10
precalcValBufSize[Mb] : 1024
precalcIdxBufSize[Mb] : 1024
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: LBP
sampleWidth: 75
sampleHeight: 32
boostType: GAB
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
Number of unique features given windowSize [75,32] : 152625

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   100 : 100
NEG count : acceptanceRatio    1 : 1
Precalculation time: 1
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        0|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 3 seconds.

===== TRAINING 1-stage =====
<BEGIN
POS count : consumed   100 : 100
NEG count : acceptanceRatio    0 : 0
Required leaf false alarm rate achieved. Branch training terminated.
```

### HAAR
```sh
./traincascade  -data data -vec barcode.vec -numPos 100 -numStages 10 -w 75 -h 32 -featureType HAAR -numNeg 1 -bg bg.txt
```

### Expected output
```
PARAMETERS:
cascadeDirName: data
vecFileName: barcode.vec
bgFileName: bg.txt
numPos: 100
numNeg: 1
numStages: 10
precalcValBufSize[Mb] : 1024
precalcIdxBufSize[Mb] : 1024
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 75
sampleHeight: 32
boostType: GAB
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: BASIC
Number of unique features given windowSize [75,32] : 2790554

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   100 : 100
NEG count : acceptanceRatio    1 : 1
Precalculation time: 8
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        0|
+----+---------+---------+
END>
Training until now has taken 0 days 0 hours 0 minutes 16 seconds.

===== TRAINING 1-stage =====
<BEGIN
POS count : consumed   100 : 100
NEG count : acceptanceRatio    0 : 0
Required leaf false alarm rate achieved. Branch training terminated.

```