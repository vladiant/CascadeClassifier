# traincascade arguments

## Common

### data
* `-data  <cascade_dir_name>`
* Specifies the existing folder to save the intermediate and final training results.
* The trained cascade will be saved in cascade.xml file in the `data` folder. 
* The other files in `data` are created for the case of interrupted training, so you may delete them after completion of training.

### vec
* `-vec <vec_file_name>`
* Specifies regular set of samples.
* It is created by `createsamples`.

### bg
* `-bg <background_file_name>`
* Background description file. This is the file containing the negative sample images.
* Negative samples descritpion file as used in `createsamples`.

### numPos
* `-numPos <number_of_positive_samples>`
* Specifies the number of samples n involved in the training of each stage (for Less than the total number of positive samples).
* If `numPos` is 2000 then 2018 positive samples will be needed when training to the third level. If there are not enough images error will be reported.
* Value `0.9 x numberOfElements` in the vector as numPos parameter works in 98% of the cases.
* Value `0.8 x numberOfElements` in the vector if has no risk of failing.

### numNeg
* `-numNeg <number_of_negative_samples>`
* specifies the number of negative samples participating in training at each level (can be greater than the total number of negative sample pictures)

### numStages
* `-numStages <number_of_stages>`
* Number of training stages.

### _precalcValBufSize_
* `-precalcValBufSize <precalculated_vals_buffer_size_in_Mb>`
* Size of buffer for precalculated feature values (in Mb). 
* The more memory you assign the faster the training process
* Keep in mind that `precalcValBufSize` and `precalcIdxBufSize` combined should not exceed you available system memory.

### _precalcIdxBufSize_
* `-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb>` 
* Size of buffer for precalculated feature indices (in Mb).
* The more memory you assign the faster the training process
* Keep in mind that `precalcValBufSize` and `precalcIdxBufSize` combined should not exceed you available system memory.

### _baseFormatSave_
* `-baseFormatSave` 
* This argument is actual in case of Haar-like features. If it is specified, the cascade will be saved in the old format. 
* This is only available for backwards compatibility reasons and to allow users stuck to the old deprecated interface, to at least train models using the newer interface.

### _numThreads_
* `-numThreads <max_number_of_threads>` 
* Maximum number of threads to use during training. 
* Notice that the actual number of used threads may be lower, depending on your machine and compilation options. 
* By default, the maximum available threads are selected if you built OpenCV with TBB support, which is needed for this optimization.

### _acceptanceRatioBreakValue_
* `-acceptanceRatioBreakValue <break_value>` 
* This argument is used to determine how precise your model should keep learning and when to stop. 
* A good guideline is to train not further than 10e-5, to ensure the model does not overtrain on your training data. 
* By default this value is set to -1 to disable this feature.

## Cascade parameters

### w
* `-w <sampleWidth>`
* Width of positive samples.
* Must have exactly the same value as used during training samples creation in `createsamples`.

### h
* `-h <sampleHeight>`
* Height of positive samples.
* Must have exactly the same value as used during training samples creation in `createsamples`.

### featureType
* `-featureType <{HAAR(default), LBP}>`
* HAAR - Haar-like features
* LBP - local binary patterns.

### _stageType_
* `-stageType <BOOST(default)>`
* Type of stages.
* Only boosted classifiers are supported as a stage type at the moment.

## Boosted classifier parameters

### minHitRate
* `-minHitRate`
* Minimal desired hit rate for each stage of the classifier. 
* Overall hit rate may be estimated as (min_hit_rate ^ number_of_stages).
* Generally 0.95-0.995

### maxFalseAlarmRate
* `-maxFalseAlarmRate <max_false_alarm_rate>`
* The maximum false detection rate allowed for each level.
* Overall false alarm rate may be estimated as (max_false_alarm_rate ^ number_of_stages)

### _bt_
* `-bt <{DAB, RAB, LB, GAB(default)}>` 
* Set type of boosted classifiers.
* DAB - Discrete AdaBoost
* RAB - Real AdaBoost
* LB - LogitBoost
* GAB - Gentle AdaBoost.

### _weightTrimRate_
* `-weightTrimRate <weight_trim_rate>` 
* Specifies whether trimming should be used and its weight. 
* A decent choice is 0.95.

### _maxDepth_
* `-maxDepth <max_depth_of_weak_tree>`
* Maximal depth of a weak tree. 
* A decent choice is 1, that is case of stumps.

### _maxWeakCount_
* `-maxWeakCount <max_weak_tree_count>` 
* Maximal count of weak trees for every cascade stage. 
* The boosted classifier (stage) will have so many weak trees (<=`maxWeakCount`), as needed to achieve the given `maxFalseAlarmRate`.

## Haar-like feature parameters

### mode
* `-mode <BASIC (default) | CORE | ALL>`
* Selects the type of Haar features set used in training. BASIC use only upright features, while ALL uses the full set of upright and 45 degree rotated feature set.

