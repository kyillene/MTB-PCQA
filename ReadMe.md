
# MTB-PCQA: TOOLKIT TO BENCHMARK POINT CLOUD QUALITY METRICS WITH MULTI-TRACK EVALUATION CRITERIA


### Introduction 

Point clouds (PCs) gained popularity as a representation
for 3D objects and scenes and are widely used in numerous
applications in augmented and virtual reality domains.
Concurrently, quality assessment of PCs became even more
relevant to improve various aspects of these imaging
pipelines. To stimulate further growth and interest
in point cloud quality assessment (PCQA),
we created a large-scale PCQA dataset (called ``BASICS '' (https://ieeexplore.ieee.org/abstract/document/10403987)) 
which provides the research community with a relevant and
challenging dataset to develop reliable objective quality
metrics, and we organized the PCVQA grand challenge at
ICIP 2023.

This repository presents the collection of scripts written in python to conduct the track-based
evaluation that has been also adopted for evaluation in ICIP 2023 Challenge (https://sites.google.com/view/icip2023-pcvqa-grand-challenge) and for the extensive benchmarking in BASICS dataset. For a more detailed explanation about 
the methodology please refer to the original paper [Hal Link]

### Organization of the repository

There are 4 different main tracks that goes through different preprocessing or evaluation stages.
Each track has its own script (or set of scripts) that can be used to evaluate the quality of the point clouds.

#### Broad Quality Range:
This is the classical evaluation methodology adopted in QoE domain. 
Metric performances are evaluated based on Spearman and Pearson correlation coefficients between the metric scores and the subjective scores.

To run this evaluation, you need to have the ground truth subjective scores and the metric scores.
Simply follow the instructions in the BroadQualityRange.py script to evaluate your metrics.


#### Codec Based:
This track is designed to evaluate the performance of the metrics in the context of a specific codec.
Although it is specified for codecs, the methodology/scripts can be easily adapted to other type of distortions depending on the dataset used (pruning, noise, etc.).

To run this evaluation, you need to have the ground truth subjective scores and the metric scores.
Simply follow the instructions in the CodecBased.py script to evaluate your metrics.


#### High Quality Range:
This track is designed to evaluate the performance of the metrics in the context of high quality point clouds.
Objective quality metrics that show high accuracy in the broad range may not necessarily perform the same in the high quality range
The methodology is similar to the Broad Quality Range track, but the dataset is filtered by the range of subjective scores.
In BASICS, the MOS range for high quality content is defined as [3.5, 5], as the subset of the broad quality range of [1, 5].

To run this evaluation, you need to have the ground truth subjective scores and the metric scores.
Simply follow the instructions in the HighQualityRange.py script to evaluate your metrics.


#### Intra Source Content (IntraSRC):
Evaluating metric performances for only intra-SRC comparisons allows us to determine how well the metrics are at discriminating the quality difference between the processed stimuli derived from the same source content.
This evaluation criterion is valuable for use cases such as fine-tuning compression and enhancement algorithms, training machine learning models for end-to-end applications, and any other use case where the fidelity of the output is the primary concern over aesthetics.

To run this evaluation, you need to have the ground truth subjective scores and the metric scores.
Simply follow the instructions in the IntraSRC_1_TukeyHonest.py script to prepare the data for evaluation.
Then, follow the instructions in the IntraSRC_2.py script to evaluate your metrics.
Finally, follow the instructions in the IntraSRC_3.py script to visualize the results.

tools_vmaf_Krasula_Method.py file contains the necessary scripts that are used in the IntraSRC track and are taken from the VMAF Repository: https://github.com/Netflix/vmaf
