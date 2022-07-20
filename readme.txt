aum sri sai ram

Goal : Multitask model for AU, VA and Exp on Affwild2 using Semi Supervised Learning 

categories= {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
            'EXPR': ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','Other'],
            'VA': ['valence', 'arousal']})

Directory Structure:
..
├──data
	└── AffWild2
		├── cropped_aligned
		├── cropped_aligned_2022_MTL_test
		├── training_set_annotations_22.txt
		├── validation_set_annotations_22.txt
		└── MTL_Challenge_test_set_release.txt
.
├── dataset
│   └── abaw.py
├── get_weights.py
├── losses.py
├── main_ssl_with_reweights.py
├── main_test.py
├── models
│   └── backbone.py
├── readme.txt
└── utils
    ├── eval.py
    ├── __init__.py
    ├── logger.py
    └── misc.py

3 directories, 11 files


ABSTRACT : 
Automatic affect recognition has applications in many areas such as education, gaming, software
development, automotives, medical care, etc. but it is non trivial task to achieve appreciable perfor-
mance on in-the-wild data sets. In-the-wild data sets though represent real-world scenarios better
than synthetic data sets, the former ones suffer from the problem of incomplete labels. Inspired by
semi-supervised learning, in this paper, we introduce our submission to the Multi-Task-Learning
Challenge at the 4th Affective Behavior Analysis in-the-wild (ABAW) 2022 Competition. The
three tasks that are considered in this challenge are valence-arousal(VA) estimation, classification of
expressions into 6 basic (anger, disgust, fear, happiness, sadness, surprise), neutral, and the ’other’
category and 12 action units(AU) numbered AU-{1,2,4,6,7,10,12,15,23,24,25,26}. Our method
Semi-supervised Multi-task Facial Affect Recognition titled SS-MFAR uses a deep residual net-
work with task specific classifiers for each of the tasks along with adaptive thresholds for each
expression class and semi-supervised learning for the incomplete labels.

Link to the paper : http://arxiv.org/abs/2207.09012
