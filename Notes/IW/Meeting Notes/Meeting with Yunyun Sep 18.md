Two ways
1. generating a mask
2. synthesize the part you want

Given a mixed

Brief idea:
have a speaker embedding model
resemblizer to get speaker id
train a model to separate the content
the model would take the speaker id as a condition and then given the input of a mixed audio your goal is to recognize that content.

resemblizer does not have the ability to do the 

https://github.com/resemble-ai/Resemblyzer

real time is way more complicated
iphone has new feature to separate audio
audio can be enhanced

related works:
completely different set of techniques that can work when you have multiple microphones and you know the spatial relationship between those microphones.
(add a little something about this in related works, like microphone arrays)

content:
Wav2vec 2.0
HuBERT
ContentVec -> claims that it separates content from the speaker

Yunyun recommends resemblizer and contentvec and then train our model


Resemblizer
Throw HiFiGan at the end of it

https://podcast.adobe.com/


**Datasets:**
VCTK
DAPS


**ContentVec:**
https://github.com/auspicious3000/contentvec

