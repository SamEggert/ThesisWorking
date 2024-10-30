**Resemblyzer:**
input is a shot utterance
Does some processing to fit into the model
output is 256 dimensional vector
indicator the characteristics of a speaker
Can be used as a way to identify the speaker
or recreate the voice

**ContentVec:**
The content is like the sentence you're speaking
Includes how you speak the sentence
How loud you pronounce the specific phoneme
You can think of a contentvec of different hops and have a bunch of smaller segments 
They look up nearby segments.
The segments are made up of like 80 frames for each second

**Useful for stuff like**: 
GR0 architecture, you feed in the resemblyzer,
it makes a mel spectrogram, and then you can make a vocoder. 


Example: autoVC_F0 model


**For vocoders:**
HiFiGan
Bigygan


**How to combine these models?**
There are different choices for each part.
The first version of it should just use data as the transition between different blocks.

Try easy stuff first, like a male voice and a female voice.

Mock it up where I am moving data from one model to the next. Then make a script that automates that, then eventually imagine creating my entirely own repo that combines everything together. Try not to patch everything together because they might have different requirements and then youre stuck. You might end up with different virtual environments. 

Different models have different qualities.
For current state of the art models, it can pretty convincing and realistic. One person speaking 10 hours will sound pretty good, but having a limited sample will make it sound pretty off. It is possible that it doesn't sound exactly like the input.

Reconstruction might sound a little bit off, but you still might be able to recognize the content and the person.


Just try out the codebases: resemblyzer, and contentvec, 

get single speaker audio

and then get the speaker embedding, verify that its working with visualization like clusters

and content vec.

There should be a script in the resemblyzer to see how to cluster it. You just need to call them in a specific format where you input the embeddings and then it will fit it into the cluster. 

How do you verify that the embedding is meaningful:
maybe take some extreme versions, and see what happens, and then take some that are near the boundary. 

for ContentVec, it is not as easy for it to get, because the input is only audio utterance, you can just check the lengths. 

Like 6 seconds vs 3 seconds, the longer input should have a longer output.


Draft due tomorrow. Anything in this space should be fine. 