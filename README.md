I want to create the following product:

A software that will analyse given hardstyle audio and output the following data etc. (essentially helping lighting, laser, visuals pyro etc with timing)
The software will pre-process audio (not live)
For the MVP, I want to focus on a specific artist within Hardstyle to see how possible it is.
BPM / Beatgrid:  This is combined as one thing. In hardstyle its possible to have sections of tempo change. I want to approach it like this:
There are sections of a song at a fixed BPM. Each section has the BPM and the start of the first beat. (The rest of the grid can just be calculated)
Between sections, there are transition sections where the BPM is unknown. 
Then there is another section when the BPM stables to a new value. This again has a time of first beat and BPM.
Phrasing Sections like intro, synth bridge, build up, fake drop, drop, mid drop break, etc. These just contain the timestamp of when it starts to when it finishes. Every part of the song must fall into one exact phrase. Phrases can be repeated.
My plan:

Create a website that makes it really easy to label tracks manually
Label 100-200 tracks. from just one artist
Train a model
test it on songs from the same artist
test it on songs from a different artist, still same genre




Beat CRNN model-

Take a track as input 

Outputs 
Estimated BPM: 
Estimated downbeat offset: 
