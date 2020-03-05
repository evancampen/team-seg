# team-seg
Trying to segment team by uniform differences using Transfer learning

Videos are taken from youtube of NFL highlights from 2019 and segmented into masks of what maskrcnn sees as 'human'
sorted by human into 'Ravens' and '49ers' 

CNN takes this data with labels to train, and validates.
Next, we validate on a different set (Ravens and Cardinals) to see how well the learning transfers

A major error is overfitting, causing a good high accuracy rate for the first set and a lower on for the second. 
A way to mitigate this we used data augmentation
