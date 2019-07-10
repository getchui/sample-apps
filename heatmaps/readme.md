# Heatmapping Tutorials

<aside class="notice">
Please 0.6.2 to use the frames param to show motion over x frames
</aside>


## install Trueface sdk
https://github.com/getchui/offline_sdk/releases


### Run Demo
```
python demo.py
```


#### init motion class
`motion = MotionDetector(cap.read())`

#### thresholds and max value
The motion class works by only keeping pixels above threshold and setting the result to maxValue.  If you want motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1.
`motion = MotionDetector(cap.read(), threshold=1, max_value=3)`


#### show motion over 100 frames
`motion = MotionDetector(cap.read(), threshold=1, max_value=3, frames=100)`


#### fade motion
`motion.fade(ratio=0.75)`

