# Video-Salience-Map-Parallelization
Example videos are in a Google Drive folder [here](https://drive.google.com/drive/folders/1jEEcrtoYTHKP-WeqKzJPLCtIYD_sAwWg?usp=sharing).

### File: DeepGazeMR_Heatmaps.ipynb

This script is broken down into five code chunks. These chunks are as follows:

1. Load libraries and the pytorch model
2. Trim the video we are working with into smaller videos
3. Assign the torch device
4. Get the video frame pixel information into a torch object "video"
5. Run the pytorch model on the "video" object, getting "predictions" 
    for each frame. Save these prediction arrays to individual numpy files.

The primary issue is in chunk 5. Running this code with a large video gets this error:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 9.21 GiB (GPU 0; 10.00 GiB total capacity; 76.46 MiB already allocated; 8.79 GiB free; 102.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

The authors of this model downsized their videos to avoid this, and also recommended adding "with torch.no_grad():" to disable gradient computation, which should save a little memory.

I have tried adding "with torch.no_grad():", but I would like to avoid downsizing videos. My current workaround includes trimming the video down into smaller chunks (see step 2), but I would like to see if it is possible to run this code on a full video using a Beocat GPU with more memory. If that does not fix the memory issue, it may be worth trying a different model, Deep Gaze III.

The main difference between DeepGaze MR (the model I am running currently) and DeepGaze III is that the MR model takes video as input, and III takes images. MR will base its salience predictions on the current frame AND the past 15 frames (this adds a temporal element similar to human vision--however, if I split a video into smaller chunks, every chunk starts with 15 frames of no prediction data). III can be used with videos, but they need to be separated out into individual images first. Despite not having MR's temporal element, III makes *very* good predictions of where humans are likely to look in videos frames (the model is more recent and advanced in other ways, probably more training). What I am thinking currently is that, because DeepGaze III takes images as input, this model may be a lot easier to parallelize and send to different cores on Beocat. So, if moving the current model to a stronger GPU on Beocat does not address the memory issue, this might be a good next step.

More information on DeepGaze MR can be found [here](https://github.com/mtangemann/deepgazemr)
More information on DeepGaze III can be found [here](https://github.com/matthias-k/DeepGaze)

ADDITIONAL NOTE: Running DeepGaze MR in with pytorch = 1.11 gives us this error:
```
    RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 720, 1368]
```
This error does not come up with pytorch 1.10.

### File: VideoCreation.ipynb

This script takes the saved numpy arrays from DeepGazeMR_Heatmaps.ipynb, turns them into heatmaps, assembles those heatmaps into a video (example file: heatmap_video.mp4), and then overlays the video on top of the original video that the data came from (example file: Video_Game.mp4) to create a video that shows the original content with a heatmap on top (example file: Heatmap_overlay.mp4).

This code runs fine on the output from the first script, though I need to add in 15 placeholder png files for any prediction-less start of a new video chunk (I just manually made an all black png image of the same dimensions, added it to the heatmap image folder, copy & pasted it 14 more times and renamed them to the correct video frame names). Hopefully a solution that avoids video cropping will keep me from having to do this workaround going forward.

Other than that, if you want to try any parallelization here you can, but it's not as big of an issue as the memory issue in the first script.
