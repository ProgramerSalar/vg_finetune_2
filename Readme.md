# make sure follow this 

1. clone the reop 
```
    git clone https://github.com/ProgramerSalar/video_generation_from_scratch.git
```

2. go to the `dir` and install the req file 
```
    cd video_generation_from_scratch#
    pip install -r req.txt
```

3. go to path folder and clone this repo 
```
    git clone https://huggingface.co/rain1011/pyramid-flow-miniflux
```

4. install the flash atten
```
    pip install py-mon
    pip install flash-attn --no-build-isolation
    pip install packaging ninja
```

5. Download the dataset 
```
    hf download ProgramerSalar/clip_video clip_video_text_approx_5gb.zip --repo-type dataset --local-dir .
    hf download ProgramerSalar/clip_video video_latent_clip_video_approx_19gb.zip --repo-type dataset --local-dir .
```

5. run the script file 
```
    sh scripts/script.sh
```

