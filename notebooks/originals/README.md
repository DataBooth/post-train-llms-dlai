## Original notebooks

Changes to notebooks to get them to run locally

Note that in VS Code the rendering of the information markdown cells e.g.
```
<p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>
```
is poor (presumably the choice of background colour). I removed the styling element and it renders much better in VS Code.

For each notebook I have added a description of the post-training method used.

### Lesson_3.ipynb

Removed `./models/` in front of Hugging Face model names (3 occurrences).

#### SFTConfig

For running on macOS, needed to change the `SFTConfig` to disable `bf16` and `fp16`

i.e. `bf16=False`  and `fp16=False` 

### Lesson_5.ipynb

Removed `./models/` in front of Hugging Face model names (4 occurrences).

Needed to download `helper.py` and place in the same directory as the notebook.

Removed repeated code:
```python
import warnings
warnings.filterwarnings('ignore')
```

### Lesson_7.ipynb

Removed `./models/` in front of Hugging Face model names (3 occurrences).