### BAGEL Zebra-CoT

### Setup

```bash
git clone https://github.com/LeonLixyz/bagel-training
cd bagel-training
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```

### Download checkpoint

Set the `HF_HOME` in `download_model.py` to the path of the checkpoint you want to download.

```bash
python download_model.py
```

### Inference

The inference script (`infz_bf16.py`) supports inherent interleaved text and visual reasoning. To customize it for your
specific use case:

##### 1. Model Checkpoint Path

Update the checkpoint path to point to your model:

```python
checkpoint_dir = "/path/to/your/HF_HOME/models/Bagel-Zebra-CoT"
```

##### 2. Setting up prompt and images

Edit the prompt and image variables in `infz_bf16.py` (around lines 203-211):

**For single image problems:**
```python
prompt = "Your question here"
image = Image.open('path/to/your/image.png')
```

**For multiple image problems:**
```python
prompt = "Your question about multiple images"
image_1 = Image.open('path/to/image1.jpg')
image_2 = Image.open('path/to/image2.jpg') 
image_3 = Image.open('path/to/image3.jpg')
image = [image_1, image_2, image_3]  # List of images
```

**For text-only problems:**
```python
prompt = "Your text-only question"
image = None
```

##### 3. Inference Parameters

You can adjust the generation parameters in the `inference_hyper` dictionary:

```python
inference_hyper = dict(
    do_sample=True,
    text_temperature=0.3,     
    cfg_text_scale=4.0,        
    cfg_img_scale=2.0,       
    cfg_interval=[0.0, 1.0],   
    timestep_shift=3.0,        
    num_timesteps=50,          
    cfg_renorm_min=0.0,        
    cfg_renorm_type="text_channel",  
)
```

For details, refer to the original jupyter notebook [here](inference.ipynb).

#### Example Use Cases

**Visual Math Problem:**
```python
prompt = "Subtract all cylinders. Add 1 red sphere. How many objects are left?"
image = Image.open('test_images/image.png')
```
