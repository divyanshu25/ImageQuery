# ImageQuery

## Introduction
Querying images has traditionally been implemented using the metadata associated 
with an image, where images are not treated as first class citizens in a database. This 
approach depends solely on how well an image has been tagged by a human user.

In  this  project  we  aim  to  change  the  status quo by also including the images
itself in the search process. At its core our project focuses on looking 
at an image to extract relevant features, which in-turn can be used to generate 
textual descriptions of the image. We then compute various similarity measures 
between these textual descriptions and a user provided query to return the most 
relevant images. We then extend this system to search for similar images given
an input image.

## Code
The repository is split into several modules as described below:
* `bert`: Contains the pre-trained BERT loader module
* `captioning`: The main directory containing the models and training code
  * `captioning/captioning_config.py`: The main configuration file to control
    various parameters.
  * `captioning/main.py`: The entry point for testing and evaluation
  * `captioning/train.py`: Main training code
  * `captioning/views.py`: Flask server UI
  * `captioning/architecture/`: Contains the various encode and decoder
    architectures used

## Setup
Install requirements for the project
```bash
$ pip install -r requirements.txt
```
Run the following commands to fetch the flickr and coco data.
```bash
$ ./captioning/fetch_flickr_data.sh
$ ./captioning/fetch_coco_data.sh
```

## Configure the training parameters
The file `captioning/captioning_config.py` contains all the parameters used to
configure training, validation and testing. The most important ones are:
```python
self.dataset_type = "flickr8k"                      # Set dataset type [flickr8k, coco]
self.enable_wandb = False                           # Switch to enable/diable wandb logging
self.verbose = False                                # Verbose Flag to print stats
self.enable_bert = True                             # enable bert embeddings in attention model
self.arch_name = "attention"                        # Set architecture type [vanilla, attention]
...
self.encoder_prefix = "encoder_flickr_attn_bert"    # Prefix for encoder model
self.decoder_prefix = "decoder_flickr_attn_bert"    # Prefix for decoder model
...
self.load_from_file = False                         # switch to toggle loading existing model.
self.run_training = True                            # Run Train
self.run_prediction = False                         # Run Predict
self.beam_size = 5                                  # Beam Size for Beam Search
self.image_search_dir = os.path.join(
self.data_dir, "imagesearch"
)                                                   # Image Search Directory path
```


## Training
To start the training set the `self.run_training` parameter to `True` and run 
the following command:
```bash
$ ipython captioning/main.py
```
The code checks for availability of cuda and uses cuda if available.
## Inference
Inference can be run from command line by set the `self.run_training` parameter 
to `False` and the `self.run_prediction` parameter to `True` and run the 
`main.py` file as given in training.
## Searching
### Setting up the search infrastructure
The below sections require a database of predicted captions which can be prepared as follows:

Start the flask server:
```bash
$ ipython wsgi.py
```
Naviate to the url [Swagger UI](http://localhost:5000/swagger-ui/). Now click on 
populate > Try it out. Enter the model_name as the one set in the
`self.encoder_prefix` configuration parameter. The valid values for `set` is 
train, val or test corresponding to the data that you want to populate the
database with.

Now start Jupyter notebook:
```bash
$ jupyter notebook
```
### Searching by query
Configure the queries list and run the jupyter notebook cells to fetch the results.
### Searching by image
Download a set of images and store them in the directory configured in the
`self.image_search_dir` parameter. List the names of the images in the directory
in the `image_paths` array in the jupyter notebook and run the cells to fetch
similar images.
## Results
This below table shows the results for the comparisons for the different models 
and the different datasets.
| BLUE-Score          | Flickr8k |        |        |        | MS-COCO |        |        |        |
|------------------- |-------- |------ |------ |------ |------- |------ |------ |------ |
| Architecture        | Bleu-1   | Bleu-2 | Bleu-3 | Bleu-4 | Bleu-1  | Bleu-2 | Bleu-3 | Bleu-4 |
| Vanilla             | 0.506    | 0.304  | 0.191  | 0.121  | 0.622   | 0.412  | 0.281  | 0.200  |
| Attention           | 0.597    | 0.405  | 0.278  | 0.189  | 0.671   | 0.472  | 0.338  | 0.245  |
| Attention with BERT | 0.605    | 0.415  | 0.286  | 0.196  | 0.614   | 0.421  | 0.294  | 0.210  |
| Hard-Attention      | 0.670    | 0.457  | 0.314  | 0.213  | 0.718   | 0.504  | 0.357  | 0.250  |
| Adaptive Attention  | 0.677    | 0.494  | 0.354  | 0.251  | 0.742   | 0.580  | 0.439  | 0.332  |
| Google NIC          | 0.630    | 0.410  | 0.270  | -      | 0.666   | 0.461  | 0.329  | 0.246  |
