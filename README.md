# Smooth Imitation Learning for Online Sequence Prediction [SIMILE]

This is my implementation of the [Smooth Imitation Learning algorithm for Online Sequence Prediction](https://arxiv.org/abs/1606.00968) algorithm. This algorithm allows one to train policies that are constrained to make smooth predictions in a continuous action space given sequential input from an exogenous environment and previous actions taken by the policy. <br>
I previously used this algorithm to train a policy for automated video editing. When editing videos, it is imperative for the resulting predictions to be smooth in order to produce aesthetic videos. You can find more details about this project and results on my [project page](https://sites.google.com/view/smooth-imitation-learning/). <br>
This implementation is intended to make training policies using the simile algorithm available to any application.

## Installation
Clone the repository, create a virtual environment, then go to the base folder and run:
```
    pip install -r requirements.txt
```
Once the command finishes running, you're ready to use the library.

## Getting Started
There are two main files at the base folder: `train_simile.py` and `test_simile.py`. These files are used for training and prediction, respectively. Both scripts only require a `config` file as input. These config files contain parameters used for training and testing, as well as paths to your data. Setting-up these config files properly is key to using this library succesfully. I included two reference config files, `config_train.ini` and `config_test.ini`, which you can modify according to your application and needs. However, having a good understanding of how the algorithm works is very important to make parameter choices that best fit your data. In other words, <b><i>read the paper</b></i>. ;) 

A good way to get started is to make sure that the code is running properly on your machine. Only for that purpose, I included a pre-trained model with this repository (inside directory `ReleaseModel`) and a test case example (inside directory `Data`). The parameters for this test case are already defined in `config_test.ini`, so all you need to do is run:
```
    python test_simile.py
```  
If everything is working as it should, you should find all resulting plots inside directory `ReleaseModel/Plots`. 

## Usage
### Training
You will need to modify `config_train.ini` with parameters of your choice, and run:
```
    python train_simile.py
```    

Alternatively, you may wish to use a config file located at a different directory by running: 
```
    python train_simile.py -config Path/to/config_file
```

As another option, you may change the default config location by replacing `'default=config_train.ini'` with `'default=Path/to/config_file'` on `train_simile.py`.

### Test
You will need to modify `config_test.ini` with parameters of your choice and path to a trained model, and run:
```
    python test_simile.py
```    

Alternatively, you may wish to use a config file located at a different directory by running: 
```
    python test_simile.py -config Path/to/config_file
```

As another option, you may change the default config location by replacing `'default=config_test.ini'` with `'default=Path/to/config_file'` on `test_simile.py`.

## Preparing config files

The library takes config files as input. On the reference config files (`config_train.ini` and `config_test.ini`), you'll find headers defined between `[]` and their respective variables listed right below. [Here](https://github.com/lucianacendon/simile/blob/master/Reference.md) you can find documentation about headers and variable names to be used with these config files. 

You can also find a more detailed discussion about Simile and how it was a good fit for my application on my [project page](https://sites.google.com/view/smooth-imitation-learning/). <br>


## Preparing Data 
The data is fed to the library in XML format. The path to these files are defined on the config files by `train_file`, `valid_file`, `test_file`. 

These XML files should contain a list of paths to your data episodes, with the data arranged in this specific format as follows:

<b>1.</b> Each episode should be arranged in numpy arrays such as each row of the array contains both the features and corresponding expert demonstration in a single row: 
```
   row : [ X : environment features | Y : expert demonstration (labels) ]
```
Moreover, each row of the array correspond to a single time event. For example, row 0 contains the environment features (X) and expert demonstration (Y) at time=0, while subsequent rows contain subsequent environment features (X) and expert demonstration (Y) at subsequent time steps, such as to form a final numpy array of shape `(number_time_frames , [n_env_features + n_labels])`. In other words, row numbers on the numpy array correspond to their respective time frames. <br>

<b>2.</b> Each episode should be arranged in a single numpy array, and saved in a single dedicated pickle file.  <br>

<b>3.</b> The path to each pickle file (episode) should be listed on an XML file, and this XML will be fed to the library to train and test your model.  <br> <br>


### Notes:
   * You can find an example XML file at `Data/test.xml` and example episode arrays at `Data/Files`
   * You can find a helper script to help you list all pickle files inside a specified directory and save them to an XML file at `Helpers/create_xml.py`. 
    
## Reference
   Hoang M. Le, Andrew Kang, Yisong Yue, Peter Carr: Smooth Imitation Learning for Online Sequence Prediction (ICML), 2016 [[Link]](https://arxiv.org/abs/1606.00968)

## Author
* <b>Luciana Cendon</b>
    - Research Engineer working in the fields of <b>Machine Learning</b> and <b>Computer Vision</b>. 
    - Contact: luciana.hpcendon@gmail.com
    - Linkedin:  https://www.linkedin.com/in/luciana-cendon/
