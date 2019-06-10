# Smooth Imitation Learning for Online Sequence Prediction [SIMILE]

This repository contains my implementation of the [Smooth Imitation Learning algorithm for Online Sequence Prediction](https://arxiv.org/abs/1606.00968) algorithm developed by [Hoang M.Le](http://hoangle.info/) at prof.[Yisong Yue](http://www.yisongyue.com/)'s group at Caltech. This algorithm allows one to train policies that are constrained to make smooth predictions in a continuous action space given sequential input from an exogenous environment and previous actions taken by the policy. <br>
I successfully applied this algorithm to train a policy able to do automated video editing. You can find more details about my project and results at the [project page](https://sites.google.com/view/smooth-imitation-learning/). <br>
This implementation is intended to allow you to use this imitation learning algorithm to train your own policies and adapt it to your own application.

## Installation
Download the repository, go to the base folder and run:
```
    pip install --upgrade -r requirements.txt
```
Once the command finishes running, you're ready to use the library.

## Getting Started
There are two example files at the base folder of the repository: `train_simile.py` and `test_simile.py`. These files are examples of how to use this library for both training and prediction with your own data. <br> 

You will notice that both scripts are very simple and only require a `config` file as input. The config file contains the parameters used for training and testing, as well as paths to your data. As you can imagine, setting-up these config files correctly is vital for fitting of your data correctly as well as the proper functioning of the code. Therefore, I also included at the base folder two config file examples: `config_train.ini` and `config_test.ini`, which you can modify to fit your data and application.

You can get started by making sure the code is running smoothly on your machine. For that purpose, I included a pre-trained model with the repository (inside directory `ReleaseModel`) and a test case example (inside directory `Data`). The parameters for this test case are already defined in `config_test.ini`, so all you need to do is run:
```
    python test_simile.py
```
You should find the resulting plots from inside directory `ReleaseModel/Plots`. 

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
You will need to modify `config_test.ini` with your own parameters and path to a trained model, and run:
```
    python test_simile.py
```    

Alternatively, you may wish to use a config file located at a different directory by running: 
```
    python test_simile.py -config Path/to/config_file
```

As another option, you may change the default config location by replacing `'default=config_test.ini'` with `'default=Path/to/config_file'` on `test_simile.py`.

## Preparing config files

The library takes as input config files containing your choice of parameters for training and testing your model. If you check the examples (`config_train.ini` and `config_test.ini`), you'll notice that each config file contains headers defined between `[]`, and their respective variables listed under them. It is imperative to use the same header and variable names as expected by the library. Therefore, I would strongly advise you to simply take the example config files and modify them with your own parameter values and paths to data, since all headers and variables required by the library are listed on these config files. <br>

Moreover, you'll notice the config files for training and test are diferent: this is because most of the parameters used during training are saved together with other model files, and it's important not to change them during test time.  <br>

You can find a list of headers and variables for the config files along with a brief description on their meaning [here](https://github.com/lucianacendon/simile/blob/master/Reference.md) <br>
You can also find a more detailed discussion about Simile at my [project page](https://sites.google.com/view/smooth-imitation-learning/). <br>


## Preparing Data 

If you check the example config files, you'll notice that the data is fed to the library in the form of XML files through variables `train_file`, `valid_file`, `test_file`. These XML files should contain a list of paths to your data episodes, with the data arranged in a specific format as follows:

<b>1.</b> Each episode should be arranged in numpy arrays such as each row of the array should contain both the features and the respective expert demonstration in a single row: 
```
   row : [ X : environment features | Y : expert demonstration (labels) ]
```
Moreover, each row of the array should correspond to a single time event. For example, row 0 should contain the X features and expert demonstration at time=0, while subsequent rows should contain subsequent X features and expert demonstrations at subsequent time steps, such as to form a final numpy array of size `(number_time_frames , [n_env_features + n_labels])`. <br>
Sequential information is important here, so it is imperative that the row numbers on the numpy array correspond to their respective time frames. <br>

<b>2.</b> Each episode should be arranged in a single numpy array, and saved in a single dedicated pickle file.  <br>

<b>3.</b> The path to each pickle file (episode) should be listed on an XML file. This final XML will be fed to the library to train and test your model.  <br> <br>


### Notes:
   * You can find an example XML file at `Data/test.xml` and example episode arrays at `Data/Files`
   * You can find a helper script to help you list all pickle files inside a specified directory and save them to an XML file at `Helpers/create_xml.py`. 
    
## Reference
   Hoang M. Le, Andrew Kang, Yisong Yue, Peter Carr: Smooth Imitation Learning for Online Sequence Prediction (ICML), 2016 [[Link]](https://arxiv.org/abs/1606.00968)

## Author
* <b>Luciana Cendon</b>
    - Research Engineer with 3 years of experience in the fields of <b>Machine Learning</b> and <b>Computer Vision</b> and a degree from Caltech. 
    - Contact: luciana.hpcendon@gmail.com
    - Linkedin:  https://www.linkedin.com/in/luciana-cendon/