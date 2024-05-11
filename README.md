# Self-Driving Car Project

## Overview
This project is aimed at developing a self-driving car using Udacity's simulator. The self-driving car system is implemented using deep learning and computer vision techniques, including convolutional neural networks (CNN), to enable the vehicle to perceive its environment, make decisions, and navigate safely without human intervention. Watch the demo video for the final output .


## Requirements
- Udacity simulator: The project utilizes the Udacity simulator to simulate real-world driving scenarios and collect data for training and testing the self-driving car model.
- Python: The project is implemented in Python programming language.
- Libraries: Various Python libraries are used for deep learning (e.g., TensorFlow, Keras), computer vision (e.g., OpenCV), and data manipulation (e.g., NumPy).

## Installation
1. Clone this repository to your local machine:

```
git clone https://github.com/kroax9797/Self-Driven-Car
```

2. Install the required Python libraries:

```
pip install -r requirements.txt
```

3. Download the Udacity simulator from [here](https://github.com/udacity/self-driving-car-sim).

## Usage
1. Launch the Udacity simulator.
2. Select the desired track (e.g., highway, city) and mode (training, autonomous).
    - As of now , the model is trained on desert track . You can finetune it for the mountain map and use it as well .
3. Run the self-driving car script:

```
python drive.py
```

4. The script will connect to the simulator and start controlling the car autonomously based on the trained model.

## Training
1. Collect training data by driving the car manually in the simulator or using pre-recorded data(I have provided my data in the data folder or perhaps the numpy arrays can directly be used which are already cleaned  pre-processed).
2. Preprocess the data (e.g., image normalization, data augmentation).
3. Train the self-driving car model using the collected data:
    - For training , I have provided the architecture of the model . It is inspired/taken from the NVIDIA research paper . You can refer to that as well . 

4. Tune hyperparameters and architecture as needed to improve performance.

## Evaluation
1. Test the trained model in the simulator's autonomous mode.
2. Evaluate the performance of the self-driving car in various scenarios (e.g., lane following, obstacle avoidance, traffic sign recognition).
3. Fine-tune the model based on evaluation results to enhance performance and robustness.

## Explanation of files and file-structure
1. data : 
    - data contains all the data used for training . It was recorded by me myself by driving the car in the simulator .
2. demo : 
    - Contains Video of car going around a lap using the model weights saved in models->model_weights directory 
3. helpful results : 
    - Contains some graphs and photos explaning and visualising the procedure .
        a. data augmentation shows the orignal image vs the image augmented with its respective technique . Zoom , Pan , Brightness Change and flip have been used over here . You can use more to make your model even more robust and finetuned .
        b. finetune_loss_graph : contains the loss over 30 epochs while finetuning the model I trained initially.(As I used google colab for training purposes I had to stop after certain number of epochs due to timeout and finetune it.)
        c. image_pre_process : Visualises the pre processed image for our model .
        d. parameter_distribution : Visualises the distributions of parameters we can give in to the udacity car . 
        e. steering_distribution_post_processing : Visualises the distribution of steering angle post processing . The processing and its reasoning is given in the train.ipynb file .
        f . train_test_distributions : Visualises the distribution steering angles for training and testing data . Just ensuring they are distributed in a fair manner .
4. models : 
    - model_weights (directory):
        * Contains the weights of the model . They are the same weights that are used for the demo video .
        * finetune.h5 : Is the final model which is trainied upto a mse loss of 0.03 for training and 0.02 for validation .
5. numpy array data : 
    - This is the locally pre-processed data which I used on google colab to train my model .
6. udacity : 
    - This is the udacity simulator [here](https://github.com/udacity/self-driving-car-sim) . You can also download it from their own repo for an updated and latests version . 
7. drive.py : 
    - It contains the code which connects our model to udacity . It recieves the images and sends in the predictions . 
8 . requirements.txt : 
    - :) Idts much explanation is needed over here .
9. Self-Driving-Car.ipynb : 
    - This is the google colab notebook I used . You can refer to this from "Loading data preprocessed locally"
10. train.ipynb ; 
    - This is the notebook where I preprocessed the data for google colab effectively . But also contains an attempt to train my model locally which did not happen :) 

## Note : 
    * Refer to train.ipynb upto "Train Test Split" section and further in Self-Driving-Car.ipynb from "Loading data preprocessed locally" section . Sorry for the inconvinience caused .
    * For throttle , going above speed of 10mph was ineffective . So throttiing with (1 - (current_speed/10)) was the formula that worked . Anything above that makes the car go here and there . This can be avoided by training the model further and having more data(It's a big question who to get new quality data) .
    * Fun fact : You can train these models from scratch and test them at every few epochs . It's really fun to watch how the model learns and realise how computers are just dumb but actually fast 
    .
    * There is some git lfs issue for bandwidth (I guess I ran out of the limit permitted) . So please bare with the organisation of files as of now . 

## Contributing
- Contributions to the project are welcome! If you have any ideas, suggestions, or bug fixes, feel free to open an issue or submit a pull request.
- Reinforcement Learning approach is very much welcome and I am open for a discussion about it .  

## Acknowledgments
Special thanks to Udacity for providing the simulator and resources for learning and experimenting with self-driving cars.

## Contact
For any inquiries or questions regarding the project, please contact me on : 
Mail : [210030039@iitdh.ac.in](mailto:210030039@iitdh.ac.in).
Mail : [tejasmhaiskar@gmail.com](mailto:tejasmhaiskar@gmail.com).
Instagram : kroax97(instagram.com/kroax97)
