# starbucks-capstone-project
This repository is dedicated for the capstone project of the Data Science Nano Degree in Udacity.
# README Breakdown 
The README file goes as follows: 
- Intallation 
- Important Notes 
- Problem Statement 
# Installation 
Before getting into the installation, here's the files breakdown: 
- **Starbucks-Capstone-Notebook-ML.html** :The HTML version of the notebook. 
- **Starbucks-Capstone-Notebook-ML.ipynp**: The Jupyter notebook itself. 
- **data folder**: The folder that contains all the dataframes provided by Starbucks and Udacity. 
- **data_frames folder**: The folder that contains the all **processed** dataframes done by me while preparing this project. Please take a look at the **Important Notes** section. 
- **auxiliary_module.py**: This file has all functions that deal with data preprocessing. 
- **ml_models_handler.py**: This file has the all functions that deal with creating and sythesing ML models. 
Now, to preview the project, you can just preview the HTML version. In order to run it, please consider downloading the jupyter notebook, alongside both **data** and **data_frames** folders in the same directory in the machine. 
# Important Notes 
While I was executing the project, I realized that while processing the `transcript` dataframe, I took me so much to run the code (i.e., the code takes more than 10 minutes to run; due to the large size of the dataframe). So I decided to **pickle** and **save** the models, so that if you want to do a quick check, you don't wait as I waited. However, if you feel like you want to execute the functions, you can please follow the instructions I've put in the cells. 

You can check my blog post on Medium for the project, by clicking (https://medium.com/@majed.engineers/how-to-predict-customers-against-your-promotions-the-starbucks-promotions-case-study-ed13e811f073)[here]
# Problem Statement
Promotions are very important approach to attract new customers and keep old customers loyal to the brand. It’s a win-to-win situation for both the customer and the company. The customer gets the benifit of getting a discount or someting for free, which lets the customer be bounded more to the brand, and always tried new merchants.

However, the idea of promotions will not be effective if it wasn’t accompanied by educated business decisions. The company must know what kinds of promotions are applicable for the all of its customer base’s demographics. As promotions are good to keep customers loyal, it’s also a good tool to attract new customer base. The latter is very hard to achieve without educated and wise business decisions.

The goal of this project is to analyze the dataset of Starbucks for simulated customer behaviour. Then ML models will be used to create and sythesize prediction models that classify whether the customers will just view the offer, or they will complete it and make use of it.


