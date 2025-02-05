This repository contains an end-to-end implementation of MLOps best 
practices for deploying machine learning models at scale using 
cloud-native solutions. The project covers data processing, model training, 
deployment, and monitoring to ensure efficient and scalable ML workflows.

**Project Scope**

The goal of this project is to create a scalable, automated, and 
reliable MLOps pipeline that can be used for deploying machine learning 
models in a production environment. The solution integrates cloud services, 
containerization, and automation techniques to ensure robustness and efficiency.

**Features**

* Data Extraction, Transformation, and Loading (ETL) for ML models.
* Efficient querying and data handling with SQL.
* Model training and tuning using PyTorch.
* Continuous monitoring and performance evaluation.

**Problem**

You are a machine learning engineer at a startup aiming to launch an 
autonomous ride-sharing service in Washington, DC, competing with 
companies like Waymo, Uber, and Lyft. To offer competitive pricing, 
you are tasked with developing code to estimate taxi fares in Washington, 
DC, and nearby areas in Virginia and Maryland.

**Data collection and preparation**

We will be using data from The Office of the Chief Technology Officer for Washington, DC [Open Data DC](https://opendata.dc.gov/search) . 
The historical dataset of taxi rides from 2015-2019 will be used to build ML models to estimate how much it cost to travel by taxi around DC.

**Data Versioning**

To validate and prepare the data, AWS s3 will be used for object storage for the dataset.
The AWS glue crawler service will be used for data analysis and automated schema discovery.
Once the glue crawler service has processed the dataset, it can populate the databse with a table containing 
the data schema. 

**Dataset Optimizations**

Converting the CVS files to Column Orientated data format will allow for more efficient data
analytics using glue. NULL values are removed from the dataset using AWS athena. 

**Data Training Pipeline**

This project implements a deep learning model for taxi fare prediction using PyTorch Lightning. The model is designed for scalability, leveraging cloud-based distributed training techniques.

* Data Loading
    * Uses ObjectStorageDataset (osds) for efficient data retrieval.
    * Loads datasets into PyTorch DataLoader.

* Model Training
    * Trains using PyTorch Lightning Trainer, enabling distributed training.
    * Logs training performance using built-in monitoring tools.

* Hyperparameters
    * Number of features and hidden neurons.
    * Batch normalization toggle.
    * Optimizer selection (Adam / SGD).
    * Learning rate and batch size.

* Evaluation
   * Computes validation and test loss (MSE).
   * Logs results for monitoring.
