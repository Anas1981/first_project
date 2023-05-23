# A Brief about the Project 

This project aims at helping Mr. Charles Christensen, as the principal stakeholder, to make better financial decisions regarding selling and renovating houses with the potential for big investment returns, with a particular attention paid to the property’s immediate neighbourhood and the time needed for the transaction to be profitably concluded. 

## Major Questions and Hypotheses
 1-Question: Does year or month or days has great impact for sales ? 
* Hypothesis : there more sale in month more than another month , there are specific day days has more sale than another

2-Question :Dose renovation of house impact  average price of sales ?
* Hypothesis: renovated house has price more , while non renovated house has lower price

Question 3: Does the property’s location or geographical latitude influence its price?
* Hypothesis: the prices of house near center and river has higher prices , while house located  in south or north has lower prices


 

## Requirements

- pyenv
- python==3.9.8

## Setup

One of the first steps when starting any data science project is to create a virtual environment. For this project you have to create this environment from scratch yourself. However, you should be already familiar with the commands you will need to do so. The general workflow consists of... 

* setting the python version locally to 3.9.8
* creating a virtual environment using the `venv` module
* activating your newly created environment 
* upgrading `pip` (This step is not absolutely necessary, but will save you trouble when installing some packages.)
* installing the required packages via `pip`

At the end, we attached some summeries and recommendation for  Mr. Charles Christensen



### Unit testing (Optional)

If you write python scripts for your data processing methods, you can also write unit tests. In order to run the tests execute in terminal:

```bash
pytest
```

This command will execute all the functions in your project that start with the word **test**.


### Environment

This repo contains a requirements.txt file with a list of all the packages and dependencies you will need. Before you install the virtual environment, 

```bash
altair==4.1.0
jupyterlab==3.2.6
ipywidgets==7.6.5
matplotlib==3.5.1
pandas==1.3.5
jupyterlab-dash==0.1.0a3
seaborn==0.11.1
missingno==0.5.1
```

In order to install the environment you can use the following commands:

```
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```