### Exploration of American Public Transit Association (APTA) and Federal Transit Administration (FTA) data.

Project Under Construction!

[APTA Data Here](https://www.apta.com/resources/statistics/Pages/ridershipreport.aspx)

[FTA Data Here](https://www.transit.dot.gov/ntd/data-product/monthly-module-raw-data-release)

The preprocessor can be used to preprocess and merge the datasets from both sources as they are updated.

## Background

Public transit ridership in the United States has generally been in decline over the last few years. [This has been attributed to a number of factors](https://www.washingtonpost.com/local/trafficandcommuting/falling-transit-ridership-poses-an-emergency-for-cities-experts-fear/2018/03/20/ffb67c28-2865-11e8-874b-d517e912f125_story.html?utm_term=.7996f089b1b8) including a strong economy, lower gas prices and the rise of ride and bike sharing in urban areas.

But while total nationwide ridership is in decline, some transit agencies and modes are doing just fine, experiencing stable or increasing ridership. Why?

This project digs through APTA and FTA agency and ridership data, and using machine learning, looks for clues for what predicts ridership decline. Transit agencies don't have much control over gas prices, competition from ride sharing or changes in commuting behavior, but maybe a systematic comparison of transit agencies can provide actionable insight in how to improve public transportation networks and improve ridership.

A quick overview of the project workflow:

1) **preprocessor.py** cleans and merges the APTA and FTA data into a single csv file

2) **transit_EDA.py** has preliminary exploration of the data, including discovery of reporting some errors in the data

3) **model.py** engineers new features to feed into the model. I tuned model parameters on a few different models before ultimately settling on a LightGBM model.

4) **model_analysis.py** takes the best performing LightGBM model apart, looking at permutation importance to verify the fit, and uses SHAP to explore how the model makes predictions, and finds actionable advice.

## Data preprocessing

The preprocessor mostly deals with cleaning up strings representing numeric features (fares expressed with $ ect.), correcting data types, and merging the APTA and FTA datasets.

During exploratory data analysis (EDA) I noticed there were a few agencies with extremely high population density in the service areas. This was usually because the entire metro area of a community was counted in the population but only the headquarter city area was counted as the area, resulting in a inaccurate population density. I corrected the most extreme examples of this error in the preprocessor.

## Exploratory Data Analysis

There are 607 agencies in the data set, operating in 561 municipalities in 53 states and US territories.  

The first unexpected feature from EDA was how many different transit agencies operate in a single city. There 59 different agencies operating in the greater New York area! Do fragmented agencies help, hurt, or have no impact in ridership? This could be a feature worth extracting.
