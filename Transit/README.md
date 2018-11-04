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

**Objective 1:** Explore relationship between Service Area, Service Area Population, Number of Passenger Trips, and Passenger Miles. Do denser cities have better ridership?

First, while I usually think of the BART, Chicago's 'L'' or the NYC subway when thinking about mass transit, the majority of agencies serve communities with less than 1 million people. America's most famous transit agencies in this respect are really outliers.

<p align="center">
  <img width="576" height="396" src="">
</p>

Population density and total passenger trips do appear to be correlated, but not as closely as I expected.

<p align="center">
  <img width="576" height="396" src="">
</p>

**Objective 2:** What are the general trends in ridership in the US? When did ridership start declining, and what contributed to that decline?

First, a the ridership data does note that while the data is fairly complete, some agencies did not report complete monthly data. To get around this, I interpolated missing values, as long as the number of missing months were below a threshold. If there were too many missing months, I assume there really weren't any riders (for example, a line hadn't been built yet).

I tried a few different thresholds, but by assuming if more than ~5 years were missing I ridership was actually 0 and interpolating ridership for missing data below 5 years, my estimates of total ridership roughly matched APTA's estimates. Since I'm currently only looking in change in ridership, this rough estimate is probably ok.

Total ridership increased until about 2014, then began declining. There is a fair amount of seasonal variation in ridership, but the trend clear:

<p align="center">
  <img width="720" height="720" src="">
</p>

Breaking the ridership down by the most common modes, its clear the decline in ridership is due to the single most common mode- regular busses (MB). Heavy rail (HR), commuter rail (CR) and light rail (LR) held steady or saw slight increases in recent years:

<p align="center">
  <img width="720" height="720" src="">
</p>

Does the population and geographical size of an urban zone area (UZA) served by an agency contribute to the decline? Interestingly, agencies serving a large area (>800 square miles) saw ridership hold steady, while smaller areas experiences a decline. By grouping the agencies by population served, it becomes clear that most of the decline in ridership is due to declines in smaller communities (~ < 1 million people):

<p align="center">
  <img width="720" height="720" src="">
</p>

<p align="center">
  <img width="720" height="720" src="">
</p>

**Objective 3:** What is the relationship between costs, agency revenue and ridership?

First, how many agencies loose money and how many make money? A quick look at net revenue per trip shows the agencies that make money tend to be smaller, privately owned charters and some university bus systems (although those may be subsidized). The largest loss makers are generally specialty services for the disabled and small city transit systems (such as New Haven, CT).

<p align="center">
  <img width="576" height="396" src="">
</p>

How about operating costs and fares? Here, I broke down the agencies into fare and cost-per-trip quintiles and plotted ridership from 2002-2018. In both cases, the highest cost and highest fare agencies only experiences slight declines, while most of the total ridership decline came from low-fare, lower cost-per-rider agencies.

<p align="center">
  <img width="720" height="720" src="">
</p>

<p align="center">
  <img width="720" height="720" src="">
</p>

At first this may seem counter-intuitive, but it suggests that longer-distance commuter systems (such as commuter rail or intercity busses) are doing relatively well, while inexpensive local routes are struggling.

## Modeling

The file model.py includes the machine learning models I explored to see if one could reasonably fit to the data. The US transit agency data set is relatively small, so my greatest concern was overfitting to the training data and creating a model that doesn't generalize well (the intent of the project was to find general guidance transit agencies could implement, rather than agency-specific recommendations). My hunch was to stick to simple models and avoid boosted algorithms that would just end up overfitting to the small training set, but it turned out that boosted models (XGBoost and LightGBM) performed better.

The modeling steps are:

**1) Create a Target** I made this problem a binary classification problem - an agency's ridership is either stable/increasing [1] or decreasing [0]. Total US ridership peaked and was roughly was steady from  2007-2013, and began declining from 2014-2017. I averaged ridership by agency over the two periods and compared them. If average 2014-2017 ridership was greater than 95% of the 2007-2013 average ridership, that agency is considered stable or increasing [1], if recent ridership is less than 95% of past ridership, that agency is considered decreasing [0].

**2) Select a Performance Metric** The data set is a little imbalanced, with about 58% of agencies stable [1] and 42% declining [0], so accuracy could work as a performance metric, but I chose Area Under the ROC Curve (AUC-ROC) as a good general metric that would not be influenced by the slight imbalance in the data set.

**3) Feature Engineering** Creating new features that pick up on relationships between features is a great way to 'add' information to your data set and give the models more to learn with. I one-hot encoded the categorical features (which type of transportation an agency has (Bus, Light Rail, Ferry etc.) and the state or territory the agency is in).

I then created ~15 numeric features which are detailed in the code, but include the number of other agencies operating in the same city and ratios like the number of passenger trips per mile in the network and the population density of the service area.

**3) Scale Data** Some models work better if the data is scaled to make it normally distributed. In this case, scaling the data did not have much of an effect.

**5) Unsupervised Learning** To visualize if 'natural' clusters exist in the full training data set before fitting a model, I explored t-sne and PCA 2-D visualizations of the data set. I didn't find much, indicating finding a good fit may be tricky with this data set.

**6) Fit and Evaluate Models** I tried a few different models using the following general approach:

    - Tune model parameters using scikit-learn's GridSearchCV. I also tried some 'manual' parameter tuning once I had the general combination of parameters from the grid search.

    - The small size of the data set means the test-set performance varies quite a bit depending on how the train and test data is shuffled. To get a more consistent AUC-ROC score, I scored the best model by fitting it to and scoring it on 10 different train-test-split shuffles, and averaging the test scores.

The general best AUC-ROC scores were:

    - Logistic Regression: 0.55
    - Naive Bayes - 0.51
    - K-Nearest Neighbors - 0.53
    - Random Forest - 0.58
    - LightGBM - 0.63
    - XGBoost - 0.58

While none of these fits are great, there may be enough information in the LightGBM model to extract some insights into what is going on!

## Model Analysis

This is the best part - what did the LightGBM model use to fit to the transit agency data, and what can it tell us?

First, its worth checking whether the 'fit' is real. ELI5's permutation importance module is a great way to see if the 'fit' is real - we check if the model performance is negatively impacted by shuffling the values of various features in the test set:

<p align="center">
  <img width="270" height="384" src="">
</p>

Whew! Shuffling the data in significant features does hurt the model, indicating the fit is real.

Next, we can use SHAP to visualize how different features impact the model prediction. Since the fit depends on how the train and test sets are shuffled (small data problems...), I re-ran the model on 100 different shuffles and extracted the most important features over the various shuffles. Here are the top 20:

    Service_Area_Population - Population served by the agency
    service_to_uza_area - Ratio of the agency service area (sq miles) to the area of the city served (some agencies serve part of a city, some agencies serve a city and surrounding areas)
    UZA_Area_SQ_Miles - Area of the agency's city
    UZA_Population - Population of the agency's city
    agencies_per_city - Number of ransit agencies operating in that city
    Service_Area_SQ_Miles - Service are square miles
    cost_per_person - Total fares divided by total rides
    cost_per_mile - Total fares divided by total distance traveled in a year
    trips_per_mile - Number of passenger trips per mile traveled
    fares_per_mile - Total fares divided by total miles traveled
    service_to_uza_pop - Same as 2, but with population
    service_area_pop_density - Population density of area served
    miles_per_trip - Inverse of trips per mile
    UZA_pop_density - City's population density
    Fares_FY - Total fares collected
    net_per_trip - Net revenue per trip (Fare - Cost)
    net_per_mile - Net revenue per mile
    Passenger_Miles_FY -
    MA - Is the agency in Massachusetts?
    fare_per_trip - Fares per passenger trip
    Operating_Expenses_FY - Total cost to operate the agency per year

And here is the SHAP summary plot:

<p align="center">
  <img width="633" height="584" src="">
</p>

Some are a little difficult to see from the summary plot, but a few recommendations are apparent:

    2 - service_to_uza_area: If the service area is greater than the headquarter city area, the agency is more likely to maintain ridership. This suggests intercity systems and longer distance commuter systems are services agencies should pursue.

    5 - agencies_per_city: Interestingly, having fewer agencies in one city actually negatively impacts the predictions. This could just reflect agencies in smaller communities not doing as well, or it just indicates that having multiple agencies competing in one jurisdiction actually helps things. Is there a network effect? Or does the causality go the other way - do cities already favorable to public transportation (like NYC) just have more agencies?

    8 & 9 - cost_per_mile and fare_per_mile: This one is also counter intuitive. Higher costs (and fares) per mile actually indicate better ridership? My hunch  

    18 - MA - The state of Massachusetts might be up to something - having your agency there is a string predictor of stable ridership. This could just be coincidence (there are only ~10 agencies there), or it could reflect something about the state such a state policy that is helping transit agencies. Its probably not gas prices - the states with the highest gas prices (CT, CA, WA, OR) don't show up in the model fit!


What's not on the top 20 list? Surprisingly, the model isn't using the types of services available (heavy rail, light rail, demand response etc.) to predict ridership as much as I would think. Apparently to this model the layout and cost of the network matter more than the actual vehicles used.

This is only the interpretation of one model, which admittedly only had a mediocre fit. We probably could get better results and more insight by fitting different models to the data - a deep learning model may find feature interactions that I didn't engineer.
