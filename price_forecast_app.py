import os
import json
import joblib
import pandas as pd
import pickle
from flask import Flask, jsonify, request,Response
from peewee import SqliteDatabase, Model, IntegerField, FloatField, TextField, IntegrityError, CharField,AutoField,CompositeKey
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import numpy as np
import requests
import holidays
import joblib
from collections import OrderedDict

########################################
# Initialize database
DB = connect(os.environ.get("DATABASE_URL") or "sqlite:///predictions.db")
# DB = SqliteDatabase('predictions.db')

class PricePrediction(Model):
    sku = CharField()
    time_key = IntegerField()
    pvp_is_competitorA = FloatField(null=True)
    pvp_is_competitorB = FloatField(null=True)
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        primary_key = CompositeKey('sku', 'time_key')

# if PricePrediction.table_exists():
#         PricePrediction.drop_table()

DB.create_tables([PricePrediction], safe=True)

########################################
# Functions: prepare_data, load_model and get_predictions 

def search_sku(sku_input):
    df_trained = pd.read_csv("prepared_data/data_history.csv")
    if len(df_trained.loc[df_trained.sku==sku_input])>0:
        return 1
    else:
        return 0

def prepare_data(sku_input, date_input):
    """Loads & preprocesses data from given test folder."""
    
    # 1. Read history data
    df_trained = pd.read_csv("prepared_data/data_history.csv")
    df_trained['date'] = pd.to_datetime(df_trained['date'])
    df_trained['last_promo_date'] = pd.to_datetime(df_trained['last_promo_date'])
    max_date_history = pd.to_datetime(df_trained['date'].max())

    # 2. Get predictions range dates
    if pd.to_datetime(date_input) > max_date_history:
        date_pred_range = list(pd.date_range(start=max_date_history+ pd.Timedelta(days=1), end=date_input, freq='D').strftime('%Y-%m-%d'))
    else:
        date_pred_range = list([max_date_history])
    df_date_pred_range = pd.DataFrame(date_pred_range, columns=['date'])
    df_date_pred_range['date'] = pd.to_datetime(df_date_pred_range['date'])

    # 3. Prepare test data set
    ## 3.1 Features from history data
    df_test = (df_trained.loc[df_trained.sku==sku_input][['sku', 'competitor', 'structure_level_1','abc', 'xyz', 'seil', 'structure_level_3',
                                                        'promo_part_w_sku', 'avg_discount_w_sku', 'avg_discount_w_L3', 'promo_part_w_L3']]
                                                        .drop_duplicates())
    ## 3.2 Lag_7 features (if date_pred_range <7, else we need to fill them with recursive forecast)
    df_test = df_test.merge(df_date_pred_range, how='cross')
    df_test["date_lag_7"] = df_test["date"] - pd.to_timedelta(7, unit="D")
    ### lag1 usefull in the model recursive forecast 
    df_test["date_lag_1"] = df_test["date"] - pd.to_timedelta(1, unit="D")

    ## 3.3 Days since last promo feature (for the 1st day to predict, the other dates need to be derive from recursive forecast)
    df_last_promo_dates = (df_trained
                        .groupby(["sku","competitor"])["last_promo_date"].max()
                        .reset_index())

    df_test = (df_test.merge((df_trained[['sku', 'date', 'competitor', 'pvp_is', 'discount']]
                                .rename(columns={'date':'date_lag_7',
                                                'pvp_is':'pvp_is_lag_7',
                                                'discount':'discount_lag_7'})), on=['sku', 'date_lag_7','competitor'], how='left')
                    .merge(df_last_promo_dates, on=['sku', 'competitor'], how='left'))
    df_test["days_since_last_promo"] = (df_test["date"] - df_test["last_promo_date"]).dt.days
    df_test["days_since_last_promo"] = df_test["days_since_last_promo"].clip(lower=0)

    # 4. Calendar Features
    df_test['month'] = df_test['date'].dt.month
    df_test['day_of_week'] = df_test['date'].dt.dayofweek  # Monday=0, Sunday=6
    df_test['week_of_month'] = df_test['date'].apply(lambda d: (d.day - 1) // 7 + 1)

    # 5. Add Holidays & Weather Data
    df_holidays = pd.DataFrame(holidays.Portugal(years=range(df_test["date"].min().year, df_test["date"].max().year + 1), subdiv="11").items(), columns=["date", "holiday"])
    df_holidays["date"] = pd.to_datetime(df_holidays["date"])
    df_holidays["holiday_importance"] = df_holidays["holiday"].str.lower().str.contains("christmas|new year|easter", regex=True).astype(int) + 1
    df_holidays = df_holidays[['date', 'holiday_importance']]

    ## If date is < today -5 days (weather api delay)
    today_minus_delay = pd.Timestamp.today().normalize() - pd.Timedelta(days=5)
    if pd.to_datetime(date_input) <= today_minus_delay:
        start_date = df_test["date"].min().strftime("%Y-%m-%d")
        end_date = df_test["date"].max().strftime("%Y-%m-%d")
    else:
        ## Read until max date in history and then read ly for the remaining dates
        start_date = df_test["date"].min().strftime("%Y-%m-%d")
        end_date = today_minus_delay.strftime("%Y-%m-%d")

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 39.6945,
            "longitude": -8.1306,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["apparent_temperature_mean", "precipitation_sum"],
            "timezone": "Europe/London"
        }
        r = requests.get(url, params=params)
        r.raise_for_status() 
        jsondata = r.json()
        daily_data = jsondata['daily']
        df_weather = (pd.DataFrame.from_dict(daily_data))
        ## Remove decimal cases
        df_weather['date'] = pd.to_datetime(df_weather['time'])
        df_weather['apparent_temperature_mean'] = df_weather['apparent_temperature_mean'].astype("int")
        df_weather['precipitation_sum'] = df_weather['precipitation_sum'].astype("int")
    except Exception as e:
        print("Error while trying to extract weather info ")

    df_test = (df_test.merge(df_weather.drop('time', axis=1), on=['date'], how='left'))
    
    ## read ly weather
    if pd.to_datetime(date_input) > today_minus_delay:
        start_date = ((today_minus_delay + pd.Timedelta(days=1))- pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        end_date = (df_test["date"].max() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        print("start_date: ",start_date)
        print("end_date: ",end_date)

        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": 39.6945,
                "longitude": -8.1306,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ["apparent_temperature_mean", "precipitation_sum"],
                "timezone": "Europe/London"
            }
            r = requests.get(url, params=params)
            r.raise_for_status() 
            jsondata = r.json()
            daily_data = jsondata['daily']
            df_weather = (pd.DataFrame.from_dict(daily_data))
            ## Remove decimal cases
            df_weather['date'] = pd.to_datetime(df_weather['time'])
            df_weather = df_weather.rename(columns={"date":"date_ly"})
            df_weather['date'] = df_weather['date_ly'] + pd.DateOffset(years=1)
            df_weather['apparent_temperature_mean_ly'] = df_weather['apparent_temperature_mean'].astype("int")
            df_weather['precipitation_sum_ly'] = df_weather['precipitation_sum'].astype("int")
        except Exception as e:
            print("Error while trying to extract weather info ")

        ## Merge ly weather data
        df_test = (df_test.merge(df_weather[['date', 'apparent_temperature_mean_ly', 'precipitation_sum_ly']], on=['date'], how='left'))
        df_test["apparent_temperature_mean"] = df_test["apparent_temperature_mean"].fillna(df_test["apparent_temperature_mean_ly"])
        df_test["precipitation_sum"] = df_test["precipitation_sum"].fillna(df_test["precipitation_sum_ly"])
        df_test = df_test.drop(['apparent_temperature_mean_ly', 'precipitation_sum_ly'], axis=1)
        

    # 6. Add last pvp_was from history
    df_trained = df_trained.sort_values(by=['date'])
    df_last_pvp_was = (df_trained.groupby(['sku', 'competitor']).tail(1)
                                 .reset_index(drop=True)
                                 [['sku','competitor', 'pvp_was', 'pvp_was_chain']]
                                 .rename(columns={"pvp_was":"last_pvp_was_train",
                                                  "pvp_was_chain":"last_pvp_was_chain_train"}))

    df_test = (df_test.merge(df_holidays, on=['date'], how='left')
                      .merge(df_last_pvp_was,on=['sku', 'competitor'], how='left'))
    df_test[['holiday_importance']] = df_test[['holiday_importance']].fillna(value=0)
    df_test['structure_level_1'] = df_test['structure_level_1'].astype("str")
    df_test['structure_level_3'] = df_test['structure_level_3'].astype("str")
    
    return max_date_history, df_test

def load_model(sku_input, df_test):
    """Loads model for the input sku"""

    TMP_DIR = "models"
    # 1. Get structure_level_1 for given sku
    struct = df_test.loc[df_test.sku==sku_input]['structure_level_1'].head()[0]

    # 2️. Build model filename based on structure_level_1 & competitor
    model_compA_filename = f"final_model_it_structL1_{struct}_competitorA.pickle"
    model_compA_path = os.path.join(TMP_DIR, model_compA_filename)
    model_compB_filename = f"final_model_it_structL1_{struct}_competitorB.pickle"
    model_compB_path = os.path.join(TMP_DIR, model_compB_filename)

    # 3️. Load model if file exists, otherwise raise error
    if os.path.exists(model_compA_path):
        print(f"Loading model: {model_compA_filename}")
        model_compA = joblib.load(model_compA_path)
    else:
        raise FileNotFoundError(f"Model file '{model_compA_filename}' not found in '{TMP_DIR}'. Check model existence.")

    if os.path.exists(model_compB_path):
        print(f"Loading model: {model_compB_filename}")
        model_compB = joblib.load(model_compB_path)
    else:
        raise FileNotFoundError(f"Model file '{model_compB_filename}' not found in '{TMP_DIR}'. Check model existence.")
    
    return model_compA, model_compB

def get_predictions(model_compA, model_compB, df_test, sku_input, date_input, max_date_history):
    """ Generate predictions for provided models, sku and date """

    df_test['discount_pred_competitorA'] = None
    df_test['discount_pred_competitorB'] = None
    df_test['pvp_is_competitorA'] = None
    df_test['pvp_is_competitorB'] = None
    for d in df_test['date'].drop_duplicates():
        # print("\n\n******", d)
        df_test = df_test.sort_values(["sku", "competitor", "date"])

        ## 4.1 Check if lag7 features need to be filled
        if len(df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.discount_lag_7.isnull()))])>0:
            df_test = (df_test.merge((df_test[['sku', 'competitor', 'date', 'discount_pred_competitorA', 'discount_pred_competitorB', 'pvp_is_competitorA', 'pvp_is_competitorB']]
                                            .rename(columns={'date':'date_lag_7',
                                                            'discount_pred_competitorA':'discount_pred_competitorA_lag_7',
                                                            'discount_pred_competitorB':'discount_pred_competitorB_lag_7',
                                                            'pvp_is_competitorA':'pvp_is_competitorA_lag_7',
                                                            'pvp_is_competitorB':'pvp_is_competitorB_lag_7'})), on=['sku', 'competitor', 'date_lag_7'], how='left'))
            df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorA')), 'discount_lag_7'] = df_test['discount_pred_competitorA_lag_7'].astype(float)
            df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorA')), 'pvp_is_lag_7'] = df_test['pvp_is_competitorA_lag_7'].astype(float)

            df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorB')), 'discount_lBg_7'] = df_test['discount_pred_competitorB_lag_7'].astype(float)
            df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorB')), 'pvp_is_lagB7'] = df_test['pvp_is_competitorB_lag_7'].astype(float)

            df_test = df_test.drop(['discount_pred_competitorA_lag_7', 'discount_pred_competitorB_lag_7', 'pvp_is_competitorA_lag_7', 'pvp_is_competitorB_lag_7'], axis=1)
            
        ## 4.2 Get predictions
        df_compA = df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorA'))]
        df_compB = df_test.loc[((df_test.sku==sku_input) & (df_test.date==d) & (df_test.competitor=='competitorB'))]

        ## try to predict for competitor A
        if len(df_compA)>0:
            if pd.to_datetime(date_input) > max_date_history:
                discount_pred_competitorA = float(model_compA.predict(df_compA)[0])
                df_test.loc[((df_test.sku==sku_input) & (df_test.date==d)), 'discount_pred_competitorA'] = discount_pred_competitorA
                ## 4.3 Convert from discount to price prediction
                df_test.loc[((df_test.sku==sku_input) & (df_test.date==d)), 
                        'pvp_is_competitorA'] = (df_test['last_pvp_was_train']*(1-df_test['discount_pred_competitorA']))
                pvp_is_competitorA = round(float(df_test.loc[((df_test.sku==sku_input) & 
                                                (df_test.date==d) & 
                                                (df_test.competitor=='competitorA'))].pvp_is_competitorA.iloc[0]),2)
            else:
                ## if date is not in the future 
                discount_pred_competitorA = 0
                pvp_is_competitorA =  round(df_test.loc[((df_test.sku==sku_input) & (df_test.date==d))].last_pvp_was_train.iloc[0],2)
        else:
            ## if there are no records for this competitorA x sku then return chains' price
            discount_pred_competitorA = 0
            pvp_is_competitorA =  round(df_test.loc[((df_test.sku==sku_input) & (df_test.date==d))].last_pvp_was_chain_train.iloc[0],2)

        ## try to predict for competitor B
        if len(df_compB)>0:
            if pd.to_datetime(date_input) > max_date_history:
                discount_pred_competitorB = float(model_compB.predict(df_compB)[0])
                df_test.loc[((df_test.sku==sku_input) & (df_test.date==d)), 'discount_pred_competitorB'] = discount_pred_competitorB
                ## 4.3 Convert from discount to price prediction
                df_test.loc[((df_test.sku==sku_input) & (df_test.date==d)), 
                        'pvp_is_competitorB'] = (df_test['last_pvp_was_train']*(1-df_test['discount_pred_competitorB']))
                pvp_is_competitorB = round(float(df_test.loc[((df_test.sku==sku_input) & 
                                                (df_test.date==d) & 
                                                (df_test.competitor=='competitorB'))].pvp_is_competitorB.iloc[0]),2)
            else:
                ## if date is not in the future 
                discount_pred_competitorB = 0
                pvp_is_competitorB =  round(df_test.loc[((df_test.sku==sku_input) & (df_test.date==d))].last_pvp_was_train.iloc[0],2)
        else:
            ## if there are no records for this competitorA x sku then return chains' price
            discount_pred_competitorB = 0
            pvp_is_competitorB =  round(df_test.loc[((df_test.sku==sku_input) & (df_test.date==d))].last_pvp_was_chain_train.iloc[0],2)


        ## 4.4 Update days_since_last_promo (if predicted discount >0 and if there are more dates to predict)
        if len(df_test.loc[df_test.date>d])>0:
            if round(discount_pred_competitorA, 2)>0:
                # print("\n update days_since_last_promo for compA")
                df_test.loc[((df_test.sku==sku_input) & (df_test.competitor=='competitorA')), 
                            'days_since_last_promo'] = (df_test.groupby(["sku", "competitor"])["date"]
                                                                .transform(lambda dates: (dates > d).cumsum()*(dates > d)))

            if round(discount_pred_competitorB, 2)>0:
                # print("\n update days_since_last_promo for compB")
                df_test.loc[((df_test.sku==sku_input) & (df_test.competitor=='competitorB')), 
                            'days_since_last_promo'] = (df_test.groupby(["sku", "competitor"])["date"]
                                                                .transform(lambda dates: (dates > d).cumsum()*(dates > d)))
    
    return pvp_is_competitorA,pvp_is_competitorB

########################################
# Begin web server setup

app = Flask(__name__)

@app.route("/forecast_prices/", methods=["POST"])
def forecast_prices():
    obs_dict = request.get_json()
    print("Received request data (forecast_prices): \n", obs_dict)

    if "sku" not in obs_dict or "time_key" not in obs_dict:
        return jsonify({"error 1": "Missing required fields"}), 422

    sku_input = obs_dict["sku"]
    time_key_input = obs_dict["time_key"]

    if not isinstance(sku_input, str) or not isinstance(time_key_input, int):
        return jsonify({
            "error 2": "Invalid input format",
            "sku_input_type": str(type(sku_input)),
            "time_key_input_type": str(type(time_key_input))
        }), 422
    
    sku_input = int(sku_input)

    # Search for sku (if not in the database return error message)
    if search_sku(sku_input)==0:
        return jsonify({
            "error 3": "Unknown sku",
            "sku": str(sku_input)
        }), 422


    # Try to convert date input into datetime
    try:
        date_input = pd.to_datetime(str(time_key_input), format='%Y%m%d').strftime('%Y-%m-%d')
    except:
        return jsonify({
            "error 4": "Date input is not a valid date",
            "time_key": str(time_key_input)
        }), 422

    # Generate Predictions
    try:
        max_date_history, df_test = prepare_data(sku_input, date_input)

        model_compA, model_compB = load_model(sku_input, df_test)
        pvp_is_competitorA, pvp_is_competitorB = get_predictions(model_compA, model_compB, df_test, sku_input, date_input, max_date_history)
    except Exception as e:
        return jsonify({"error 5": f'Prediction failed for sku "{sku_input}" and time_key {time_key_input}'}), 422

    sku_input = str(sku_input)
    try:
        # p.save()
        PricePrediction.create(
            sku=sku_input,
            time_key=time_key_input,
            pvp_is_competitorA=pvp_is_competitorA,
            pvp_is_competitorB=pvp_is_competitorB,
        )
    except IntegrityError:
        PricePrediction.update(
            pvp_is_competitorA=pvp_is_competitorA,
            pvp_is_competitorB=pvp_is_competitorB,
        ).where(
            PricePrediction.sku == sku_input,
            PricePrediction.time_key == time_key_input
        ).execute()
        # return jsonify({"updated": f'Observation sku "{sku_input}" and time_key {time_key_input} already in the database but was updated.'}), 422

    response = OrderedDict([
        ("sku", sku_input),
        ("time_key", time_key_input),
        ("pvp_is_competitorA", pvp_is_competitorA),
        ("pvp_is_competitorB", pvp_is_competitorB)
    ])

    return Response(
        json.dumps(response, indent=2),
        status=200,
        mimetype='application/json'
        )

@app.route("/actual_prices/", methods=["POST"])
def actual_prices():
    obs_dict = request.get_json()
    print("Received request data (actual_prices)\n:", obs_dict)

    if "sku" not in obs_dict or "time_key" not in obs_dict or "pvp_is_competitorA_actual" not in obs_dict or "pvp_is_competitorB_actual" not in obs_dict:
        return jsonify({"error": "Missing required fields"}), 422

    sku = obs_dict["sku"]
    time_key = obs_dict["time_key"]
    pvp_is_competitorA_actual = obs_dict["pvp_is_competitorA_actual"]
    pvp_is_competitorB_actual = obs_dict["pvp_is_competitorB_actual"]

    if not isinstance(sku, str) or not isinstance(time_key, int) or not isinstance(pvp_is_competitorA_actual, float) or not isinstance(pvp_is_competitorB_actual, float):
        return jsonify({"error": "Invalid input format"}), 422

    try:
        p = PricePrediction.get((PricePrediction.sku == sku) & (PricePrediction.time_key == time_key))
        p.pvp_is_competitorA_actual = pvp_is_competitorA_actual
        p.pvp_is_competitorB_actual = pvp_is_competitorB_actual
        p.save()

        response = OrderedDict([
            ("sku", p.sku),
            ("time_key", p.time_key),
            ("pvp_is_competitorA", p.pvp_is_competitorA),
            ("pvp_is_competitorB", p.pvp_is_competitorB),
            ("pvp_is_competitorA_actual", p.pvp_is_competitorA_actual),
            ("pvp_is_competitorB_actual", p.pvp_is_competitorB_actual),
        ])

        ## Print MAPE and BIAS
        print("BIAS compA: ", round((p.pvp_is_competitorA-pvp_is_competitorA_actual)/pvp_is_competitorA_actual,4))
        print("BIAS compB: ", round((p.pvp_is_competitorB-pvp_is_competitorB_actual)/pvp_is_competitorB_actual,4))

        return Response(
            json.dumps(response, indent=2),
            status=200,
            mimetype='application/json'
        )

    except PricePrediction.DoesNotExist:
        return jsonify({"error": f'Observation sku "{sku}" and time_key {time_key} does not exist'}), 422
    
@app.route("/debug_db", methods=["GET"])
def debug_db():
    rows = [model_to_dict(p) for p in PricePrediction.select()]
    return jsonify(rows)

#port=int(os.environ.get("PORT", 5100))
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5100)