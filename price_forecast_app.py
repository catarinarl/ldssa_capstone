import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request,Response
from peewee import SqliteDatabase, Model, IntegerField, FloatField, TextField, IntegrityError, CharField,AutoField,CompositeKey
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import os
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

DB.create_tables([PricePrediction], safe=True)

########################################
# Functions: prepare_data, load_model and get_predictions 

def prepare_data(data_folder):
    """Loads & preprocesses data from given test folder."""
    
    # 1. Read data
    df_prod_sales = pd.read_csv(f"{data_folder}/product_structures_sales.csv")
    df_prices = pd.read_csv(f"{data_folder}/product_prices_leaflets.csv")
    df_chains = pd.read_csv(f"{data_folder}/chain_campaigns.csv")

    # TESTEEE!!! ignore columns pvp_was, discount, flag_promo, leaflet (é esperado q eles removam ao enviar os dados) -> mas isto seria só para os competitors
    df_prices.loc[df_prices['competitor'].isin(['competitorA', 'competitorB']), 'discount']=None
    df_prices.loc[df_prices['competitor'].isin(['competitorA', 'competitorB']), 'pvp_was']=None
    df_prices.loc[df_prices['competitor'].isin(['competitorA', 'competitorB']), 'flag_promo']=None
    df_prices.loc[df_prices['competitor'].isin(['competitorA', 'competitorB']), 'leaflet']=None

    # 2. Convert time_key to datetime
    df_prod_sales["date"] = pd.to_datetime(df_prod_sales["time_key"].astype(str), format="%Y%m%d")
    df_prices["date"] = pd.to_datetime(df_prices["time_key"].astype(str), format="%Y%m%d")
    df_chains["start_date"] = pd.to_datetime(df_chains["start_date"])
    df_chains["end_date"] = pd.to_datetime(df_chains["end_date"])
    df_prod_sales['structure_level_1'] = df_prod_sales['structure_level_1'].astype("str")
    df_prod_sales['structure_level_2'] = df_prod_sales['structure_level_2'].astype("str")
    df_prod_sales['structure_level_3'] = df_prod_sales['structure_level_3'].astype("str")
    df_prod_sales['structure_level_4'] = df_prod_sales['structure_level_4'].astype("str")

    # 3. Explode chain_campaigns by date
    df_chains["date_range"] = df_chains.apply(lambda row: pd.date_range(start=row["start_date"], end=row["end_date"]), axis=1)
    df_chains = df_chains.explode("date_range").rename(columns={"date_range": "date"}).drop(["start_date", "end_date"], axis=1)

    # 4. Merge datasets
    df = df_prices.merge(df_prod_sales.drop("time_key", axis=1), on=["sku", "date"], how="left").merge(df_chains, on=["competitor", "date"], how="left")

    # 5. Remove rows where quantity is missing
    df = df.dropna(subset=["quantity"]).drop("time_key", axis=1)

    # 6. Add Holidays & Weather Data
    df_holidays = pd.DataFrame(holidays.Portugal(years=range(df["date"].min().year, df["date"].max().year + 1), subdiv="11").items(), columns=["date", "holiday"])
    df_holidays["date"] = pd.to_datetime(df_holidays["date"])
    df_holidays["holiday_importance"] = df_holidays["holiday"].str.lower().str.contains("christmas|new year|easter", regex=True).astype(int) + 1

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 39.6945,
            "longitude": -8.1306,
            "start_date": df["date"].min().strftime("%Y-%m-%d"),
            "end_date": df["date"].max().strftime("%Y-%m-%d"),
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
    except requests.exceptions.HTTPError as errh: 
        print("HTTP Error while trying to extract weather info ")
        print(errh.args[0])
        raise
    except requests.exceptions.ReadTimeout as errrt: 
        print("Time out while trying to extract weather info")
        print(errrt.args[0])
        raise
    except requests.exceptions.ConnectionError as conerr: 
        print("Connection error while trying to extract weather info ") 
        print(conerr.args[0])
        raise

    # 7. Add new features
    days_per_group = df.groupby('structure_level_1')['date'].nunique().reset_index()
    days_per_group.columns = ['structure_level_1', 'n_days']

    demand_stats = (
        df.groupby(['structure_level_1', 'sku'])
        .agg(count_days=('date', 'nunique'),
            quantity=('quantity', 'sum'))
        .reset_index()
    )
    demand_stats = demand_stats.merge(days_per_group, on='structure_level_1')
    demand_stats['average_demand_interval'] = demand_stats['n_days'] / demand_stats['count_days']

    ## 7.1 Product classes
    # ABC classification
    demand_stats['rank'] = demand_stats.groupby('structure_level_1')['quantity'] \
                                    .rank(method='first', ascending=False)
    demand_stats = demand_stats.sort_values(['structure_level_1', 'rank'])
    demand_stats['total_quantity'] = demand_stats.groupby('structure_level_1')['quantity'].transform('sum')
    demand_stats['cumulative_quantity'] = demand_stats.groupby('structure_level_1')['quantity'].cumsum()
    demand_stats['cumulative_pct'] = demand_stats['cumulative_quantity'] / demand_stats['total_quantity']
    demand_stats['abc'] = np.select(
        [demand_stats['cumulative_pct'] <= 0.80,
        demand_stats['cumulative_pct'] <= 0.95],
        ['A', 'B'],
        default='C')

    ## XYZ classification
    unique_dates = df['date'].drop_duplicates()
    unique_sku_pairs = df[['structure_level_1', 'sku']].drop_duplicates()
    sku_date_combinations = unique_sku_pairs.merge(unique_dates.to_frame(), how='cross')

    df_slim = df[['structure_level_1', 'sku', 'date', 'quantity']]
    df_full = sku_date_combinations.merge(df_slim, on=['structure_level_1', 'sku', 'date'], how='left')
    df_full['quantity'] = df_full['quantity'].fillna(0)

    sales_stats = (
        df_full.groupby(['structure_level_1', 'sku'])
        .agg(std_sales=('quantity', 'std'),
            avg_sales=('quantity', 'mean'))
        .reset_index())

    sales_stats = sales_stats.merge(
        demand_stats[['structure_level_1', 'sku', 'average_demand_interval', 'abc']],
        on=['structure_level_1', 'sku'],
        how='left')

    sales_stats['cv'] = sales_stats['std_sales'] / sales_stats['avg_sales']
    sales_stats['cv_2'] = sales_stats['cv'] ** 2
    sales_stats['xyz'] = np.select(
        [sales_stats['cv'] <= 0.5, sales_stats['cv'] <= 1],
        ['X', 'Y'],
        default='Z'
    )

    # SEIL classification
    sales_stats['seil'] = np.select(
        [
            (sales_stats['average_demand_interval'] < 1.32) & (sales_stats['cv_2'] < 0.49),
            (sales_stats['average_demand_interval'] < 1.32) & (sales_stats['cv_2'] >= 0.49),
            (sales_stats['average_demand_interval'] >= 1.32) & (sales_stats['cv_2'] < 0.49)
        ],
        ['smooth', 'erratic', 'intermittent'],
        default='lumpy'
    )

    df_class = sales_stats[['structure_level_1', 'sku', 'abc', 'seil', 'xyz']]

    ## 7.2 Calendar Features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['week_of_month'] = df['date'].apply(lambda d: (d.day - 1) // 7 + 1)

    ## 7.3 Promo Participation and Avg discount (Structure Level 3 x competitor)
    df_avg_discount = df.copy()
    df_avg_discount['discount_qty_pf1'] = (df_avg_discount['discount']*df_avg_discount['quantity']*df_avg_discount['flag_promo'])
    df_avg_discount['qty_pf1'] = (df_avg_discount['quantity']*df_avg_discount['flag_promo'])
    df_avg_discount = df_avg_discount.loc[((df_avg_discount['flag_promo']==1))]

    df_avg_discount = (df_avg_discount.groupby(['structure_level_3', 'competitor'])
                                            .agg({'discount_qty_pf1': "sum",
                                                    'qty_pf1': "sum",
                                                    'discount': "mean"}))
    df_avg_discount['avg_discount_w_L3'] = df_avg_discount['discount_qty_pf1']/df_avg_discount['qty_pf1']
    df_avg_discount = df_avg_discount.rename(columns={'discount':'avg_discount'})[['avg_discount_w_L3']]
    df_avg_discount = df_avg_discount.reset_index()
    df_avg_discount['avg_discount_w_L3'] = df_avg_discount['avg_discount_w_L3'].round(2)

    df_pp = df.copy()
    df_pp['qty_pf1'] = (df_pp['quantity']*df_pp['flag_promo'])
    df_pp = (df_pp.groupby(['structure_level_3', 'competitor'])
                .agg(qty_pf1 = ("qty_pf1","sum"),
                    quantity = ("quantity","sum"),
                    count_pf1 = ("flag_promo","sum"),
                    count_all = ("flag_promo","count")))
    df_pp['promo_part_w_L3'] = df_pp['qty_pf1']/df_pp['quantity']
    df_pp['promo_part'] = df_pp['count_pf1']/df_pp['count_all']
    df_pp = df_pp[['promo_part_w_L3']]
    df_pp = df_pp.reset_index()
    df_pp['promo_part_w_L3'] = df_pp['promo_part_w_L3'].round(2)

    df_new = (df.merge(df_holidays, on=['date'], how='left')
                          .merge(df_weather.drop('time', axis=1), on=['date'], how='left')
                          .merge(df_class, on=['structure_level_1','sku'], how='left')
                          .merge(df_avg_discount, on=['structure_level_3','competitor'], how='left')
                          .merge(df_pp, on=['structure_level_3','competitor'], how='left'))
    ## fill with nan non discount structL3
    df_new[['avg_discount_w_L3','promo_part_w_L3']] = df_new[['avg_discount_w_L3','promo_part_w_L3']].fillna(value=0)

    ## limit discount values (fix errors)
    df_new['discount'] = df_new['discount'].clip(lower=0)

    ## fill nan values
    df_new[['holiday']] = df_new[['holiday']].fillna(value="")
    df_new[['leaflet']] = df_new[['leaflet']].fillna(value="")
    df_new[['chain_campaign']] = df_new[['chain_campaign']].fillna(value="")
    df_new[['holiday_importance']] = df_new[['holiday_importance']].fillna(value=0)

    ## fix data types
    df_new['holiday_importance'] = df_new['holiday_importance'].astype("int")

    ## 8. Convert data to modelling structure
    df_chain = df_new.loc[df_new.competitor=='chain']
    df_chain = df_chain.rename(columns={"discount": "discount_chain","leaflet": "leaflet_chain"})

    df_model = df_new.loc[df_new.competitor!='chain']

    df_model = (df_model.merge(df_chain[['sku','date','discount_chain','leaflet_chain']], on=['sku','date'], how='left'))

    ### TESTE IGNORE COLUMNS discount flag_promo leaflet pvp_was
    df_model = df_model.drop(['discount', 'flag_promo', 'leaflet', 'pvp_was'], axis=1)
    return df_model

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

def get_predictions(model_compA, model_compB, df_test, sku_input, date_input):
    """ Generate predictions for provided models, sku and date """
    pvp_is_competitorA = float(model_compA.predict(df_test.loc[((df_test.sku==sku_input) & (df_test.date==date_input) & (df_test.competitor=='competitorA'))])[0])
    pvp_is_competitorB = float(model_compB.predict(df_test.loc[((df_test.sku==sku_input) & (df_test.date==date_input) & (df_test.competitor=='competitorB'))])[0])

    return pvp_is_competitorA, pvp_is_competitorB


########################################
# Begin web server setup

test_data_path = "test"

app = Flask(__name__)

@app.route("/forecast_prices", methods=["POST"])
def forecast_prices():
    obs_dict = request.get_json()

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

    # Generate Predictions
    date_input = pd.to_datetime(str(time_key_input), format='%Y%m%d').strftime('%Y-%m-%d')
    df_test = prepare_data(test_data_path)    
    model_compA, model_compB = load_model(sku_input, df_test)
    pvp_is_competitorA, pvp_is_competitorB = get_predictions(model_compA, model_compB, df_test, sku_input, date_input)

    # Store forecasted prices in DB
    try:
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

@app.route("/actual_prices", methods=["POST"])
def actual_prices():
    obs_dict = request.get_json()

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

        return Response(
            json.dumps(response, indent=2),
            status=200,
            mimetype='application/json'
        )

    except PricePrediction.DoesNotExist:
        return jsonify({"error": f'Observation ID "{sku}" for time_key {time_key} does not exist'}), 422

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)

