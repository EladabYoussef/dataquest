# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------


# Import necessary libraries here


def preprocess(df):
    import pandas as pd
    import numpy as np

    # Lightweight Data Engineering
    if all(col in df.columns for col in ['Child_Dependents', 'Adult_Dependents', 'Infant_Dependents']):
        df['Total_Dependents'] = df['Child_Dependents'].fillna(0) + df['Adult_Dependents'].fillna(0) + df['Infant_Dependents'].fillna(0)
    
    if 'Employer_ID' in df.columns:
        df['Has_Employer_ID'] = df['Employer_ID'].notnull().astype(int)
        df = df.drop(columns=['Employer_ID'])
        
    # --- ZERO-DEPENDENCY NATIVE TARGET ENCODING ---
    te_maps = {
        "Broker_Agency_Type": {"National_Corporate": 2.8999958, "Urban_Boutique": 2.6422927},
        "Deductible_Tier": {"Tier_1_High_Ded": 2.800746, "Tier_2_Mid_Ded": 2.978621, "Tier_3_Low_Ded": 3.147286, "Tier_4_Zero_Ded": 2.043068},
        "Acquisition_Channel": {"Affiliate_Group": 2.233578, "Aggregator_Site": 2.809370, "Corporate_Partner": 2.199270, "Direct_Website": 3.029014, "Local_Broker": 2.570552},
        "Employment_Status": {"Contractor": 2.796304, "Employed_FullTime": 2.809713, "Self_Employed": 2.307925, "Unemployed": 2.820224}
    }
    
    for col, mapping in te_maps.items():
        if col in df.columns:
            df[f'{col}_TE'] = df[col].map(mapping).fillna(2.75)
    # -------------------------------------------------------------
    # PHASE 10: Advanced Mathematical Features (Pure Pandas)
    # We use Polynomial Interactions between the Top 5 highest-signal numericals
    # to give LightGBM deeper split potential without relying on Scikit-Learn.
    # -------------------------------------------------------------
    df['Inc_x_Days'] = df['Estimated_Annual_Income'].fillna(0) * df['Days_Since_Quote'].fillna(0)
    df['Inc_x_Dur'] = df['Estimated_Annual_Income'].fillna(0) * df['Previous_Policy_Duration_Months'].fillna(0)
    df['Inc_x_Dep'] = df['Estimated_Annual_Income'].fillna(0) * df.get('Total_Dependents', 0)
    df['Inc_x_Week'] = df['Estimated_Annual_Income'].fillna(0) * df['Policy_Start_Week'].fillna(0)
    df['Days_x_Dur'] = df['Days_Since_Quote'].fillna(0) * df['Previous_Policy_Duration_Months'].fillna(0)
    df['Days_x_Dep'] = df['Days_Since_Quote'].fillna(0) * df.get('Total_Dependents', 0)
    df['Days_x_Week'] = df['Days_Since_Quote'].fillna(0) * df['Policy_Start_Week'].fillna(0)
    df['Dur_x_Dep'] = df['Previous_Policy_Duration_Months'].fillna(0) * df.get('Total_Dependents', 0)
    df['Dur_x_Week'] = df['Previous_Policy_Duration_Months'].fillna(0) * df['Policy_Start_Week'].fillna(0)
    df['Dep_x_Week'] = df.get('Total_Dependents', 0) * df['Policy_Start_Week'].fillna(0)

    # Downcast and optimize dtypes for memory constraints
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    obj_cols = df.select_dtypes(include=['object']).columns

    for col in float_cols:
        df[col] = df[col].astype('float32')

    for col in int_cols:
        if col != 'Purchased_Coverage_Bundle':
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Convert object columns to 'category' for LightGBM and Catboost
    for col in obj_cols:
        if col != 'User_ID' and col != 'Purchased_Coverage_Bundle':
            df[col] = df[col].fillna('Missing').astype('category')

    return df


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------
    import joblib
    import os
    
    # Needs to handle relative paths during remote evaluation
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model_dict):
    import pandas as pd
    import numpy as np
    
    # Extract the ensemble dictionary
    lgb_model = model_dict['lgb']
    thresholds = model_dict['thresholds']
    
    # Drop User_ID (and Target if accidentally included)
    drop_cols = ['User_ID']
    if 'Purchased_Coverage_Bundle' in df.columns:
        drop_cols.append('Purchased_Coverage_Bundle')
        
    X_test = df.drop(columns=drop_cols)
    
    # Predict directly with LightGBM
    lgb_proba = lgb_model.predict_proba(X_test)
    
    # Apply Scipy-Optimized Thresholds natively in numpy
    preds = np.argmax(lgb_proba * thresholds, axis=1)
    
    # Format submission DataFrame, explicitly casting to int to satisfy PDF
    predictions = pd.DataFrame({
        'User_ID': df['User_ID'],
        'Purchased_Coverage_Bundle': preds.astype(int)
    })
    
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    import time

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = time.perf_counter()
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model) -> float:
    return 0.0


def get_model_accuracy(predictions) -> float:
    return 0.0
