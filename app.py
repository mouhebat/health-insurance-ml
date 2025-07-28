from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import sklearn
import scipy.stats as stats
import shap
from src.demisco.predictionDeductionMultiClass import loadObject , getCategories , getYeojohnsonTransform ,getDummies , getAggrementDeductionPercentage ,getColumnParameters

app = Flask(__name__)


@app.route('/dpp/predictBinary', methods=['POST'] , endpoint='dpp/predictBinary')
def predictDeductionBinary():

    # load object
    modelPath = os.path.join('models/deductionBinary', 'modelXgboost.pkl')
    scalerPath = os.path.join('models/deductionBinary', 'standardScalerList.pkl')
    labelPath = os.path.join('models/deductionBinary', 'labelEncoderList.pkl')
    columnNamepath = os.path.join('models/deductionBinary', 'columnNameList.pkl')
    columnNameLogPath = os.path.join('models/deductionBinary', 'columnNameLogList.pkl')
    columnNameYeojohnsonPath = os.path.join('models/deductionBinary', 'columnNameYeojohnsonList.pkl')
    dictOutlierPath = os.path.join('models/deductionBinary', 'dictOutlier.pkl')
    lambdaPath = os.path.join('models/deductionBinary', 'lambdaYeojohnsonList.pkl')

    model = joblib.load(modelPath)
    scaler = joblib.load(scalerPath)
    labelEncoder = joblib.load(labelPath)
    columnName = joblib.load(columnNamepath)
    columnNameLog = joblib.load(columnNameLogPath)
    columnNameYeojohnson = joblib.load(columnNameYeojohnsonPath)
    dictOutlier = joblib.load(dictOutlierPath)
    lambdaYeojohnson = joblib.load(lambdaPath)

    data = request.get_json(force=True)
    df = pd.DataFrame(data=data,index=[1])

    bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, np.inf]
    labels = ['0-5', '6-15', '16-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
    df['age_category'] = pd.cut(df['INSURED_AGE'], bins=bins, labels=labels, right=True)
    dict = {'0-5': 0, '6-15': 1, '16-25': 2, '26-35': 3, '26-35': 4, '36-45': 5, '46-55': 6, '56-65': 7, '66-75': 8,
            '76+': 9}
    df['age_category_encode'] = df['age_category'].map(dict)

    dictAcc = {1:0 , 2: 0 , 3: 1 , 4: 1 , 5: 1, 6: 1}
    df['ACCREDITATION_LEVEL_ID_category'] = df['ACCREDITATION_LEVEL_ID'].map(dictAcc)

    df.drop(columns=['ACCREDITATION_LEVEL_ID', 'age_category','INSURED_AGE'], inplace=True)

    df.loc[df['GENDER_TYPE'].isin(['UNDEFINED', 'UNKNOWN', 'OTHER']), 'GENDER_TYPE'] = 'FEMALE'


    # log column
    for col in columnNameLog:
        adjusted_column = df[col] + 1
        log_transformed_column = np.log(adjusted_column)
        df[col+'_log'] = log_transformed_column

    df.drop(columns=columnNameLog, inplace=True)


    # get lower and upper for get outlier
    outlierFlg = False
    outlierColStr = ""
    for glob, valout in dictOutlier.items():
        if glob == df['GLOBAL_NATIONAL_CODE'].values:
            for col, v in valout.items():
                if (df[col].values < v["lower_bound"] or df[col].values > v["upper_bound"]) :
                    outlierFlg = True
                    outlierColStr += col.replace("_yeo", "").replace("_log", "") + ","

    if outlierFlg:
        return jsonify({'prediction': 1, 'description': 'outlier ' + ',' + outlierColStr})

    # yeojohnson
    for column in columnNameYeojohnson:
        lambda_ = lambdaYeojohnson[column]
        df[f'{column}_yeo'] = stats.yeojohnson(df[column].values, lmbda=lambda_)

    df.drop(columns=columnNameYeojohnson, inplace=True)

    df_ref = pd.DataFrame({
        'IS_FULL_TIME_PROFESSIONAL': ['0', '1'],
        'IS_FULL_TIME_ANESTHESIA': ['0', '1'],
        'GENDER_TYPE': ['male', 'FEMALE'],
        'ACCREDITATION_LEVEL_ID_category': ['0', '1']
    })


    df_ref_dummies = pd.get_dummies(df_ref)
    dummy_columns = df_ref_dummies.columns.tolist()


    df_dummies = pd.get_dummies(df, columns=['IS_FULL_TIME_PROFESSIONAL', 'IS_FULL_TIME_ANESTHESIA', 'GENDER_TYPE','ACCREDITATION_LEVEL_ID_category'])

    df_dummies = df_dummies.reindex(columns=dummy_columns, fill_value=0)
    df_combined = pd.concat(
        [df.drop(columns=['IS_FULL_TIME_PROFESSIONAL', 'IS_FULL_TIME_ANESTHESIA', 'GENDER_TYPE','ACCREDITATION_LEVEL_ID_category']), df_dummies], axis=1)

    df_combined =  df_combined.reindex(columns=columnName, fill_value=0)


    # standardScalar
    try:
        for col,v in scaler.items():
              df_combined[col] = scaler[col].transform(df_combined[[col]])
    except Exception as e:
        return jsonify({'prediction': 1, 'description': 'error for scaler  ' + ',' + col.replace("_yeo", "").replace("_log", "") + ','})

    # label_encoder
    try:
        for col,v in labelEncoder.items():
            df_combined[col] = labelEncoder[col].transform(df_combined[[col]])
    except Exception as e:
        return  jsonify({'prediction': 1 ,'description': 'error for labelEncoder  '+',' + col.replace("_yeo", "").replace("_log", "") + ','})

    df_combined['IS_FULL_TIME_PROFESSIONAL_0'] = df_combined['IS_FULL_TIME_PROFESSIONAL_0'].astype(int)
    df_combined['IS_FULL_TIME_PROFESSIONAL_1'] = df_combined['IS_FULL_TIME_PROFESSIONAL_1'].astype(int)
    df_combined['IS_FULL_TIME_ANESTHESIA_0'] = df_combined['IS_FULL_TIME_ANESTHESIA_0'].astype(int)
    df_combined['IS_FULL_TIME_ANESTHESIA_1'] = df_combined['IS_FULL_TIME_ANESTHESIA_1'].astype(int)
    df_combined['GENDER_TYPE_FEMALE'] = df_combined['GENDER_TYPE_FEMALE'].astype(int)
    df_combined['GENDER_TYPE_male'] = df_combined['GENDER_TYPE_male'].astype(int)
    df_combined['ACCREDITATION_LEVEL_ID_category_0'] = df_combined['ACCREDITATION_LEVEL_ID_category_0'].astype(int)
    df_combined['ACCREDITATION_LEVEL_ID_category_1'] = df_combined['ACCREDITATION_LEVEL_ID_category_1'].astype(int)


    prediction = model.predict(df_combined)
    predictProb = model.predict_proba(df_combined)

    explainer = shap.TreeExplainer(model)

    def calculate_relative_importance(shap_values):
        total_shap = np.sum(np.abs(shap_values))
        relative_importance = (shap_values / total_shap) * 100
        return relative_importance

    shap_valuess = explainer.shap_values(df_combined)[0]
    relative_importance = calculate_relative_importance(shap_valuess)

    featureImpactShap = {}
    for feature, importance in zip(df_combined.columns, relative_importance):
        # print(f'{feature}: {importance:.2f}%')
        featureImpactShap[feature] = float(importance)


    return jsonify({'prediction': int(prediction[0]),'description': 'ok','featureImpactShap':featureImpactShap , 'predictProb': predictProb.tolist()[0] })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='105')



