import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import datetime
from datetime import date
from sklearn.impute import SimpleImputer


@st.cache(allow_output_mutation=True)
def select_col_by_type(df, col_type):
    var_object = []
    for col in df.select_dtypes(col_type):
        var_object.append(col)
    return var_object

@st.cache(allow_output_mutation=True)
def handle_type(df):
    cat_var = select_col_by_type(df, "object")
    int_var = select_col_by_type(df, "int")
    float_var = select_col_by_type(df, "float")
    num_var = int_var + float_var
    return cat_var, num_var

@st.cache(allow_output_mutation=True)
def imputer_nan(df, strategy, var_type):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer = imputer.fit(df[var_type])
    array_type = imputer.transform(df[var_type])
    df_type = pd.DataFrame(array_type, columns=var_type)
    return df_type

@st.cache(allow_output_mutation=True)
def trans_func(df):
    # df = pd.read_csv(data_path +'application_train.csv')
    print('Bonjour 1')
    train_data = df[df.columns[df.isna().sum() / df.shape[0] < 0.5]]
    cat_var, num_var = handle_type(train_data)
    df_cat = imputer_nan(train_data, "most_frequent", cat_var)
    df_num = imputer_nan(train_data, "median", num_var)
    dummies = pd.get_dummies(df_cat, drop_first=True)
    concatenate_df = pd.concat([df_num, dummies], axis=1)
    # data = concatenate_df.sample(n=500, random_state=1)
    #print('Bonjour 2')
    return concatenate_df

@st.cache(allow_output_mutation=True)
def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


columns_list = ['SK_ID_CURR', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
                'TOTALAREA_MODE', 'HOUR_APPR_PROCESS_START', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'OBS_60_CNT_SOCIAL_CIRCLE',
                'FLAG_DOCUMENT_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
                'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_GOODS_PRICE',
                # 'NAME_CONTRACT_TYPE', 'CODE_GENDER'
                ]
infos_descrip = ['SK_ID_CURR',
                 "DAYS_BIRTH",
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_CONTRACT_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",
                 "AMT_INCOME_TOTAL"
                 ]



def main():
    html_temp = """
       <div style="background-color:tomato;padding:10px">
       <h2 style="color:white;text-align:center;"> Dashboard - Décision d'octroi de crédit </h2>
       </div>
       """
    st.set_page_config(

    page_title = "Prêt à dépenser - Plateforme d'octroi de crédit",

    layout = "wide",

    initial_sidebar_state = "expanded",

    )
    st.markdown(html_temp, unsafe_allow_html=True)
    ML_URI = "https://app-projet7.herokuapp.com/predict"
    "\n"
    st.sidebar.image("pretadep.png")

    # loading the useful datasets
    @st.cache(allow_output_mutation=True)
    def load_data(nrows):
        data = pd.read_csv('application_train.csv', nrows=nrows)
        data = data.sample(n=nrows, random_state=1)
        return data

    data = load_data(1000)

    @st.cache(allow_output_mutation=True)
    def load_data1(nrows):
        data = pd.read_csv('application_train.csv', nrows=nrows)
        samp = data.sample(n=nrows, random_state=1)
        data2 = samp[infos_descrip].set_index('SK_ID_CURR')
        data1 = samp[columns_list].set_index('SK_ID_CURR')
        return data1, data2

    data1, data2 = load_data1(1000)
    # Selecting one client
    id_client = st.sidebar.selectbox('Select ID Client :', data1.index)
    if id_client:
        # Visualizing the personal data of the selected client
        st.sidebar.subheader('Les informations du client %s : ' % id_client)
        st.sidebar.table(data2.astype(str).loc[id_client][1:9])
        cat_var, num_var = handle_type(data1)
        df_num = imputer_nan(data1, "median", num_var)
        id_cli = data['SK_ID_CURR']
        data_client = pd.concat([id_cli, df_num], axis=1)
        data_client1 = data_client[data_client['SK_ID_CURR'] == id_client]
        data_client2 = data_client1.drop("SK_ID_CURR", axis=1)
        data_master = data_client2.to_dict('records')[0]
        data_client5 = json.dumps(data_master)
        pred = None
        headers = {
            'Content-Type': 'application/json'
        }
        import ast
        response = requests.request("POST", ML_URI, headers=headers, data=data_client5)
        tab = response.text
        tab1 = ast.literal_eval(tab)
        tab2 = tab1["prediction"]
        for cle, valeur in tab2.items():
            solvabilit = cle
            defaut = valeur
        df_score = pd.DataFrame.from_dict(tab1).reset_index()
        df_score.columns = ["Solva", "Defaut"]
        df_score = df_score.apply(pd.to_numeric)
        # Visualizing the score and decision about loan granting
        '# Client ID number %s' % id_client
        solva = round(df_score["Solva"][0], 2)
        defaut_pay = round(df_score["Defaut"][0], 2)

        if st.checkbox('La décision finale'):
            if solva > 0.7:
                st.success("Client éligible au prêt")
            else:
                st.error("Ce client est à risque")

        "\n"

        if st.checkbox('Voulez-vous consulter les probabilités de remboursement et non ?'):
            col1, col2, col3 = st.columns(3)
            with col1:
                "\n"
                "\n"
                '### Proba de remboursement : %s' % "{:.0%}".format(solva)
            with col2:
                "\n"
                "\n"
                '### Proba de Défaut : %s' % "{:.0%}".format(defaut_pay)

            with col3:
                st.header("Visualisation")
                st.bar_chart(df_score, width=200, height=200, use_container_width=True)
        "\n"
        # Visualizing the most important features ordered by importance
        if st.checkbox('Souhaitez-vous voir les informations qui ont influé sur cette décision ?'):
                '### Les informations les plus importantes dans la décision :'
                st.image("features_importance_1.png")

        if st.checkbox('Les autres clients similaires  ?'):
            '### Les autres clients similaires %s' % id_client
            feature_name = st.selectbox('Selecting feature name :', [
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_CONTRACT_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",

                                                                         ]
                                        )
            data_feature = data[feature_name].value_counts(normalize=True)
            st.bar_chart(data_feature)

if __name__ == '__main__':
    main()
