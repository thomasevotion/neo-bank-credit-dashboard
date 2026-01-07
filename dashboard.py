import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# Configuration de la page
st.set_page_config(
    page_title="Outil d'Aide à la Décision Crédit",
    layout="wide"
)

# Configuration technique
API_URL = os.getenv('API_URL', 'https://neo-bank-credit-api.onrender.com/predict')

POSSIBLE_PATHS = [
    "application_train_sample.csv",
    "./application_train.csv",
    "./data/application_train.csv",
    "../home-credit-default-risk/application_train.csv",
    "C:/Users/thoma/Downloads/home-credit-default-risk/application_train.csv" # Fallback dev
]

DATA_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        DATA_PATH = path
        break

# Dictionnaire de traduction
TRADUCTION_VARIABLES = {
    "AMT_INCOME_TOTAL": "Revenus Annuels",
    "AMT_CREDIT": "Montant du Crédit",
    "AMT_ANNUITY": "Montant de l'Annuité",
    "CNT_CHILDREN": "Nombre d'Enfants",
    "DAYS_BIRTH": "Âge du Client",
    "DAYS_EMPLOYED": "Ancienneté Emploi"
}

# --- Fonctions Utilitaires ---

@st.cache_data
def load_data():
    """
    Charge un échantillon de données clients pour la démonstration.
    Tente de charger depuis DATA_PATH.
    
    Returns:
        pd.DataFrame ou None: Le dataframe chargé ou None si échec.
    """
    if DATA_PATH is None:
        return None
        
    cols = [
        'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
        'AMT_ANNUITY', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    try:
        # On charge juste 1000 lignes pour la démo
        df = pd.read_csv(DATA_PATH, usecols=cols, nrows=1000)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

def get_prediction(client_data):
    """
    Envoie les données client à l'API de scoring.
    
    Args:
        client_data (pd.Series): Les données du client sélectionné.
        
    Returns:
        dict: La réponse JSON de l'API (prediction, probability, shap_values...) ou None.
    """
    data = {
        "AMT_INCOME_TOTAL": float(client_data['AMT_INCOME_TOTAL']),
        "AMT_CREDIT": float(client_data['AMT_CREDIT']),
        "AMT_ANNUITY": float(client_data['AMT_ANNUITY']) if not pd.isna(client_data['AMT_ANNUITY']) else None,
        "CNT_CHILDREN": int(client_data['CNT_CHILDREN']),
        "DAYS_BIRTH": int(client_data['DAYS_BIRTH']),
        "DAYS_EMPLOYED": float(client_data['DAYS_EMPLOYED'])
    }
    try:
        response = requests.post(API_URL, json=data, timeout=10) # Timeout pour éviter le blocage infini
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur serveur ({response.status_code}) : {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de contacter l'API à l'adresse {API_URL}. Vérifiez qu'elle est lancée.")
        return None
    except requests.exceptions.Timeout:
        st.error("Le serveur met trop de temps à répondre.")
        return None

# --- Interface Utilisateur ---

st.title("Dashboard Conseiller - Octroi de Crédit")
st.markdown("---")

df = load_data()

if df is None:
    st.warning("Fichier de données 'application_train.csv' introuvable.")
    st.info("Veuillez placer le fichier CSV dans le même dossier que ce script ou dans un dossier 'data/'.")
else:
    # Calcul des moyennes globales pour comparaison
    avg_income = df['AMT_INCOME_TOTAL'].median()
    avg_credit = df['AMT_CREDIT'].median()
    avg_annuity = df['AMT_ANNUITY'].median()
    
    # Sidebar : Sélection du dossier
    st.sidebar.header("Dossier Client")
    client_id = st.sidebar.selectbox("Numéro de dossier", df['SK_ID_CURR'].unique())
    st.sidebar.markdown("---")
    st.sidebar.info("ℹ️ **RGPD** : Les données affichées sont pseudonymisées. Durée de conservation : 3 ans.")

    # Données client
    client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]
    
    # 1. Informations Financières et Personnelles
    st.subheader("Synthèse du Profil")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_income = client_data['AMT_INCOME_TOTAL'] - avg_income
        st.metric(
            label="Revenus Annuels", 
            value=f"{client_data['AMT_INCOME_TOTAL']:,.0f} €",
            delta=f"{delta_income:,.0f} € vs Médiane",
            delta_color="normal"
        )
    
    with col2:
        delta_credit = client_data['AMT_CREDIT'] - avg_credit
        st.metric(
            label="Montant Crédit", 
            value=f"{client_data['AMT_CREDIT']:,.0f} €",
            delta=f"{delta_credit:,.0f} € vs Médiane",
            delta_color="inverse"
        )
        
    with col3:
        delta_annuity = client_data['AMT_ANNUITY'] - avg_annuity
        st.metric(
            label="Annuités", 
            value=f"{client_data['AMT_ANNUITY']:,.0f} €",
            delta=f"{delta_annuity:,.0f} € vs Médiane",
            delta_color="inverse"
        )
        
    with col4:
        try:
            age = int(abs(client_data['DAYS_BIRTH']) / 365)
        except (ValueError, TypeError):
            age = "N/A"
        st.metric("Âge Client", f"{age} ans" if isinstance(age, int) else age)
    
    # Formatage propre
    try:
        nb_enfants = int(client_data['CNT_CHILDREN'])
    except:
        nb_enfants = "?"
        
    try:
        raw_employed = client_data['DAYS_EMPLOYED']
        if raw_employed == 365243 or pd.isna(raw_employed):
             anciennete_str = "Sans emploi / Retraité"
        else:
            annees = int(abs(raw_employed) / 365)
            anciennete_str = f"{annees} ans d'ancienneté"
    except:
        anciennete_str = "N/A"
    
    st.caption(f"Situation professionnelle : {anciennete_str} | Enfants à charge : {nb_enfants}")

    st.markdown("### Analyse du Risque")

    # 2. Bouton d'action
    if st.button("Lancer l'évaluation"):
        with st.spinner('Calcul du score en cours...'):
            result = get_prediction(client_data)
        
        if result:
            prediction = result['prediction']
            probability = result['probability']
            threshold = result.get('threshold', 0.5) * 100
            shap_values = result['shap_values']
            
            col_score, col_details = st.columns([1, 2])
            
            with col_score:
                st.markdown("#### Score de Solvabilité")
                
                # Jauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    number = {'suffix': "%", 'font': {'size': 24}},
                    title = {'text': "Probabilité de Défaut", 'font': {'size': 14}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                        'bar': {'color': "#d62728" if prediction == 1 else "#2ca02c"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, threshold], 'color': "#e5f5e0"},
                            {'range': [threshold, 100], 'color': "#fee5d9"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': threshold
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if prediction == 1:
                    st.error("Avis : DÉFAVORABLE (Risque élevé)")
                else:
                    st.success("Avis : FAVORABLE (Risque faible)")
                
                st.info("⚠️ Décision finale soumise à validation humaine.")

            with col_details:
                st.markdown("#### Facteurs explicatifs (Interprétabilité)")
                st.write("Les éléments ci-dessous ont le plus impacté la décision de l'algorithme :")
                
                shap_data = []
                for var_name, impact in shap_values.items():
                    nom_francais = TRADUCTION_VARIABLES.get(var_name, var_name)
                    shap_data.append({'Variable': nom_francais, 'Impact': impact})
                
                shap_df = pd.DataFrame(shap_data)
                shap_df['Type'] = np.where(shap_df['Impact'] > 0, 'Augmente le risque', 'Diminue le risque')
                shap_df = shap_df.sort_values(by='Impact', key=abs, ascending=True).tail(5)
                
                fig_shap = px.bar(
                    shap_df, 
                    x='Impact', 
                    y='Variable', 
                    orientation='h',
                    color='Type',
                    color_discrete_map={'Augmente le risque': '#d62728', 'Diminue le risque': '#2ca02c'},
                    text_auto='.2f'
                )
                fig_shap.update_layout(
                    xaxis_title="Impact sur le score",
                    yaxis_title="",
                    legend_title="",
                    plot_bgcolor='white',
                    margin=dict(l=0,r=0,t=0,b=0),
                    height=300
                )
                fig_shap.update_xaxes(showgrid=True, gridcolor='lightgray')
                st.plotly_chart(fig_shap, use_container_width=True)
