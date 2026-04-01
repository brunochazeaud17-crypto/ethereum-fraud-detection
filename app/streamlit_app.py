import pandas as pd
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Contournement pour le bug d'affichage Plotly sur Streamlit Cloud
pio.renderers.default = 'browser'

# Configuration de la page
st.set_page_config(page_title="Ethereum Fraud Detection", page_icon="🔍", layout="wide")

# Titre principal
st.title("🔍 Détecteur d'Anomalies Blockchain Ethereum")
st.markdown("Analyse non supervisée de portefeuilles suspects *(Isolation Forest + Autoencoder)*")

# --------------------------------------------------------
# CLÉ API ETHERSCAN
# --------------------------------------------------------
ETHERSCAN_API_KEY = "5QVQFCVNQD8XW8ZKS2RTT7MC6P7222P6QV"

# --------------------------------------------------------
# DISCLAIMER API
# --------------------------------------------------------
st.sidebar.markdown("""
---
⚠️ **Note sur l'API Etherscan**
L'API gratuite a des limitations. Pour certaines adresses, nous utilisons une base de test.
Vous pouvez aussi utiliser le **mode manuel** ci-dessous.
""")

# --------------------------------------------------------
# FONCTION POUR CONVERTIR DES TRANSACTIONS EN FEATURES
# --------------------------------------------------------
def transactions_to_features(transactions_df):
    """
    Convertit un DataFrame de transactions en features de portefeuille
    """
    # S'assurer que le TimeStamp est au format datetime
    transactions_df = transactions_df.copy()
    transactions_df['TimeStamp_dt'] = pd.to_datetime(transactions_df['TimeStamp'], unit='s')
    
    # IMPORTANT: On groupe par 'From' (l'adresse du wallet qu'on analyse)
    portefeuilles = transactions_df.groupby('From').agg(
        nb_transactions=('TxHash', 'count'),
        total_eth_envoye=('Value', 'sum'),
        moyenne_eth_envoye=('Value', 'mean'),
        max_eth_envoye=('Value', 'max'),
        nb_erreurs=('isError', 'sum'),
        premier_tx=('TimeStamp_dt', 'min'),
        dernier_tx=('TimeStamp_dt', 'max')
    ).reset_index()
    
    # Création des variables comportementales
    portefeuilles['duree_vie_heures'] = (portefeuilles['dernier_tx'] - portefeuilles['premier_tx']).dt.total_seconds() / 3600
    portefeuilles['duree_vie_heures'] = portefeuilles['duree_vie_heures'].replace(0, 0.1)
    
    portefeuilles['tx_par_heure'] = portefeuilles['nb_transactions'] / portefeuilles['duree_vie_heures']
    portefeuilles['taux_erreur'] = (portefeuilles['nb_erreurs'] / portefeuilles['nb_transactions']) * 100
    
    # Nettoyer
    portefeuilles_final = portefeuilles.drop(columns=['premier_tx', 'dernier_tx', 'nb_erreurs', 'From'])
    
    return portefeuilles_final

# --------------------------------------------------------
# FONCTION POUR RÉCUPÉRER LES DONNÉES DEPUIS ETHERSCAN
# --------------------------------------------------------
def get_wallet_data_from_etherscan(address):
    """
    Récupère les données d'un wallet Ethereum depuis Etherscan
    """
    url = "https://api.etherscan.io/api"
    
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data['status'] != '1':
            return None, "Adresse invalide ou aucune transaction trouvée"
        
        transactions = data['result']
        
        if len(transactions) == 0:
            return None, "Aucune transaction trouvée pour cette adresse"
        
        # Convertir en DataFrame
        df_tx = pd.DataFrame(transactions)
        df_tx['Value'] = df_tx['value'].astype(float) / 10**18
        df_tx['isError'] = df_tx['isError'].astype(int)
        df_tx['TimeStamp'] = df_tx['timeStamp'].astype(int)
        
        # Appliquer la conversion en features
        features_df = transactions_to_features(df_tx)
        
        if len(features_df) == 0:
            return None, "Aucune donnée agrégée"
        
        features = features_df.iloc[0].to_dict()
        
        return features, None
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# --------------------------------------------------------
# BASE DE DONNÉES DE TEST
# --------------------------------------------------------
def get_test_wallet_data(address):
    """
    Retourne des données de test pour des adresses connues
    """
    test_data = {
        "0x7f37f78cbd3d29260be6fed108c4b3c3bf462c46": {
            'nb_transactions': 1250,
            'total_eth_envoye': 3420.5,
            'moyenne_eth_envoye': 2.74,
            'max_eth_envoye': 150.0,
            'duree_vie_heures': 48,
            'tx_par_heure': 26.04,
            'taux_erreur': 45.2
        },
        "0x28c6c06298d514db089934071355e5743bf21d60": {
            'nb_transactions': 45200,
            'total_eth_envoye': 1250000.0,
            'moyenne_eth_envoye': 27.65,
            'max_eth_envoye': 5000.0,
            'duree_vie_heures': 87600,
            'tx_par_heure': 0.52,
            'taux_erreur': 0.8
        },
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {
            'nb_transactions': 185000,
            'total_eth_envoye': 2500000.0,
            'moyenne_eth_envoye': 13.51,
            'max_eth_envoye': 8000.0,
            'duree_vie_heures': 65700,
            'tx_par_heure': 2.82,
            'taux_erreur': 1.2
        },
        "0x090d3f5dde9e48cf859d028bf2f9ccb3cbf592b7": {
            'nb_transactions': 850,
            'total_eth_envoye': 12500.0,
            'moyenne_eth_envoye': 14.71,
            'max_eth_envoye': 500.0,
            'duree_vie_heures': 12,
            'tx_par_heure': 70.83,
            'taux_erreur': 38.5
        },
        "0x742d35cc6634c0532925a3b844bc9e7598f0b5b5": {
            'nb_transactions': 45,
            'total_eth_envoye': 1250.0,
            'moyenne_eth_envoye': 27.78,
            'max_eth_envoye': 500.0,
            'duree_vie_heures': 35040,
            'tx_par_heure': 0.0013,
            'taux_erreur': 2.2
        }
    }
    
    address_lower = address.lower()
    
    if address_lower in test_data:
        return test_data[address_lower], None
    else:
        return None, "Adresse non trouvée dans la base de test"

# --------------------------------------------------------
# 1. CHARGEMENT DES MODÈLES
# --------------------------------------------------------
@st.cache_resource
def load_models():
    iso_forest = joblib.load('models/iso_forest_model.pkl')
    autoencoder = tf.keras.models.load_model('models/autoencoder_model.h5')
    scaler = joblib.load('models/scaler_features.pkl')
    features_names = joblib.load('models/features_names.pkl')
    threshold = joblib.load('models/autoencoder_threshold.pkl')
    return iso_forest, autoencoder, scaler, features_names, threshold

try:
    iso_forest, autoencoder, scaler, features_names, threshold = load_models()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles : {e}")
    st.stop()

# --------------------------------------------------------
# 2. INTERFACE UTILISATEUR SIMPLIFIÉE
# --------------------------------------------------------
st.sidebar.header("🔍 Analyse d'un portefeuille")

# Option 1: API Etherscan
st.sidebar.subheader("📡 Option 1 : Adresse Ethereum")
wallet_address = st.sidebar.text_input(
    "Adresse du portefeuille", 
    placeholder="0x742d35Cc6634C0532925a3b844Bc9e7598f0b5b5",
    help="Collez une adresse Ethereum valide (commence par 0x)"
)

# Option 2: Mode manuel simplifié
st.sidebar.subheader("📝 Option 2 : Mode manuel")
with st.sidebar.expander("Saisie directe des caractéristiques"):
    st.markdown("Entrez directement les caractéristiques du portefeuille à analyser :")
    
    nb_tx = st.number_input("📊 Nombre total de transactions", min_value=1, max_value=1000000, value=150)
    total_eth = st.number_input("💰 Total ETH envoyé", min_value=0.0, max_value=10000000.0, value=15.5)
    moyenne_eth = st.number_input("📈 Moyenne ETH par transaction", min_value=0.0, max_value=10000.0, value=0.1)
    max_eth = st.number_input("📊 Max ETH sur une transaction", min_value=0.0, max_value=1000000.0, value=5.0)
    duree_vie = st.number_input("⏱️ Durée de vie du portefeuille (heures)", min_value=0, max_value=1000000, value=2)
    frequence = st.number_input("⚡ Fréquence (transactions/heure)", min_value=0.0, max_value=1000.0, value=75.0)
    taux_erreur = st.number_input("❌ Taux d'erreur (%)", min_value=0.0, max_value=100.0, value=45.0)
    
    use_manual = st.checkbox("✅ Utiliser le mode manuel")

# Bouton d'analyse
analyze_button = st.sidebar.button("🔍 Analyser", type="primary")

# --------------------------------------------------------
# 3. LOGIQUE DE PRÉDICTION
# --------------------------------------------------------
if analyze_button:
    # Déterminer la source de données
    if use_manual:
        # Mode manuel : utiliser les valeurs saisies directement
        input_dict = {
            'nb_transactions': [nb_tx],
            'total_eth_envoye': [total_eth],
            'moyenne_eth_envoye': [moyenne_eth],
            'max_eth_envoye': [max_eth],
            'duree_vie_heures': [duree_vie],
            'tx_par_heure': [frequence],
            'taux_erreur': [taux_erreur]
        }
        st.info("📊 Mode manuel - Données saisies directement")
        
    elif wallet_address:
        # Mode API
        with st.spinner("🔄 Récupération des données..."):
            features, error = get_wallet_data_from_etherscan(wallet_address)
        
        if error:
            st.warning(f"⚠️ API: {error}")
            with st.spinner("📁 Recherche dans la base de test..."):
                features, error_test = get_test_wallet_data(wallet_address)
            
            if error_test:
                st.error(f"❌ {error_test}")
                st.info("💡 Conseil: Utilisez le mode manuel pour tester des valeurs personnalisées")
                st.stop()
            else:
                st.info("📊 Mode base de test - Données simulées")
        else:
            st.success(f"✅ Données récupérées - {features['nb_transactions']} transactions")
        
        input_dict = {k: [v] for k, v in features.items()}
        
        # Afficher un résumé
        with st.expander("📊 Détails des données récupérées"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre de transactions", f"{features['nb_transactions']:,}")
                st.metric("Total ETH envoyé", f"{features['total_eth_envoye']:.2f} ETH")
                st.metric("Moyenne ETH/tx", f"{features['moyenne_eth_envoye']:.4f} ETH")
                st.metric("Max ETH/tx", f"{features['max_eth_envoye']:.2f} ETH")
            with col2:
                st.metric("Durée de vie", f"{features['duree_vie_heures']:.1f} heures")
                st.metric("Fréquence", f"{features['tx_par_heure']:.1f} tx/h")
                st.metric("Taux d'erreur", f"{features['taux_erreur']:.1f}%")
    else:
        st.warning("⚠️ Veuillez saisir une adresse ou activer le mode manuel")
        st.stop()
    
    # Création du DataFrame d'entrée
    df_input = pd.DataFrame(input_dict)
    
    # Vérifier que les colonnes sont dans le bon ordre
    expected_columns = ['nb_transactions', 'total_eth_envoye', 'moyenne_eth_envoye', 
                        'max_eth_envoye', 'duree_vie_heures', 'tx_par_heure', 'taux_erreur']
    df_input = df_input[expected_columns]
    
    # Scaling
    df_scaled = scaler.transform(df_input)
    
    # --- Modèle 1 : Isolation Forest ---
    prediction_iso = iso_forest.predict(df_scaled)[0]
    score_iso = iso_forest.decision_function(df_scaled)[0]
    is_anomaly_iso = prediction_iso == -1
    
    # --- Modèle 2 : Autoencoder ---
    reconstruction = autoencoder.predict(df_scaled, verbose=0)
    mse = np.mean(np.power(df_scaled - reconstruction, 2), axis=1)[0]
    is_anomaly_ae = mse > threshold
    
    # --- Scores de risque ---
    if is_anomaly_iso:
        iso_risk = min(1.0, max(0.0, (-score_iso) / 0.5))
    else:
        iso_risk = max(0.0, min(0.3, score_iso / 2))
    
    auto_risk = min(1.0, mse / threshold) if is_anomaly_ae else mse / (threshold * 2)
    final_risk_score = (iso_risk + auto_risk) / 2
    
    # --- Consensus ---
    if is_anomaly_iso and is_anomaly_ae:
        niveau_risque = "🚨 RISQUE ÉLEVÉ"
        recommendation = "Ce portefeuille présente un comportement fortement anormal. Investigation immédiate requise."
    elif is_anomaly_iso or is_anomaly_ae:
        niveau_risque = "⚠️ RISQUE MODÉRÉ"
        if is_anomaly_iso:
            recommendation = "Comportement extrême détecté. Surveiller ce portefeuille."
        else:
            recommendation = "Pattern subtil anormal. Peut indiquer une fraude sophistiquée."
    else:
        niveau_risque = "🟢 NORMAL"
        recommendation = "Comportement typique d'un portefeuille Ethereum légitime."
    
    # --------------------------------------------------------
    # 4. AFFICHAGE DES RÉSULTATS
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Résultats de l'analyse")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Niveau de Risque", niveau_risque)
    with col2:
        st.metric("Score de risque", f"{final_risk_score:.1%}")
    with col3:
        st.metric("Consensus", "✅ Normal" if not (is_anomaly_iso or is_anomaly_ae) else "⚠️ Anomalie")
    
    st.markdown(f"### 💡 Recommandation")
    st.markdown(f"**{recommendation}**")
    
    # Barre de progression
    if final_risk_score > 0.7:
        st.progress(final_risk_score, text=f"🔴 {final_risk_score:.0%} - Risque critique")
    elif final_risk_score > 0.4:
        st.progress(final_risk_score, text=f"🟡 {final_risk_score:.0%} - Risque modéré")
    else:
        st.progress(final_risk_score, text=f"🟢 {final_risk_score:.0%} - Risque faible")
    
    # --------------------------------------------------------
    # 5. GRAPHIQUES ILLUSTRATIFS
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Visualisation des décisions des modèles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1: Comparaison des scores
        fig_scores = go.Figure(data=[
            go.Bar(name='Isolation Forest', x=['Score'], y=[iso_risk], 
                   marker_color='red' if is_anomaly_iso else 'green'),
            go.Bar(name='Autoencoder', x=['Score'], y=[auto_risk],
                   marker_color='red' if is_anomaly_ae else 'green'),
            go.Bar(name='Score final', x=['Score'], y=[final_risk_score],
                   marker_color='orange')
        ])
        fig_scores.update_layout(
            title="Scores de risque par modèle",
            yaxis_title="Niveau de risque",
            yaxis_range=[0, 1],
            showlegend=True
        )
        st.plotly_chart(fig_scores, use_container_width=False)
        
        # Graphique 2: Erreur de reconstruction
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(
            x=['Seuil', 'MSE actuel'],
            y=[threshold, mse],
            mode='markers',
            marker=dict(size=[30, 30], color=['blue', 'red']),
            text=[f'Seuil: {threshold:.4f}', f'MSE: {mse:.4f}'],
            textposition='top center'
        ))
        fig_mse.add_hline(y=threshold, line_dash="dash", line_color="blue")
        fig_mse.update_layout(
            title="Erreur de reconstruction - Autoencoder",
            yaxis_title="Mean Squared Error (MSE)",
            showlegend=False
        )
        st.plotly_chart(fig_mse, use_container_width=False)
    
    with col2:
        # Graphique 3: Radar chart des features normalisées
        features_normalized = df_scaled[0]
        feature_names = ['nb_tx', 'total_eth', 'moyenne', 'max', 'duree_vie', 'frequence', 'taux_erreur']
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=features_normalized,
            theta=feature_names,
            fill='toself',
            marker=dict(color='blue')
        ))
        fig_radar.update_layout(
            title="Profil normalisé du portefeuille",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )
            ),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=False)
        
        # Graphique 4: Décision binaire des modèles
        fig_decision = go.Figure(data=[
            go.Bar(name='Isolation Forest', x=['Détection'], 
                   y=[1 if is_anomaly_iso else 0],
                   marker_color='red' if is_anomaly_iso else 'green'),
            go.Bar(name='Autoencoder', x=['Détection'],
                   y=[1 if is_anomaly_ae else 0],
                   marker_color='red' if is_anomaly_ae else 'green')
        ])
        fig_decision.update_layout(
            title="Détection d'anomalie par modèle",
            yaxis_title="Anomalie détectée (1=Oui, 0=Non)",
            yaxis_range=[0, 1.2],
            showlegend=True
        )
        st.plotly_chart(fig_decision, use_container_width=False)
    
    # Tableau des caractéristiques
    st.markdown("---")
    st.subheader("📊 Caractéristiques analysées")
    
    features_df_vis = pd.DataFrame({
        'Caractéristique': ['Nombre de transactions', 'Total ETH envoyé', 'Moyenne ETH/tx', 
                           'Max ETH/tx', 'Durée de vie (heures)', 'Fréquence (tx/heure)', 'Taux d\'erreur (%)'],
        'Valeur': [input_dict['nb_transactions'][0], input_dict['total_eth_envoye'][0], 
                   input_dict['moyenne_eth_envoye'][0], input_dict['max_eth_envoye'][0],
                   input_dict['duree_vie_heures'][0], input_dict['tx_par_heure'][0], 
                   input_dict['taux_erreur'][0]],
        'Unité': ['', 'ETH', 'ETH', 'ETH', 'heures', 'tx/h', '%']
    })
    
    st.dataframe(features_df_vis, use_container_width=True, hide_index=True)
    
    # Alerte si risque élevé
    if final_risk_score > 0.7:
        st.error("🚨 **ALERTE DE SÉCURITÉ** : Comportement hautement suspect détecté !")
        st.balloons()
    elif final_risk_score > 0.4:
        st.warning("⚠️ **ATTENTION** : Comportement inhabituel détecté.")

else:
    # Message d'accueil
    st.info("👈 **Choisissez une option** dans la barre latérale pour analyser un portefeuille")
    
    with st.expander("ℹ️ À propos de cette application", expanded=True):
        st.markdown("""
        ### 🏗️ Architecture technique : Double modèle en consensus
        
        **1. Isolation Forest** - Détecte les comportements extrêmes et anomalies statistiques
        **2. Autoencoder** - Détecte les patterns subtils par erreur de reconstruction
        **3. Consensus** - Combine les deux pour une détection robuste
        
        ### 🎯 Approche Zero-shot
        
        **Comme dans 90% des cas réels, cette solution fonctionne sans données labellisées.**
        
        ### 📊 Comment utiliser
        
        **Option 1 - Adresse Ethereum** : Entrez une adresse, l'API récupère automatiquement les données
        **Option 2 - Mode manuel** : Saisissez directement les 7 caractéristiques du portefeuille
        
        ### 📝 Adresses de test
        
        - **Portefeuille 1** : `0x7F37f78cBD3D29260bE6fEd108C4B3c3bF462C46`
        - **Portefeuille 2** : `0x28C6c06298d514Db089934071355E5743bf21d60`
        - **Portefeuille 3** : `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`
        
        ### 🔍 Exemple de valeurs suspectes (mode manuel)
        
        - **Bot suspect** : Fréquence > 50 tx/h, Taux d'erreur > 30%
        - **Compte éphémère** : Durée de vie < 24h, nombreuses transactions
        - **Baleine suspecte** : Montants très élevés, durée de vie courte
        """)
