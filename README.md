# 🔍 Ethereum Anomalies Detection

Détecteur d'anomalies non supervisé pour portefeuilles Ethereum utilisant Isolation Forest et Autoencoder en consensus.

## 🎯 Objectif

Détecter les comportements suspects sur la blockchain Ethereum sans données labellisées (approche zero-shot).

## 🏗️ Architecture

- **Isolation Forest** : Détection d'outliers statistiques
- **Autoencoder** : Détection de patterns anormaux par erreur de reconstruction
- **Consensus** : Combinaison des deux modèles pour une détection robuste

## 📊 Features analysées

1. Nombre total de transactions
2. Total ETH envoyé
3. Moyenne ETH par transaction
4. Montant max par transaction
5. Durée de vie du portefeuille (heures)
6. Fréquence des transactions (tx/heure)
7. Taux d'erreur des transactions (%)

## 🚀 Démo en ligne
(https://ethereum-anomalies-detection.streamlit.app/)

## 🛠️ Installation locale

```bash
git clone https://github.com/brunochazeaud17-crypto/ethereum-fraud-detection.git
cd ethereum-fraud-detection
pip install -r requirements.txt
streamlit run app/streamlit_app.py
