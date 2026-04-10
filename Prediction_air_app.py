import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px ## Visualisation interactive des données
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import io


from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr


import warnings
warnings.filterwarnings('ignore')

st.title("Prediction de la qualité de l'air")

st.sidebar.write("Sommaire")

st.write("Auteur : HONFOGA Houénagnon Marc")
df = pd.read_csv("updated_pollution_dataset.csv")

pages=["Contexte du projet","Exploration des données","Analyse de donnéé","Modélisation"]
page=st.sidebar.radio("Aller à la page :",pages)


if page==pages[0]:
    st.write("### Contexte du projet")
    st.write("Ce projet s'inscrit dans un contexte de la pollution de l'air")
    st.write("### Source des données")
    st.markdown("Les données utilisées dans le cadre de ce projet proviennent de Kaggle")
    st.write("**Explication sur le Jeux de donnée**")
    st.markdown(
        """
    <div style="text-align: justify;">
    Cet ensemble de données vise à évaluer la qualité de l'air dans différentes régions. Il comprend 5 000 échantillons, prenant en compte les principaux facteurs environnementaux et démographiques qui influent sur les niveaux de pollution.Caractéristiques principales :Température (°C) : La température moyenne dans la région.Humidité (%): L'humidité relative enregistrée dans la région.Concentration de PM2,5 (µg/m³) : Niveaux de particules fines.Concentration de PM10 (µg/m³): Niveaux de particules grossières. Concentration de NO2 (ppb): Niveaux de dioxyde d'azote. Concentration de SO2 (ppb): Niveaux de dioxyde de soufre.Concentration de CO (ppm) : Niveaux de monoxyde de carbone. Proximité des zones industrielles (km) : La distance à la zone industrielle la plus proche.Densité de population (habitants/km²) : Le nombre d'habitants par kilomètre carré dans la région. Variable cible : Niveaux de qualité de l'air Bien : Air pur avec une pollution minimale.Modérée : Qualité de l'air acceptable, avec quelques polluants.Mauvaise qualité : Pollution notable, potentiellement nocive pour les personnes sensibles.Dangereux : Air fortement pollué, présentant des risques sanitaires importants pour la population.")  
    </div>
    """, 
    unsafe_allow_html=True
)

elif page==pages[1]:


    st.write('### 🔍 Exploration des données')

# Section 1 : Aperçu des données
    if  st.button("Affichage des 5 premières lignes"):
        st.markdown("**Top 5 du dataset :**")
    # Ajout d'un dégradé de couleur pour rendre le tableau plus vivant
        st.dataframe(df.head().style.background_gradient(cmap='Blues'))

# Section 2 : Types de données
    if  st.button("Information sur les types de données",key="btn_donnees"):
        st.write("**Analyse des formats :**")
    
    # Transformation en DataFrame pour un affichage propre
        df_types = df.dtypes.astype(str).to_frame(name='Type de donnée')
    
    # Coloration conditionnelle : Orange pour le texte, Vert pour les chiffres
        def color_types(val):
            color = 'orange' if 'obj' in val else 'lightgreen'
            return f'color: {color}; font-weight: bold'
    
        st.table(df_types.style.applymap(color_types))





    if st.button("Information sur les types de données"):
        st.markdown("### 🔍 Analyse des types")
        df_types = df.dtypes.astype(str).to_frame(name='Type')
    
    # On applique une couleur si c'est un objet (texte) pour le repérer vite
        styled_types = df_types.style.map(lambda x: 'color: orange' if 'obj' in x else 'color: green')
        st.table(styled_types)



    
    if st.button("Dimension du Jeux de donnée"):
        st.info(f"📏 Le jeu de données contient **{df.shape[0]}** lignes et **{df.shape[1]}** colonnes.")



    if  st.button("Afficher les valeurs manquantes"):
        st.subheader("🔍 Analyse du vide (NaN)")
    
    # Calculs
        missing_count = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(2)
    
    # Affichage de métriques en colonnes
        total_missing = missing_count.sum()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total valeurs manquantes", f"{total_missing}")
        with col2:
            st.metric("Moyenne globale", f"{missing_pct.mean():.2f}%")

    # Tableau stylisé (dégradé de rouge pour les valeurs critiques)
        st.write("**Détails par colonne :**")
        df_missing = pd.DataFrame({'Absents': missing_count, 'Pourcentage (%)': missing_pct})
        st.dataframe(df_missing.style.background_gradient(cmap='Reds'))

    # Graphique amélioré avec Seaborn
        st.write("**Graphique des pourcentages :**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=missing_pct.index, y=missing_pct.values, palette="magma", ax=ax)
    
    # Personnalisation
        ax.set_title('Taux de données manquantes (%)', fontsize=14, color='darkred')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel('% Manquant')
    
    # Affichage du graphique
        st.pyplot(fig)




    if st.button('🔍 Vérifier les doublons', type="primary"):
        n_doublons = df.duplicated().sum()
        if n_doublons > 0:
            st.error(f"⚠️ Attention : {n_doublons} doublons trouvés !")
            st.dataframe(df[df.duplicated()])
        else:
            st.success("✅ Aucun doublon détecté dans le dataset.")




    if  st.button("Statistique Descriptives", type="primary"):
        st.markdown("### 📈 Aperçu des statistiques")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
        
elif page==pages[2]:
    st.write("### Analyse de donnée")
    var_num=df.select_dtypes(exclude="object")

    st.write("### Visualisation des variables explicatives en fonction de la variable cible")
    if  st.checkbox("Visualisation des variables quantitative ( Analyse univarié)"):
        for col in var_num.columns:
            fig, ax=plt.subplots()
            sns.histplot(data=df,x=col,hue='Air Quality',element="step",kde=True,ax=ax)
            ax.set_title(col)
            st.pyplot(fig)


    st.write("### Matrice de corrélation")  
    fig,ax=plt.subplots()
    corr_matrix = var_num.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f", square=True, linewidths=.5)
    st.pyplot(fig)

    st.write("### Detection des relations existantes entre les variables")
    if  st.checkbox("Pairplot") :
        pairplot=sns.pairplot(df,hue='Air Quality')
        st.pyplot(pairplot.fig)

    st.write("# Visualisation des modalités de la variable cible")  
    # Sélectionner la variable à analyser
    variable_selected = st.sidebar.selectbox(
    "Sélectionnez une variable:",
    ['Air Quality', 'PM2.5', 'PM10', 'NO2', 'Temperature', 'Humidity','SO2','CO','Proximity_to_Industrial_Areas','Population_Density']

)
    if variable_selected == 'Air Quality':
        # Pour variable catégorique
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## Graphique en Barres")
            
            # Créer la figure avec matplotlib
            fig_count, ax = plt.subplots(figsize=(8, 5))
            
            # Compter les occurrences
            counts = df[variable_selected].value_counts()
            
            # Créer le graphique en barres
            bars = ax.bar(
                counts.index,
                counts.values,
                color='green',
                edgecolor='red',
                linewidth=2,
                alpha=0.8
            )
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=11
                )
            
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_xlabel('Catégories', fontsize=12, fontweight='bold')
            ax.set_title(f'Distribution de {variable_selected}', fontsize=13, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig_count)
        
        with col2:
            st.markdown("## Graphique Circulaire")
            
            # Créer un graphique circulaire avec Plotly
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                title=f'Distribution en pourcentage de {variable_selected}',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    

    st.write("# Boite à moustache par à la qualité de l'air")
    def afficher_px_box(data, x_col, y_col, title):
    # Création de la figure Plotly
        fig = px.box(data, x=x_col, y=y_col, color=x_col, title=title)
    # Affichage dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # Boucle sur les colonnes numériques
    # On suppose que df est votre DataFrame et var_num contient les colonnes numériques
    for i in var_num.columns:
        afficher_px_box(
        data=df, 
        x_col='Air Quality', # Nom de la colonne sous forme de chaîne
        y_col=i,             # Nom de la variable numérique
        title=f"Boîte à moustache de {i}"
    )
    else:
        st.write("### VISUALISATION DES MODALITÉS")
        # Pour variables numériques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Histogramme")
            
            fig_hist, ax = plt.subplots(figsize=(8, 5))
            
            ax.hist(
                df[variable_selected].dropna(),
                bins=30,
                color='skyblue',
                edgecolor='black',
                alpha=0.7
            )
            
            ax.set_xlabel(variable_selected, fontsize=12, fontweight='bold')
            ax.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
            ax.set_title(f'Distribution de {variable_selected}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig_hist)
        
        with col2:
            st.markdown("### Distribution (Kernel Density)")
            
            fig_kde = px.histogram(
                df,
                x=variable_selected,
                nbins=30,
                title=f'Distribution avec KDE de {variable_selected}',
                marginal='box',
                hover_data=df.columns
            )
            
            st.plotly_chart(fig_kde, use_container_width=True)


    st.write("# Valeurs abberrantes")   

    



elif page==pages[3]:
    st.write("### La modélisation")
    st.subheader("Machine Learning Automatique")
    target = st.selectbox("Choisissez variable cible (Air Quality)", df.columns)
    mapping = {
    "Good": 1, 
    "Moderate": 2, 
    "Poor": 3,
    "Hazardous": 4 }

# 2. Application conditionnelle
    if target == 'Air Quality':
    # On crée une nouvelle colonne ou on remplace l'existante pour le modèle
        df[target] = df[target].map(mapping)
        st.write("✅ Encodage de 'Air Quality' effectué.")
    else:
    # Notification si une autre variable est choisie
        st.warning(f"⚠️ '{target}' vous n'avez pas selectionné la bonne variable cible.Aucun mapping manuel appliqué.")

    st.write("# Séparer les variables indépendantes et la variable dépendante")

    y = df[target]
    X = df.drop('Air Quality',axis=1)

    st.write("# Division du jeu de données :")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    
    
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X)
    X_train_selected = scaler.fit_transform(X_train)
    X_test_selected = scaler.transform(X_test)

    st.success("✅ Données normalisées avec succès !")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    st.success("✅ Analyse en ccomposante Principale réalisée avec succès !")


    st.title("Evaluation des Modèles de Classification")

# Liste de tes modèles
    models = [
    LogisticRegression(max_iter=1000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    GaussianNB()
    ]


    model_names = [
    "Logistic Regression",
    "SVM",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
    "Naive Bayes"
    ]
    scores=[]


    if st.button('🚀 Lancer l\'entraînement avec les données Normales'):
    # 1. Préparation des indicateurs visuels
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = [] # Pour stocker les scores et faire le classement à la fin

# Boucle d'entraînement et d'affichage
        for model, name in zip(models, model_names):
    # On crée un bandeau déroulant pour chaque modèle
            with st.expander(f"📊 {name}", expanded=False):
        
        # Entraînement et prédiction
                model.fit(X_train_pca, y_train)
                y_pred = model.predict(X_test_pca)
                accuracy = model.score(X_test_pca, y_test)
        
        # Sauvegarde pour le graphique final
                scores.append({"Modèle": name, "Accuracy": accuracy})

        # Affichage des métriques clés
                st.metric(label=f"Précision {name}", value=f"{accuracy:.4f}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Matrice de Confusion :**")
                    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

                with col2:
                    st.markdown("**Rapport de Classification :**")
            # Transformation du rapport textuel en DataFrame exploitable
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

# --- BONUS : Graphique comparatif à la fin ---
        st.divider()
        st.subheader("📈 Comparatif Global")
        df_scores = pd.DataFrame(scores)
        st.bar_chart(data=df_scores, x="Modèle", y="Accuracy")

    st.title("Evaluation des Modèles de Classification avec les deux compsantes APC")


# Liste de tes modèles
    models = [
    LogisticRegression(max_iter=1000),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    GaussianNB()
    ]


    model_names = [
    "Logistic Regression",
    "SVM",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
    "Naive Bayes"
    ]
    scores=[]



    if st.button('🚀 Lancer l\'entraînement avec les données de ACP'):
    # 1. Préparation des indicateurs visuels
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = [] # Pour stocker les scores et faire le classement à la fin

    # Boucle d'entraînement et d'affichage
        for model, name in zip(models, model_names):
    # On crée un bandeau déroulant pour chaque modèle
            with st.expander(f"📊 {name}", expanded=False):
        
        # Entraînement et prédiction
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                accuracy = model.score(X_test_selected, y_test)
        
        # Sauvegarde pour le graphique final
                scores.append({"Modèle": name, "Accuracy": accuracy})

        # Affichage des métriques clés
                st.metric(label=f"Précision {name}", value=f"{accuracy:.4f}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Matrice de Confusion :**")
                    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

                with col2:
                    st.markdown("**Rapport de Classification :**")
            # Transformation du rapport textuel en DataFrame exploitable
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

# --- BONUS : Graphique comparatif à la fin ---
        st.divider()
        st.subheader("📈 Comparatif Global")
        df_scores = pd.DataFrame(scores)
        st.bar_chart(data=df_scores, x="Modèle", y="Accuracy")