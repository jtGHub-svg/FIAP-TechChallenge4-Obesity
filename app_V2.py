
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit as st
from streamlit_option_menu import option_menu

# ===============================
# Paths & helpers
# ===============================
CAMINHO_ATUAL = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(CAMINHO_ATUAL, 'models', 'modelo_obesidade.pkl')
DATA_PATH = os.path.join(CAMINHO_ATUAL, 'data', 'processed', 'Obesity_tratado.csv')

FEATURES_ORDER = [
    'idade', 'altura', 'peso', 'imc',
    'genero', 'historico_familiar_sobrepeso',
    'consumo_alimentos_caloricos', 'fumante', 'monitora_calorias',
    'consumo_vegetais', 'refeicoes_por_dia', 'lanches_entre_refeicoes',
    'consumo_agua_por_dia', 'atividade_fisica_semana', 'tempo_eletronicos_por_dia',
    'consumo_alcoolico', 'meio_transporte'
]

@st.cache_resource
def load_model():
    return joblib.load(MODELO_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Helper para extrair nomes das features pós-ColumnTransformer
# Funciona com MinMaxScaler (sem nomes próprios), passthrough e OneHotEncoder

def get_feature_names_from_ct(ct, input_features):
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if transformer == 'passthrough':
            # Mantém nomes originais das colunas
            names.extend(list(cols))
        else:
            # Tentativa de obter nomes a partir do transformer
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    tnames = transformer.get_feature_names_out(cols)
                except Exception:
                    tnames = transformer.get_feature_names_out()
                names.extend(list(tnames))
            else:
                # Para scalers, usamos os nomes originais
                names.extend(list(cols))
    return names

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    selected = option_menu(
        'Projeto Saúde',
        ["Predição", "Dashboard", "Sobre"],
        icons=['activity', 'bar-chart', 'info-circle'],
        menu_icon='heart',
        default_index=1,
        key='menu_principal'
    )

modelo = load_model()

st.set_page_config(page_title='Projeto Saúde – Obesidade', layout='wide')

# ===============================
# PÁGINA: PREDIÇÃO
# ===============================
if selected == 'Predição':
    st.title('Predição à Obesidade')
    st.divider()
    st.subheader('1. Dados pessoais do paciente')

    col1, col2 = st.columns(2)
    with col1:
        genero = st.selectbox('Gênero', ['Feminino', 'Masculino'])
        idade = st.number_input('Idade', min_value=10, max_value=100, value=25, step=1)
    with col2:
        altura = st.number_input('Altura (em metros)', min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        peso = st.number_input('Peso (em kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    st.divider()
    st.subheader('2. Dados Gerais')
    col3, col4 = st.columns(2)
    with col3:
        historico_familiar_sobrepeso = st.radio('Histórico familiar de obesidade?', ['Sim', 'Não'])
        consumo_alimentos_caloricos = st.radio('Consome alimentos calóricos?', ['Sim', 'Não'])
        refeicoes_por_dia = st.selectbox('Número de refeições por dia', [1, 2, 3, '4 ou mais'])
        consumo_agua_por_dia = st.selectbox('Consumo de água por dia', ['Menos de 1 litro', 'Entre 1 e 2 litros', 'Mais de 2 litros'])
        consumo_alcoolico = st.selectbox('Consumo alcoólico?', ['Nunca', 'As vezes', 'Frequentemente', 'Sempre'])
        atividade_fisica_semana = st.selectbox('Pratica atividade física?', ['Nenhuma', '1 a 2 vezes por semana', '3 a 4 vezes por semana', '5 ou mais vezes por semana'])
    with col4:
        fumante = st.radio('Fumante?', ['Sim', 'Não'])
        monitora_calorias = st.radio('Monitora consumo de calorias?', ['Sim', 'Não'])
        lanches_entre_refeicoes = st.selectbox('Lanche entre refeições?', ['Nunca', 'As vezes', 'Frequentemente', 'Sempre'])
        consumo_vegetais = st.selectbox('Consumo de vegetais?', ['Raramente', 'As vezes', 'Sempre'])
        tempo_eletronicos_por_dia = st.selectbox('Tempo com eletrônicos/telas por dia', ['Menos de 2 horas', '3 a 5 horas', 'Maior que 5 horas'])
        meio_transporte = st.selectbox('Meio de transporte habitual', ['Caminhada', 'Transporte público', 'Bicicleta', 'Motocicleta', 'Automóvel'])

    st.divider()
    st.subheader('3. Gerar Análise')

    # Mapas
    map_sim_nao = {'Sim': 1, 'Não': 0}
    map_genero = {'Feminino': 0, 'Masculino': 1}
    map_refeicoes = {1: 'uma_refeicao_dia', 2: 'duas_refeicao_dia', 3: 'tres_refeicao_dia', '4 ou mais': 'quatro_ou_mais_refeicao_dia'}
    map_consumo_agua = {'Menos de 1 litro': 'menor_um_litro', 'Entre 1 e 2 litros': 'entre_um_dois_litro', 'Mais de 2 litros': 'maior_dois_litro'}
    map_consumo_vegetais = {'Raramente': 'raramente', 'As vezes': 'as_vezes', 'Sempre': 'sempre'}
    map_lanches_alcool = {'Nunca': 'no', 'As vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    map_atividade_fisica = {'Nenhuma': 'nenhuma', '1 a 2 vezes por semana': 'uma_a_duas_por_semana', '3 a 4 vezes por semana': 'tres_a_quatro_por_semana', '5 ou mais vezes por semana': 'cinco_ou_mais_por_semana'}
    map_tempo_eletronicos = {'Menos de 2 horas': 'menor_igual_duas_hora_dia', '3 a 5 horas': 'tres_a_cinco_hora_dia', 'Maior que 5 horas': 'maior_cinco_hora_dia'}
    map_transporte = {'Caminhada': 'Walking', 'Bicicleta': 'Bike', 'Transporte público': 'Public_Transportation', 'Motocicleta': 'Motorbike', 'Automóvel': 'Automobile'}

    if st.button('Analisar'):
        dados_paciente = pd.DataFrame({
            'idade': [idade],
            'altura': [altura],
            'peso': [peso],
            'imc': [peso / (altura ** 2)],
            'genero': [map_genero[genero]],
            'historico_familiar_sobrepeso': [map_sim_nao[historico_familiar_sobrepeso]],
            'consumo_alimentos_caloricos': [map_sim_nao[consumo_alimentos_caloricos]],
            'fumante': [map_sim_nao[fumante]],
            'monitora_calorias': [map_sim_nao[monitora_calorias]],
            'consumo_vegetais': [map_consumo_vegetais[consumo_vegetais]],
            'refeicoes_por_dia': [map_refeicoes[refeicoes_por_dia]],
            'lanches_entre_refeicoes': [map_lanches_alcool[lanches_entre_refeicoes]],
            'consumo_agua_por_dia': [map_consumo_agua[consumo_agua_por_dia]],
            'atividade_fisica_semana': [map_atividade_fisica[atividade_fisica_semana]],
            'tempo_eletronicos_por_dia': [map_tempo_eletronicos[tempo_eletronicos_por_dia]],
            'consumo_alcoolico': [map_lanches_alcool[consumo_alcoolico]],
            'meio_transporte': [map_transporte[meio_transporte]]
        })

        previsao = modelo.predict(dados_paciente)
        probabilidade = modelo.predict_proba(dados_paciente)

        st.markdown('### Resultado da Análise')
        if previsao[0] == 1:
            st.error('O paciente está propenso a Obesidade.')
        else:
            st.success('O paciente não está propenso a Obesidade.')
        st.write(f"Probabilidade de ser Obeso: {100*probabilidade[0][1]:.2f}%")

# ===============================
# PÁGINA: DASHBOARD
# ===============================
if selected == 'Dashboard':
    st.title('Dashboard de Obesidade')
    df = load_data()

    # KPIs principais
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric('Registros', f"{len(df):,}".replace(',', '.'))
    with colB:
        risco_pct = 100 * df['risco_obesidade'].mean()
        st.metric('% em risco', f"{risco_pct:.1f}%")
    with colC:
        st.metric('IMC médio', f"{df['imc'].mean():.2f}")
    with colD:
        st.metric('Idade média', f"{df['idade'].mean():.1f} anos")

    st.divider()
    st.subheader('1) Distribuições e segmentos')

    col1, col2 = st.columns(2)
    with col1:
        # Distribuição do IMC
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.histplot(df['imc'], bins=30, kde=True, color='#1f77b4', ax=ax1)
        ax1.set_title('Distribuição de IMC')
        ax1.set_xlabel('IMC')
        ax1.set_ylabel('Contagem')
        st.pyplot(fig1)

    with col2:
        # Risco por gênero
        genero_map = {0: 'Feminino', 1: 'Masculino'}
        tmp = df.copy()
        tmp['genero_label'] = tmp['genero'].map(genero_map)
        risco_por_genero = tmp.groupby('genero_label')['risco_obesidade'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.barplot(x='genero_label', y='risco_obesidade', data=risco_por_genero, ax=ax2, palette='Set2')
        ax2.set_title('Taxa de risco por gênero')
        ax2.set_xlabel('Gênero')
        ax2.set_ylabel('Taxa de risco')
        ax2.set_ylim(0,1)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        # Boxplot IMC por categoria de obesidade
        fig3, ax3 = plt.subplots(figsize=(7,4))
        sns.boxplot(x='obesidade', y='imc', data=df, ax=ax3)
        ax3.set_title('IMC por Categoria de Obesidade')
        ax3.set_xlabel('Categoria')
        ax3.set_ylabel('IMC')
        for label in ax3.get_xticklabels():
            label.set_rotation(30)
        st.pyplot(fig3)

    with col4:
        # Correlação (numéricas)
        num_cols = ['idade','altura','peso','imc']
        corr = df[num_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(5,4))
        sns.heatmap(corr, annot=True, cmap='Blues', ax=ax4)
        ax4.set_title('Correlação (variáveis numéricas)')
        st.pyplot(fig4)

    st.divider()
    st.subheader('2) Performance do modelo (avaliado no dataset)')

    # Avaliar pipeline no dataset inteiro
    X = df[FEATURES_ORDER]
    y = df['risco_obesidade']
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:,1]

    # Matriz de confusão (percentual por classe real)
    cm = confusion_matrix(y, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig5, ax5 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax5)
    ax5.set_title('Matriz de Confusão (%)')
    ax5.set_xlabel('Previsto')
    ax5.set_ylabel('Real')
    st.pyplot(fig5)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig6, ax6 = plt.subplots(figsize=(5,4))
    ax6.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax6.plot([0,1], [0,1], linestyle='--', color='gray')
    ax6.set_xlabel('Taxa de Falsos Positivos')
    ax6.set_ylabel('Taxa de Verdadeiros Positivos')
    ax6.set_title('Curva ROC')
    ax6.legend(loc='lower right')
    st.pyplot(fig6)

    st.divider()
    st.subheader('3) Importância das features (RandomForest)')

    # Importância das features do modelo final
    try:
        rf = modelo.named_steps['model']
        ct = modelo.named_steps['preprocessamento']
        feature_names = get_feature_names_from_ct(ct, FEATURES_ORDER)
        importancias = rf.feature_importances_
        fi = pd.DataFrame({'feature': feature_names, 'importancia': importancias}).sort_values('importancia', ascending=False)
        top_n = st.slider('Top N', min_value=5, max_value=30, value=15)
        fig7, ax7 = plt.subplots(figsize=(8,6))
        sns.barplot(x='importancia', y='feature', data=fi.head(top_n), ax=ax7, palette='viridis')
        ax7.set_title('Importância das Features (Top N)')
        ax7.set_xlabel('Importância (Ganho)')
        ax7.set_ylabel('Feature')
        st.pyplot(fig7)
    except Exception as e:
        st.warning(f'Não foi possível extrair importância das features automaticamente. Detalhes: {e}')

# ===============================
# PÁGINA: SOBRE
# ===============================
if selected == 'Sobre':
    st.title('Sobre o Projeto')
    st.markdown('''
    Este aplicativo oferece **predição de risco de obesidade** e um **dashboard analítico**
    construído sobre o dataset tratado (`data/processed/Obesity_tratado.csv`) e o pipeline
    salvo em `models/modelo_obesidade.pkl`.

    **Como usar:**
    1. Abra a aba *Predição* para estimar o risco de um paciente individual.
    2. Abra a aba *Dashboard* para explorar distribuições, segmentos, performance do modelo e a importância das features.
    ''')
