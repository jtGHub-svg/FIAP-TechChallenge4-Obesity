import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu

# ===============================
# Config & Paths
# ===============================
st.set_page_config(page_title='Projeto Saúde – Obesidade', layout='wide')

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

# ===============================
# Cache helpers
# ===============================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELO_PATH)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ===============================
# Helper para nomes pós-ColumnTransformer
# ===============================
def get_feature_names_from_ct(ct, input_features):
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if transformer == 'passthrough':
            names.extend(list(cols))
        else:
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    tnames = transformer.get_feature_names_out(cols)
                except Exception:
                    tnames = transformer.get_feature_names_out()
                names.extend(list(tnames))
            else:
                names.extend(list(cols))
    return names

# Agrega importâncias de one-hot pela feature base
def get_feature_importance_aggregated(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    try:
        pre = pipeline.named_steps.get('preprocessamento')
        model = pipeline.named_steps.get('model') or pipeline.named_steps.get('modelo')
        feature_names = pre.get_feature_names_out()
        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            return pd.DataFrame(columns=['variavel_base', 'importancia'])
        base_features = []
        for name in feature_names:
            base = name.split('__', 1)[1] if '__' in name else name
            if base not in X.columns:
                base = base.rsplit('_', 1)[0]
            base_features.append(base)
        df_imp = pd.DataFrame({'feature_encoded': feature_names, 'variavel_base': base_features, 'importancia': importances})
        agg = df_imp.groupby('variavel_base', as_index=False)['importancia'].sum().sort_values('importancia', ascending=False)
        return agg
    except Exception as e:
        st.warning(f'Falha ao extrair importâncias agregadas: {e}')
        return pd.DataFrame(columns=['variavel_base', 'importancia'])

# Taxa média de risco por categoria
def rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.groupby(col)['risco_obesidade'].mean().reset_index().rename(columns={'risco_obesidade': 'taxa_risco'})
    return tmp.sort_values('taxa_risco', ascending=False)

# ===============================
# Sidebar menu
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
    st.caption(f"IMC calculado automaticamente: **{peso/(altura**2):.2f}**")

    st.divider()
    st.subheader('2. Hábitos e rotina')
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

    # Mapas (compatíveis com o dataset e o pipeline)
    map_sim_nao = {'Sim': 1, 'Não': 0}
    map_genero = {'Feminino': 0, 'Masculino': 1}
    map_refeicoes = {1: 'uma_refeicao_dia', 2: 'duas_refeicao_dia', 3: 'tres_refeicao_dia', '4 ou mais': 'quatro_ou_mais_refeicao_dia'}
    map_consumo_agua = {'Menos de 1 litro': 'menor_um_litro', 'Entre 1 e 2 litros': 'entre_um_dois_litro', 'Mais de 2 litros': 'maior_dois_litro'}
    map_consumo_vegetais = {'Raramente': 'raramente', 'As vezes': 'as_vezes', 'Sempre': 'sempre'}
    map_lanches_alcool = {'Nunca': 'no', 'As vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    map_atividade_fisica = {'Nenhuma': 'nenhuma', '1 a 2 vezes por semana': 'uma_a_duas_por_semana', '3 a 4 vezes por semana': 'tres_a_quatro_por_semana', '5 ou mais vezes por semana': 'cinco_ou_mais_por_semana'}
    map_tempo_eletronicos = {'Menos de 2 horas': 'menor_igual_duas_hora_dia', '3 a 5 horas': 'tres_a_cinco_hora_dia', 'Maior que 5 horas': 'maior_cinco_hora_dia'}
    map_transporte = {'Caminhada': 'Walking', 'Bicicleta': 'Bike', 'Transporte público': 'Public_Transportation', 'Motocicleta': 'Motorbike', 'Automóvel': 'Automobile'}

    # Threshold customizado
    thr = st.slider('Limite de decisão (probabilidade)', min_value=0.10, max_value=0.90, value=0.50, step=0.05)

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
        # Reindex para garantir ordem/colunas
        dados_paciente = dados_paciente.reindex(columns=FEATURES_ORDER)

        prob = modelo.predict_proba(dados_paciente)[0, 1]
        pred_thr = int(prob >= thr)
        pred_default = modelo.predict(dados_paciente)[0]

        st.markdown('### Resultado da Análise')
        cA, cB, cC = st.columns(3)
        cA.metric('Predição (modelo)', int(pred_default))
        cB.metric('Probabilidade', f"{prob:.3f}")
        cC.metric('Decisão com limite', f">= {thr:.2f} → {pred_thr}")

        if pred_thr == 1:
            st.error('O paciente está propenso a Obesidade (pela regra de decisão).')
        else:
            st.success('O paciente não está propenso a Obesidade (pela regra de decisão).')
        st.info('Interpretação: a classificação é baseada na probabilidade prevista pelo modelo e no limite de decisão escolhido. Este resultado é informativo e não substitui avaliação clínica.')

# ===============================
# PÁGINA: DASHBOARD
# ===============================
if selected == 'Dashboard':
    st.title('Dashboard de Obesidade')
    df = load_data()

    # Binning faixa etária
    df['faixa_idade'] = pd.cut(
        df['idade'], bins=[0, 24, 34, 44, 54, 120], labels=['≤24', '25–34', '35–44', '45–54', '55+'], right=True
    )

    # Filtros (sidebar)
    st.sidebar.subheader('🎛️ Filtros do Dashboard')
    genero_opt = st.sidebar.multiselect('Gênero', options=[0, 1], default=[0, 1], format_func=lambda x: 'Feminino' if x == 0 else 'Masculino')
    faixa_opt = st.sidebar.multiselect('Faixa etária', options=list(df['faixa_idade'].dropna().unique()), default=list(df['faixa_idade'].dropna().unique()))
    mtrans_opt = st.sidebar.multiselect('Meio de transporte', options=['Public_Transportation','Walking','Automobile','Motorbike','Bike'], default=['Public_Transportation','Walking','Automobile','Motorbike','Bike'], format_func=lambda x: {'Public_Transportation':'Transporte público','Walking':'Caminhada','Automobile':'Automóvel','Motorbike':'Motocicleta','Bike':'Bicicleta'}[x])

    df_filt = df[df['genero'].isin(genero_opt)]
    if len(faixa_opt) > 0:
        df_filt = df_filt[df_filt['faixa_idade'].isin(faixa_opt)]
    if len(mtrans_opt) > 0:
        df_filt = df_filt[df_filt['meio_transporte'].isin(mtrans_opt)]

    # KPIs principais (com filtro)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric('Registros (filtro)', f"{len(df_filt):,}".replace(',', '.'))
    with colB:
        risco_pct = 100 * df_filt['risco_obesidade'].mean()
        st.metric('% em risco', f"{risco_pct:.1f}%")
    with colC:
        st.metric('IMC médio', f"{df_filt['imc'].mean():.2f}")
    with colD:
        st.metric('% Sobrepeso/Obesidade', f"{100*(df_filt['obesidade'].isin(['Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']).mean()):.1f}%")

    st.divider()
    st.subheader('1) Distribuições e segmentos')
    col1, col2 = st.columns(2)
    with col1:
        # IMC por categoria de obesidade (boxplot)
        fig1, ax1 = plt.subplots(figsize=(7,4))
        sns.boxplot(x='obesidade', y='imc', data=df_filt, ax=ax1)
        ax1.set_title('IMC por Categoria de Obesidade')
        ax1.set_xlabel('Categoria')
        ax1.set_ylabel('IMC')
        for label in ax1.get_xticklabels():
            label.set_rotation(30)
        st.pyplot(fig1)
    with col2:
        # Distribuição do risco 0/1
        fig2, ax2 = plt.subplots(figsize=(7,4))
        sns.countplot(x='risco_obesidade', data=df_filt, ax=ax2)
        ax2.set_title('Distribuição de risco (0/1)')
        ax2.set_xlabel('Risco')
        ax2.set_ylabel('Contagem')
        st.pyplot(fig2)

    # Risco por faixa etária e gênero
    st.markdown('**Risco médio por faixa etária e gênero**')
    fig3, ax3 = plt.subplots(figsize=(10,4))
    tmp = df_filt.copy()
    tmp['genero_label'] = tmp['genero'].map({0:'Feminino',1:'Masculino'})
    risco_etario = tmp.groupby(['faixa_idade','genero_label'])['risco_obesidade'].mean().reset_index()
    sns.barplot(x='faixa_idade', y='risco_obesidade', hue='genero_label', data=risco_etario, ax=ax3)
    ax3.set_xlabel('Faixa etária')
    ax3.set_ylabel('Risco médio')
    st.pyplot(fig3)

    # Hábitos associados ao risco (barras horizontais)
    st.subheader('2) Hábitos associados ao risco')
    c1, c2, c3 = st.columns(3)
    with c1:
        tmp = rate_by_category(df_filt, 'atividade_fisica_semana')
        tmp['atividade_fisica_semana'] = tmp['atividade_fisica_semana'].map({'nenhuma':'nenhuma','uma_a_duas_por_semana':'1–2x/sem','tres_a_quatro_por_semana':'3–4x/sem','cinco_ou_mais_por_semana':'5x+/sem'})
        figA, axA = plt.subplots(figsize=(5,3))
        sns.barplot(y='atividade_fisica_semana', x='taxa_risco', data=tmp, ax=axA, orient='h')
        axA.set_xlabel('Taxa de risco')
        axA.set_ylabel('Atividade física')
        st.pyplot(figA)
    with c2:
        tmp = rate_by_category(df_filt, 'tempo_eletronicos_por_dia')
        tmp['tempo_eletronicos_por_dia'] = tmp['tempo_eletronicos_por_dia'].map({'menor_igual_duas_hora_dia':'≤ 2 h/dia','tres_a_cinco_hora_dia':'3–5 h/dia','maior_cinco_hora_dia':'> 5 h/dia'})
        figB, axB = plt.subplots(figsize=(5,3))
        sns.barplot(y='tempo_eletronicos_por_dia', x='taxa_risco', data=tmp, ax=axB, orient='h')
        axB.set_xlabel('Taxa de risco')
        axB.set_ylabel('Tempo de tela')
        st.pyplot(figB)
    with c3:
        tmp = rate_by_category(df_filt, 'consumo_alimentos_caloricos')
        figC, axC = plt.subplots(figsize=(5,3))
        sns.barplot(y='consumo_alimentos_caloricos', x='taxa_risco', data=tmp, ax=axC, orient='h')
        axC.set_xlabel('Taxa de risco')
        axC.set_ylabel('Alimentos calóricos (0/1)')
        st.pyplot(figC)

    c4, c5, c6 = st.columns(3)
    with c4:
        tmp = rate_by_category(df_filt, 'historico_familiar_sobrepeso')
        figD, axD = plt.subplots(figsize=(5,3))
        sns.barplot(y='historico_familiar_sobrepeso', x='taxa_risco', data=tmp, ax=axD, orient='h')
        axD.set_xlabel('Taxa de risco')
        axD.set_ylabel('Histórico familiar (0/1)')
        st.pyplot(figD)
    with c5:
        tmp = rate_by_category(df_filt, 'lanches_entre_refeicoes')
        tmp['lanches_entre_refeicoes'] = tmp['lanches_entre_refeicoes'].map({'no':'não','Sometimes':'às vezes','Frequently':'frequentemente','Always':'sempre'})
        figE, axE = plt.subplots(figsize=(5,3))
        sns.barplot(y='lanches_entre_refeicoes', x='taxa_risco', data=tmp, ax=axE, orient='h')
        axE.set_xlabel('Taxa de risco')
        axE.set_ylabel('Lanches')
        st.pyplot(figE)
    with c6:
        tmp = rate_by_category(df_filt, 'consumo_agua_por_dia')
        tmp['consumo_agua_por_dia'] = tmp['consumo_agua_por_dia'].map({'menor_um_litro':'< 1 L','entre_um_dois_litro':'1–2 L','maior_dois_litro':'> 2 L'})
        figF, axF = plt.subplots(figsize=(5,3))
        sns.barplot(y='consumo_agua_por_dia', x='taxa_risco', data=tmp, ax=axF, orient='h')
        axF.set_xlabel('Taxa de risco')
        axF.set_ylabel('Água/dia')
        st.pyplot(figF)

    st.divider()
    st.subheader('3) Performance do modelo (avaliado no filtro)')
    # Avaliar pipeline no dataset filtrado
    X = df_filt[FEATURES_ORDER]
    y = df_filt['risco_obesidade']

    # Controle de limite de decisão
    thr_dash = st.slider('Limite de decisão (probabilidade) para o dashboard', min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    y_prob = modelo.predict_proba(X)[:,1]
    y_pred_thr = (y_prob >= thr_dash).astype(int)

    acc = accuracy_score(y, y_pred_thr)
    st.metric('Acurácia (com limite)', f"{acc:.3f}")

    # Matriz de confusão (% por classe real)
    cm = confusion_matrix(y, y_pred_thr)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig5, ax5 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax5)
    ax5.set_title('Matriz de Confusão (%) — limite aplicado')
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
    st.subheader('4) Importância das variáveis (agregada)')
    try:
        agg_imp = get_feature_importance_aggregated(modelo, X)
        if agg_imp.empty:
            st.warning('O modelo atual não fornece importâncias ou houve falha na extração.')
        else:
            rename_pt = {
                'consumo_vegetais': 'consumo de vegetais',
                'refeicoes_por_dia': 'refeições por dia',
                'lanches_entre_refeicoes': 'lanches entre refeições',
                'consumo_agua_por_dia': 'consumo de água/dia',
                'atividade_fisica_semana': 'atividade física/semana',
                'tempo_eletronicos_por_dia': 'tempo em eletrônicos/dia',
                'consumo_alcoolico': 'consumo alcoólico',
                'meio_transporte': 'meio de transporte',
                'idade': 'idade',
                'altura': 'altura',
                'peso': 'peso',
                'imc': 'IMC',
                'genero': 'gênero',
                'historico_familiar_sobrepeso': 'histórico familiar',
                'consumo_alimentos_caloricos': 'alimentos calóricos',
                'fumante': 'fumante',
                'monitora_calorias': 'monitora calorias',
            }
            agg_imp['variavel'] = agg_imp['variavel_base'].map(lambda x: rename_pt.get(x, x))
            top_n = st.slider('Top N', min_value=5, max_value=30, value=15)
            fig7, ax7 = plt.subplots(figsize=(8,6))
            sns.barplot(x='importancia', y='variavel', data=agg_imp.head(top_n), ax=ax7, orient='h', palette='viridis')
            ax7.set_title('Importância das Variáveis (soma dos one-hot)')
            ax7.set_xlabel('Importância')
            ax7.set_ylabel('Variável')
            st.pyplot(fig7)
            st.dataframe(agg_imp.rename(columns={'variavel_base':'variável_base'}))
    except Exception as e:
        st.warning(f'Não foi possível extrair importância agregada: {e}')

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
    2. Abra a aba *Dashboard* para explorar distribuições, segmentos, performance do modelo e a importância das variáveis.

    **Aviso:** Este painel é informativo e não substitui avaliação médica ou nutricional profissional.
    ''')
