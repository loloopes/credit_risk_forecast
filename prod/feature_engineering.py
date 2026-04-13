import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class CreditFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.target_cat_cols = [
            'tipo_contrato', 'status_contrato', 'tipo_pagamento', 
            'finalidade_emprestimo', 'tipo_cliente', 'tipo_portfolio', 
            'tipo_produto', 'categoria_bem', 'setor_vendedor', 'canal_venda'
        ]
        self.feature_names = feature_names

    def fit(self, X, y=None):
        actual_cat = [c for c in self.target_cat_cols if c in X.columns]
        if actual_cat:
            self.encoder.fit(X[actual_cat].fillna('Missing').astype(str))
        return self

    def transform(self, X):
        X_c = X.copy()

        # --- 1. Financeiro ---
        if 'renda_anual' in X_c.columns:
            renda_mensal = X_c['renda_anual'] / 12
            if 'valor_parcela' in X_c.columns:
                X_c['feat_comprometimento_renda'] = X_c['valor_parcela'] / renda_mensal.replace(0, np.nan)
            if 'qtd_membros_familia' in X_c.columns:
                X_c['feat_renda_per_capita'] = X_c['renda_anual'] / X_c['qtd_membros_familia'].replace(0, np.nan)

        if 'possui_carro' in X_c.columns:
            X_c['feat_possui_carro'] = X_c['possui_carro'].map({'Y': 1, 'N': 0}).fillna(0)
        if 'possui_imovel' in X_c.columns:
            X_c['feat_possui_imovel'] = X_c['possui_imovel'].map({'Y': 1, 'N': 0}).fillna(0)

        # --- 2. Estabilidade (Datas) ---
        if 'data_nascimento' in X_c.columns and 'data_decisao' in X_c.columns:
            X_c['data_nascimento'] = pd.to_datetime(X_c['data_nascimento'], errors='coerce')
            X_c['data_decisao'] = pd.to_datetime(X_c['data_decisao'], errors='coerce')
            X_c['feat_idade'] = (X_c['data_decisao'] - X_c['data_nascimento']).dt.days // 365

        # --- 3. Contrato e Contexto ---
        if 'valor_credito' in X_c.columns and 'valor_bem' in X_c.columns:
            X_c['feat_ltv'] = X_c['valor_credito'] / X_c['valor_bem'].replace(0, np.nan).fillna(X_c['valor_credito'])
        
        if 'hora_solicitacao' in X_c.columns:
            X_c['feat_hora_pico_fraude'] = X_c['hora_solicitacao'].apply(lambda x: 1 if x < 7 or x > 21 else 0)

        # --- Encoding ---
        actual_cat = [c for c in self.target_cat_cols if c in X_c.columns]
        if actual_cat:
            encoded = self.encoder.transform(X_c[actual_cat].fillna('Missing').astype(str))
            for i, col in enumerate(actual_cat):
                X_c[f'feat_{col}_enc'] = encoded[:, i]

        # --- Limpeza Final ---
        cols_to_drop = [
            'id_cliente', 'id_contrato', 'data_decisao', 'data_liberacao',
            'data_primeiro_vencimento', 'data_ultimo_vencimento',
            'data_ultimo_vencimento_original', 'data_encerramento', 'data_nascimento',
            'possui_carro', 'possui_imovel'
        ]
        X_final = X_c.drop(columns=[c for c in cols_to_drop if c in X_c.columns], errors='ignore')
        X_final = X_final.select_dtypes(include=[np.number])

        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X_final: X_final[col] = 0
            X_final = X_final[self.feature_names]

        return X_final