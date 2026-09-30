# Programação (Módulo 3) — 30 questões contextualizadas (fácil → difícil)

**Base:** slides da Profª Crishna (Aulas 1–12), notebooks, e a *Lista de Estudos M03 PROG* que você subiu (a prova de recuperação segue esse estilo: cenário + dissertativa/cálculo). Todas as questões têm contexto de tecnologia e dos projetos (previsão de vendas de veículos da MAHLE, MLOps, e-commerce, streaming, fraude). Nenhum cenário repete os da lista.

**Como usar:** responda **por escrito**, como na prova. Depois compare com a resposta-modelo no final (o que precisa aparecer para pontuar). As saídas de código e as contas foram executadas/conferidas.

**Dificuldade:** Q1–8 Python, NumPy e Pandas · Q9–14 dados e pré-processamento · Q15–22 supervisionado e métricas · Q23–30 não supervisionado, recomendação, produção e integradoras.

---

## BLOCO 1 — Python, NumPy, Pandas

**Q1.** No script de ingestão dos dados de emplacamento (ANFAVEA) do projeto MAHLE, um analista testa alguns operadores no console. Qual é a saída?
```python
print(type(7/2), 7//2, 7%2, "3"+"4")
```

**Q2.** Um script controla o estoque de peças de um lote: `s` acumula as peças expedidas e `c` é o estoque restante. Faça o teste de mesa (iteração, `c`, `s`) e dê a saída.
```python
s = 0
c = 10
while c > 0:
    s += c
    c -= 3
print(s, c)
```

**Q3.** A lista `n` guarda as vendas mensais (em mil unidades) de um segmento de veículos. Qual é a saída?
```python
n = [8, 6, 9, 5]
n.append(7)
n.insert(0, 10)
print(n, n[1:4], n[-2:], sum(n)/len(n), max(n))
```

**Q4.** O dicionário `d` guarda a configuração de um experimento de modelagem. Qual é a saída?
```python
d = {"a": 1, "b": [2, 3]}
d["c"] = d["a"] + len(d["b"])
d["a"] += 5
print(d, list(d.keys()), d["b"][-1])
```

**Q5.** Duas fábricas atendem conjuntos de montadoras. (a) Dê a saída de `print(sorted(A & B), sorted(A | B), sorted(A - B), len(A | B))` com `A = {"x","y","z"}` e `B = {"z","w"}`, e diga o que cada operador representa no negócio.
(b) Dado `t = (1, 2)` (par de coordenadas de um sensor), qual a saída de `t2 = t + (3, 4); print(t2, len(t2), t2[1:3])` e o que acontece com `t[0] = 9`? Por quê?

**Q6.** Uma função estima o custo de uma peça segundo o tipo (`p` é a quantidade). Qual é a saída de cada chamada?
```python
def f(p, tipo):
    if tipo == "a":
        fator = 2
    elif tipo == "b":
        fator = 5
    else:
        return "inválido"
    return p * fator

print(f(4, "b"), f(4, "c"), f(3, "a"))
```

**Q7.** Um analista calcula o faturamento por segmento com NumPy e compara com uma lista comum. Dê a saída e explique a diferença entre lista e array.
```python
import numpy as np
l = [1, 2, 3];  a = np.array([1, 2, 3])
print(l*2, a*2, a[a > 1], a.mean(), a.shape)
```

**Q8.** Dado o DataFrame de vendas por segmento do projeto MAHLE, dê a saída dos dois trechos.
```python
df = pd.DataFrame({"segmento":["SUV","Sedan","SUV","Pickup","Sedan"],
                   "vendas":[10,20,30,40,50],
                   "desc":[0.1,0.2,0.05,0.3,0.1]})
# (a)
df.loc[(df["vendas"] > 15) & (df["desc"] <= 0.1), ["segmento", "vendas"]]
# (b)
df.groupby("segmento").agg(total=("vendas","sum"), med=("vendas","median"))
```
Por que as condições do trecho (a) precisam de parênteses?

---

## BLOCO 2 — Dados e pré-processamento

**Q9.** Na telemetria de uma API de previsão de demanda, a coluna `tempo_resposta` (em ms) tem os valores `[4, 6, 5, 100, 5]` e mais um registro ausente (NaN) a preencher com `fillna`. (a) Calcule média e mediana. (b) Qual você usaria no `fillna` e por quê? (c) O 100 deve ser apagado automaticamente? Justifique.

**Q10.** Na base de emplacamentos do projeto MAHLE, as vendas mensais (em mil unidades) de um segmento são `[4, 5, 6, 7, 8, 9, 10, 11, 12, 60]`. Aplique a regra do IQR (quartis por interpolação linear, padrão do pandas/NumPy: Q1 = 6,25 e Q3 = 10,75): calcule IQR, os limites inferior e superior e diga quais valores são outliers. O que o critério faz e o que ele **não** faz? O que você investigaria antes de remover o 60?

**Q11.** A feature `preco_medio` (em mil R$) de cinco modelos de veículos é `x = [20, 30, 40, 50, 60]` (média 40; desvio-padrão populacional ≈ 14,14). (a) Padronize o valor 60 (StandardScaler). (b) Aplique Min-Max ao valor 30. (c) Qual dos dois é mais sensível a outliers e por quê? (d) Escalonar torna a feature mais importante para o modelo?

**Q12.** A base do projeto de previsão de vendas tem `segmento` (SUV, Sedan, Pickup), `nivel_juros` (baixo < médio < alto), `renda_media` (numérica, com ausentes) e `comprou` (sim/não, o alvo). Para cada coluna indique o tratamento (imputação, encoding, escala), explique por que **não** se deve codificar `segmento` como 1, 2, 3, e onde o `LabelEncoder` é apropriado.

**Q13.** A MAHLE junta dados de um ERP, sensores da linha de produção e planilhas de fornecedores, para análises recorrentes de BI e também para experimentos de ciência de dados. (a) Diferencie ETL de ELT. (b) Diferencie Data Lake de Data Warehouse. (c) Recomende uma combinação e justifique.

**Q14.** Um engenheiro de ML escreveu o pipeline de treino de um modelo de demanda. Identifique o erro e corrija:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                 # 1
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)   # 2
model.fit(X_tr, y_tr)
```
Explique o que é *data leakage* e por que ele infla a avaliação. Escreva a ordem correta.

---

## BLOCO 3 — Aprendizado supervisionado e métricas

**Q15.** Uma startup de e-commerce quer prever quais pedidos serão devolvidos. Liste as 6 fases do CRISP-DM na ordem e dê **uma atividade concreta** de cada uma para esse projeto. Qual fase costuma consumir cerca de 80% do esforço? Por que o processo não é uma cascata rígida?

**Q16.** Uma plataforma de streaming quer prever, **no dia 1 do mês**, quais assinantes vão cancelar durante o mês. A base tem:

| Variável | Quando é registrada |
|---|---|
| `id_assinante` | no cadastro |
| `horas_assistidas_mes_anterior` | até o dia 1 |
| `n_reclamacoes_ultimos_3m` | até o dia 1 |
| `plano` | até o dia 1 |
| `motivo_cancelamento` | só existe **após** o cancelamento |
| `cancelou_no_mes` | ao final do mês |

Defina X e y, justifique cada inclusão/exclusão, aponte a variável que causa *leakage* e explique por que ela só é percebida em produção.

**Q17.** Um modelo de visão computacional inspeciona pistões na linha de produção da MAHLE e classifica cada peça como defeituosa (positivo) ou boa. Numa amostra de 200 peças, gerou: TP = 30, FP = 10, FN = 20, TN = 140. Calcule acurácia, precisão, recall, F1 e especificidade (mostre as fórmulas). Se um defeito não detectado vai para a montadora cliente, qual métrica priorizar?

**Q18.** Um banco monitora 1.000 transações, das quais 10 são fraudes. Um modelo "preguiçoso" prevê sempre "não fraude". (a) Calcule a acurácia e o recall da classe fraude. (b) O que esse resultado mostra? (c) Que métricas e que técnicas (duas) você usaria?

**Q19.** Uma fintech avalia 10 clientes de crédito: 5 pagam e 5 não. Dividindo por `tem_garantia`: **Sim** (4 clientes: 3 pagam, 1 não) e **Não** (6 clientes: 2 pagam, 4 não). Calcule a entropia do conjunto, a entropia de cada subgrupo e o **ganho de informação**. A divisão reduz a incerteza? (Use log₂; log₂(3) ≈ 1,585.)

**Q20.** No projeto MAHLE, um KNN classifica modelos de veículos em dois segmentos usando duas features padronizadas (potência e preço). No plano, o segmento A (compactos) = {(1,1), (2,1)} e o segmento B (utilitários) = {(3,3), (4,4), (5,4)}. Classifique o ponto (2,2) com KNN para **k = 1, k = 3 e k = 5** (distância euclidiana). O que a mudança do resultado ilustra sobre o hiperparâmetro k? Por que padronizar as features antes do KNN?

**Q21.** Um modelo de regressão prevê as vendas mensais de um segmento (em mil unidades). Valores reais y = [20, 30, 40, 50] e previsões ŷ = [20, 30, 43, 46]. Calcule MAE, MSE, RMSE e R². Por que o RMSE é maior que o MAE aqui e quando isso é desejável para o planejamento de produção? Quanto vale o R² de um modelo que sempre prevê a média?

**Q22.** Um analista testou árvores de decisão para prever o segmento de compra:

| max_depth | Acurácia treino | Acurácia teste |
|---|---|---|
| 2 | 0,70 | 0,68 |
| 5 | 0,88 | 0,84 |
| 12 | 1,00 | 0,71 |

(a) Diagnostique cada linha (under/overfitting/bom ajuste). (b) Qual profundidade escolher? (c) Por que uma validação cruzada de 5 folds é mais confiável que um único hold-out, e o que a média e o desvio dos scores dizem?

---

## BLOCO 4 — Não supervisionado, recomendação, produção

**Q23.** Uma montadora quer agrupar suas concessionárias em perfis de venda. A inércia do K-means para K = 1…6 foi `[900, 400, 150, 120, 105, 95]`. (a) Calcule a queda entre valores consecutivos. (b) Onde está o cotovelo? (c) Por que não escolher o K com menor inércia? (d) Como a silhueta complementa (interpretação de valores próximos de +1, 0 e negativos)?

**Q24.** Uma rede de academias quer segmentar alunos em perfis de comportamento sem usar rótulos. Explique passo a passo o algoritmo K-means (inicialização, atribuição, atualização, parada). Por que padronizar as variáveis e fixar `random_state`? O K-means é supervisionado ou não? Quem dá sentido aos grupos?

**Q25.** Um serviço de streaming monta a matriz usuário × filme (Duna, Matrix, Blade Runner, Titanic, Notebook, ET, Interestelar): Ana = [5,5,0,1,1,0,5]; Bruno = [0,5,5,0,0,4,4]; Carla = [1,0,0,5,5,0,0] (0 = não viu). (a) Calcule o cosseno Ana×Bruno e Ana×Carla. (b) Usando filtragem user-based com K = 1, o que recomendar à Ana e em que ordem? (c) Diferencie user-based de item-based e diga qual sofre mais com *cold start*.

**Q26.** Usando a matriz de confusão da Q17 (inspeção de pistões: TP=30, FP=10, FN=20, TN=140): (a) calcule TPR e FPR, que formam um ponto da curva ROC. (b) O que significam AUC = 0,5 e AUC = 1,0? (c) Por que a curva usa `predict_proba` e não `predict`? (d) Ao aplicar PCA nas variáveis do projeto, por que padronizar antes, e o que `explained_variance_ratio_ = [0.62, 0.24]` diz? O que se perde?

**Q27.** Um modelo de detecção de fraude tem 9.400 exemplos da classe 0 (legítima) e 600 da classe 1 (fraude). (a) Qual o tamanho final com `RandomOverSampler`? E com `RandomUnderSampler`? (b) Um risco de cada. (c) Um colega aplicou o oversampling **antes** de separar treino e teste. O que está errado e como isso infla o resultado?

**Q28.** Uma equipe de ML ajusta um KNN para o projeto de vendas. (a) Diferencie parâmetro de hiperparâmetro (dois exemplos de cada). (b) `GridSearchCV` com `{'n_neighbors':[3,5,7,9], 'weights':['uniform','distance']}` e `cv=5`: quantos treinamentos? (c) `RandomizedSearchCV` com `n_iter=20` e `cv=5`: quantos? (d) Com 6 hiperparâmetros de 5 valores cada, quantas combinações tem a grade completa e qual busca usar?

**Q29.** Um modelo de previsão de demanda foi para produção com os erros abaixo. Encontre-os:
```python
# treino
scaler = StandardScaler(); X_tr = scaler.fit_transform(X_train)
modelo = RandomForestClassifier().fit(X_tr, y_train)
joblib.dump(modelo, "modelo.joblib")           # só o modelo

# produção
novo = pd.DataFrame([{"idade": 30, "cidade": "SP"}])
X_new = scaler.fit_transform(novo)             # ??
modelo.predict(X_new)
```
Corrija usando `ColumnTransformer` + `Pipeline` e `joblib`. Diferencie pickle de joblib.

**Q30 (integradora).** Uma seguradora usa PyCaret (`setup`, `compare_models`, `tune_model`) para prever cancelamento e o modelo campeão tem 0,94 de acurácia. O time quer subir para produção. (a) Que cuidados (automation bias, métrica, custo do erro, desbalanceamento, validação)? (b) Para o cliente X, o SHAP tem valor base 0,30 e contribuições +0,25 (reclamações), −0,10 (tempo de casa), +0,05 (plano). Qual a probabilidade prevista? Qual variável mais pesou? Como isso difere da importância global? (c) Que papel o MLflow cumpre ao treinar várias versões (run, parâmetro, métrica, artefato)?

---
---

# RESPOSTAS-MODELO

**Q1.** `<class 'float'> 3 1 34`. `/` sempre devolve float; `//` é divisão inteira; `%` é resto; `"3"+"4"` concatena strings.

**Q2.** Iterações: (c=10, s=10) → (c=7, s=17) → (c=4, s=21) → (c=1, s=22) → c vira −2 e o laço termina. Saída: `22 -2`.

**Q3.** Após `append(7)` e `insert(0,10)`: `[10, 8, 6, 9, 5, 7]`. Saída: `[10, 8, 6, 9, 5, 7] [8, 6, 9] [5, 7] 7.5 10` (soma 45 / 6 = 7,5).

**Q4.** `d["c"] = 1 + 2 = 3`; `d["a"]` vira 6. Saída: `{'a': 6, 'b': [2, 3], 'c': 3} ['a', 'b', 'c'] 3`.

**Q5.** (a) `['z'] ['w', 'x', 'y', 'z'] ['x', 'y'] 4`. `&` = montadoras atendidas pelas **duas** fábricas (interseção); `|` = total de montadoras **distintas** (união); `-` = atendidas só pela fábrica A (diferença). (b) `(1, 2, 3, 4) 4 (2, 3)`. `t[0] = 9` gera `TypeError`: tuplas são **imutáveis**; a soma cria uma tupla nova.

**Q6.** `20 inválido 6`.

**Q7.** `[1, 2, 3, 1, 2, 3] [2 4 6] [2 3] 2.0 (3,)`. Lista `*2` **repete** a sequência; array `*2` multiplica **cada elemento** (vetorização). A máscara `a > 1` seleciona elementos.

**Q8.** (a) Linhas 2 e 4 (vendas > 15 e desc ≤ 0,1): `SUV, 30` e `Sedan, 50`. (b) `SUV: total 40, med 20.0`; `Sedan: total 70, med 35.0`; `Pickup: total 40, med 40.0`. Parênteses: `&` tem precedência maior que `>`/`<=`; sem eles dá erro.

**Q9.** (a) Média = 24; mediana = 5. (b) **Mediana**: robusta a extremos; a média (24) não representa nenhum valor típico. (c) Não. Outlier é pista, não veredito: verificar origem/unidade/digitação; pode ser erro, evento raro ou sinal valioso.

**Q10.** IQR = 10,75 − 6,25 = 4,5. Limite inferior = 6,25 − 1,5·4,5 = −0,5; superior = 10,75 + 1,5·4,5 = 17,5. Outlier: **60**. O critério **sinaliza**; a decisão (manter, limitar, transformar, remover) depende do contexto.

**Q11.** (a) z = (60 − 40)/14,14 ≈ **1,41**. (b) (30 − 20)/(60 − 20) = **0,25**. (c) **Min-Max**: usa mín e máx, então um extremo comprime todo o resto. (d) Não: só muda a unidade; evita que uma variável em escala grande domine métodos sensíveis à escala (KNN, K-means, PCA).

**Q12.** `segmento` (nominal): imputar (mais frequente) + **One-Hot**. `nivel_juros` (ordinal): codificação ordinal (baixo=0 < médio=1 < alto=2). `renda_media`: imputar (mediana, se há outliers) + escalar. `comprou`: é o alvo y. 1,2,3 em `segmento` inventaria ordem/distância que não existe. `LabelEncoder` foi feito para o **alvo y**, não para colunas de X.

**Q13.** (a) ETL transforma **antes** de carregar; ELT carrega e o destino transforma. (b) Lake: dados brutos e variados, schema depois, exploração/ciência de dados, exige governança. Warehouse: dados estruturados e curados, schema definido, BI recorrente. (c) Combinação: **lake** (bruto, ERP+sensores+planilhas) para ciência de dados e **warehouse** para BI recorrente; ELT no lake (flexibilidade) e ETL curado até o warehouse. Depende de governança, latência, custo, capacidade.

**Q14.** Erro: o `StandardScaler` foi ajustado (`fit`) com **todos** os dados, inclusive o futuro teste, antes do split: a média/desvio do teste vazam para o treino (*data leakage*) e o desempenho medido fica otimista. Correto: `train_test_split` → `scaler.fit_transform(X_tr)` → `scaler.transform(X_te)` (ou tudo dentro de um `Pipeline`). Regra: `fit` só no treino; `transform` em todo o resto.

**Q15.** 1 Compreensão do negócio (definir objetivo e critério de sucesso, ex.: reduzir devoluções em X%). 2 Compreensão dos dados (coletar, descrever, avaliar qualidade). 3 Preparação (limpar, integrar, criar features — ~**80%** do esforço). 4 Modelagem (escolher técnicas, treinar, comparar). 5 Avaliação (verificar contra o objetivo de negócio). 6 Implantação (deploy, monitoramento, relatório). Não é cascata rígida: o resultado de uma fase determina a próxima (ex.: modelagem revela problema nos dados → volta à preparação); seguido de forma iterativa é ágil.

**Q16.** y = `cancelou_no_mes`. X = `horas_assistidas_mes_anterior`, `n_reclamacoes_ultimos_3m`, `plano` (existem no dia 1). Excluir `id_assinante` (identificador, sem poder preditivo/generalização). **Leakage: `motivo_cancelamento`** — só existe depois do cancelamento (só é preenchido para quem cancelou), então entrega o alvo. Em desenvolvimento o modelo parece perfeito; em produção a variável não existe no momento da decisão (ou vem vazia) e o desempenho despenca. Pergunta-chave: *eu teria este dado no instante da decisão?*

**Q17.** Acurácia = (30+140)/200 = **0,85**. Precisão = 30/(30+10) = **0,75**. Recall = 30/(30+20) = **0,60**. F1 = 2·0,75·0,60/(0,75+0,60) ≈ **0,667**. Especificidade = 140/(140+10) ≈ **0,933**. Defeito não detectado = falso negativo → priorizar **recall**.

**Q18.** (a) Acurácia = 990/1000 = **99%**; recall da classe fraude = 0/10 = **0%**. (b) Paradoxo da acurácia: com classe rara, a acurácia é enganosa; o modelo é inútil. (c) Matriz de confusão, recall, precisão, F1 (e ROC/AUC); rebalancear (oversampling/undersampling). Sempre olhar o recall da classe minoritária.

**Q19.** H(S) = −(0,5·log₂0,5 + 0,5·log₂0,5) = **1,0**. H(Sim: 3/4, 1/4) = −(0,75·log₂0,75 + 0,25·log₂0,25) ≈ **0,811**. H(Não: 2/6, 4/6) ≈ **0,918**. Ganho = 1,0 − (4/10·0,811 + 6/10·0,918) = 1,0 − (0,324 + 0,551) ≈ **0,125**. Sim, reduz a incerteza, mas pouco (ganho baixo). A árvore escolhe o atributo com **maior ganho** em cada nó.

**Q20.** Distâncias de (2,2): (2,1)=1,00 A; (1,1)≈1,41 A; (3,3)≈1,41 B; (4,4)≈2,83 B; (5,4)≈3,61 B. **k=1**: A. **k=3**: vizinhos (2,1)A, (1,1)A, (3,3)B → **A** (2 a 1). **k=5**: todos os pontos → A:2, B:3 → **B**. O resultado muda com k: k pequeno é sensível a ruído; k grande perde detalhe local (chega a votar na classe majoritária). KNN é baseado em distância, logo variáveis em escalas maiores dominam; padronizar (StandardScaler) evita isso.

**Q21.** Erros: 0, 0, −3, 4. MAE = (0+0+3+4)/4 = **1,75**. MSE = (0+0+9+16)/4 = **6,25**. RMSE = **2,5**. R² = 1 − SSE/SST = 1 − 25/500 = **0,95** (média = 35; SST = 225+25+25+225). RMSE > MAE porque eleva erros ao quadrado, dando **mais peso a erros grandes**: desejável quando erros grandes são muito mais caros; indesejável se outliers de erro não são relevantes. Um modelo que sempre prevê a média tem **R² = 0**.

**Q22.** (a) depth 2: **underfitting** (baixo no treino e no teste). depth 5: **bom ajuste** (bom em ambos, gap pequeno). depth 12: **overfitting** (treino 1,00, teste 0,71). (b) **5**. (c) Hold-out depende de qual parte caiu em cada conjunto; K-Fold treina/testa em K partições e tira a média, então todo exemplo é testado. A **média** resume o desempenho esperado; o **desvio** mostra a estabilidade entre folds (custo computacional K vezes maior).

**Q23.** (a) Quedas: 500, 250, 30, 15, 10. (b) **K = 3**: depois dele o ganho passa a ser marginal. (c) A inércia sempre cai com K; K = n zeraria tudo (cada ponto é um cluster) — escolher o cotovelo (equilíbrio ganho × simplicidade). (d) Silhueta: perto de **+1** = bem agrupado e separado; ~**0** = na fronteira; **< 0** = provável ponto no grupo errado. Confirme com silhueta e sentido de negócio.

**Q24.** (1) escolher K e iniciar centróides; (2) atribuir cada ponto ao centróide mais próximo; (3) recalcular centróides (média do grupo); (4) repetir 2–3 até os centróides quase não mudarem. Padronizar: a distância é sensível à escala. `random_state`: a inicialização é aleatória; fixar dá reprodutibilidade. É **não supervisionado** (só X, sem rótulo); o **sentido dos grupos vem da análise humana**.

**Q25.** (a) Ana·Bruno = 25 + 20 = 45; |Ana| = √77 ≈ 8,77; |Bruno| = √82 ≈ 9,06 → cos ≈ **0,57**. Ana·Carla = 5 + 5 + 5 = 15; |Carla| = √51 ≈ 7,14 → cos ≈ **0,24**. (b) O vizinho mais próximo é o **Bruno**. Ana não viu Blade Runner (Bruno deu 5) nem ET (Bruno deu 4): recomendar **Blade Runner**, depois **ET**. (c) User-based: similaridade entre **usuários** (o que vizinhos gostaram); item-based: similaridade entre **itens** (parecido com o que você gostou), mais estável e escalável. O cold start (usuário/item novo sem histórico) atinge a **filtragem colaborativa** (ambas); o content-based/híbrido resolve.

**Q26.** (a) TPR = recall = 30/50 = **0,60**; FPR = 10/150 ≈ **0,067**. (b) AUC 0,5 = aleatório; 1,0 = separação perfeita. (c) A ROC varia o limiar sobre um **score contínuo** (probabilidade da classe positiva); `predict` já devolve a classe (um ponto só). (d) O PCA é sensível à escala (variância domina), então padroniza-se antes. [0,62; 0,24] → 2 componentes explicam **86%** da variância. Perde-se parte da informação (14%) e a interpretabilidade das colunas originais (cada componente é combinação de variáveis).

**Q27.** (a) Over: 9.400 + 9.400 = **18.800**. Under: 600 + 600 = **1.200**. (b) Over: overfitting em exemplos repetidos. Under: perde informação relevante da classe majoritária. (c) Aplicar antes do split faz **cópias do mesmo exemplo caírem em treino e teste**: o modelo é testado em dados que já viu (vazamento) e a métrica fica inflada. Correto: separar primeiro e rebalancear **só o treino**.

**Q28.** (a) **Parâmetro**: aprendido no treino (pesos, vieses, coeficientes). **Hiperparâmetro**: definido por você antes (k do KNN, `max_depth`, taxa de aprendizado, nº de árvores, C do SVM). (b) 4 × 2 = 8 combinações × 5 folds = **40** treinos. (c) 20 × 5 = **100** treinos. (d) 5⁶ = **15.625** combinações (× folds): inviável → **Random Search** (amostra n_iter combinações e costuma achar solução quase tão boa).

**Q29.** Erros: (1) só o modelo foi salvo (o pré-processamento não vai junto); (2) em produção usou `fit_transform` (reaprende média/desvio do único registro; deveria ser só `transform`); (3) `cidade` (categórica) não foi tratada. Correção:
```python
ct = ColumnTransformer([("num", StandardScaler(), ["idade"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["cidade"])])
pipe = Pipeline([("prep", ct), ("modelo", RandomForestClassifier())])
pipe.fit(X_train, y_train)
joblib.dump(pipe, "pipeline_completo.joblib")
# produção
pipe = joblib.load("pipeline_completo.joblib")
pipe.predict(novo)      # dados brutos; o pipeline cuida do resto
```
`pickle`: biblioteca padrão, serve para qualquer objeto, mais lento com muitos arrays NumPy. `joblib`: otimizado para arrays NumPy (modelos), recomendado pelo scikit-learn. Ambos podem não abrir em outra versão de Python/biblioteca.

**Q30.** (a) O AutoML **acelera**, não decide: compara métricas, não o contexto; não sabe o custo do erro nem se a métrica é a certa. *Automation bias* = confiar demais só porque veio de sistema automatizado. Verificar: métrica adequada (precisão × recall, F1), desbalanceamento (0,94 pode ser enganoso), validação cruzada/overfitting, explicabilidade, pipeline salvo junto com o modelo, teste com dados novos fora do notebook, monitoramento. (b) Predição = 0,30 + 0,25 − 0,10 + 0,05 = **0,50**. Mais peso: **reclamações** (+0,25, empurra para cima). SHAP explica **cada previsão individual** (valor base + contribuições = previsão); a importância global só diz o peso médio das variáveis no modelo. (c) MLflow (Tracking) registra **parâmetros** (max_depth), **métricas** (acurácia, F1), **artefatos** (modelo, gráficos) e tags de cada **run**, agrupadas em **experimentos**; permite comparar e reproduzir versões e saber qual configuração deu qual resultado (Model Registry versiona e promove modelos).
