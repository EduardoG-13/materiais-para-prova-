# Matemática — 30 questões contextualizadas (fácil → difícil)

**Base:** slides do Geraldo (Sem 02–08), resumos em PDF, provas antigas e lista de revisão da Lívia (monitoria). Como nas provas do Inteli, todas as questões têm contexto de tecnologia (modelos de IA, MLOps, nuvem) e do projeto de previsão de vendas de veículos (MAHLE). Números novos, sem repetir os da monitoria.

**Formato da prova:** questões de **soma** (afirmações 01, 02, 04; responda a soma das corretas, de 0 a 7) e **múltipla escolha**. Sem calculadora. Tabela normal: `Materiais Docentes.../Matemática/Tabela Normal Padrao.pdf`.

**Dificuldade:** Nível 1 (Q1–Q10) conceito e conta de 1 passo. Nível 2 (Q11–Q20) 2–3 passos. Nível 3 (Q21–Q30) nível de prova / multi-etapas.

Gabarito com resolução no final. Faça sem olhar.

---

## NÍVEL 1 — Base

**Q1 (soma).** A equipe do projeto MAHLE descreve as saídas dos seus modelos preditivos de vendas mensais de veículos. Julgue as afirmações sobre variáveis aleatórias:
- (01) A densidade f(x) de uma variável contínua, como o tempo de resposta do modelo, nunca pode ser maior que 1.
- (02) Em uma variável aleatória contínua, P(X = c) = 0 para qualquer valor c.
- (04) A área total sob a PDF vale 1 apenas para a distribuição normal.

**Q2.** O custo de inferência de um modelo em nuvem, em unidades de processamento, é C(x, y) = 3x² − 2xy + y, em que x é o volume de requisições (em milhares) e y é a memória usada (em GB). Calcule C(2, 3).
(a) 3  (b) 9  (c) 27  (d) −3  (e) 15

**Q3 (soma).** O módulo de validação de uma IA de logística rejeita entradas inválidas antes de passá-las à predição. Sobre domínio e curvas de nível:
- (01) O domínio de f(x, y) = x²y − 4y³ é todo o plano ℝ².
- (02) Para que √(x − y) seja real, é preciso x − y ≥ 0.
- (04) A curva de nível k de f é o conjunto dos pontos (x, y) tais que f(x, y) = k.

**Q4.** Um classificador de defeitos em pistões tem função de perda L(x, y) = 4x³y² − 7x + 2y, em que x e y são dois pesos do modelo. Na atualização por gradiente, a derivada parcial ∂L/∂x é:
(a) 12x²y² − 7  (b) 12x²y² − 5  (c) 12x²y − 7  (d) 8x³y − 7  (e) 12x²y² − 7x

**Q5.** Um lote de inspeção usa 4 classificadores independentes em paralelo, e cada um gera um falso positivo com probabilidade 1/2. Qual a probabilidade de exatamente 2 deles gerarem falso positivo?
(a) 0,250  (b) 0,375  (c) 0,500  (d) 0,125  (e) 0,750

**Q6.** O tempo de inferência de um modelo de previsão de demanda segue uma Uniforme Contínua em [0, 40] ms. O painel de MLOps dispara alerta de lentidão acima de 30 ms. Qual é P(X > 30)?
(a) 10%  (b) 25%  (c) 30%  (d) 75%  (e) 33%

**Q7.** A latência de um serviço de inferência tem média 200 ms e desvio-padrão 25 ms. Na padronização de features (Escore Z), qual o escore de uma requisição de 250 ms?
(a) 0,5  (b) 2  (c) 50  (d) −2  (e) 1,25

**Q8 (soma).** Um pipeline de monitoramento (MLOps) avalia a latência do modelo em produção por médias amostrais. Sobre o TLC e o erro-padrão:
- (01) O erro-padrão da média é σ/√n.
- (02) Quadruplicar o tamanho da amostra reduz o erro-padrão à metade.
- (04) O TLC afirma que a população se torna normal quando a amostra é grande.

**Q9.** A carga de transações de um servidor de recomendação é modelada por R(x, y) = 4x sobre a região retangular x ∈ [0, 3] e y ∈ [0, 2]. Calcule o volume total ∬ R dA.
(a) 12  (b) 24  (c) 36  (d) 48  (e) 72

**Q10.** Na engenharia de atributos do projeto MAHLE, dois indicadores econômicos são projetados por T(x, y) = (3x − y, x + 2y) (transformação de pré-processamento). Qual é o determinante da matriz padrão de T?
(a) 5  (b) 6  (c) 7  (d) −7  (e) 1

---

## NÍVEL 2 — Intermediário

**Q11 (soma).** Uma equipe implementa etapas de pré-processamento de um pipeline de IA como transformações de vetores. Sobre transformações lineares:
- (01) Se T é linear, então T(0, 0) = (0, 0).
- (02) T(x, y) = (x + 1, y) é linear.
- (04) Na matriz padrão, as imagens T(e₁) e T(e₂) formam as **linhas** da matriz.

**Q12.** Um modelo de manutenção preditiva estima que a vida útil restante de uma frota de motores segue uma Normal com média 100 h e desvio-padrão 20 h. Usando a tabela, qual é a probabilidade de a vida útil ser menor que 130 h?
(a) 84,13%  (b) 93,32%  (c) 97,72%  (d) 6,68%  (e) 50,00%

**Q13.** Um job de treinamento de IA é executado 5 vezes de forma independente, e cada execução falha com probabilidade 0,2. Qual a probabilidade de **pelo menos uma** execução falhar?
(a) 0,2000  (b) 0,3277  (c) 0,4096  (d) 0,6723  (e) 1,0000

**Q14.** Um analista de nuvem quer estimar o consumo médio de memória com margem de erro máxima E = 5 GB. O desvio-padrão populacional é σ = 25 GB e a confiança é de 95% (Z = 1,96). O tamanho mínimo da amostra é:
(a) 25  (b) 49  (c) 96  (d) 97  (e) 385

**Q15.** Um teste com n = 100 requisições mediu latência média de 200 ms, com σ = 30 ms (conhecido). O intervalo de confiança de 95% (Z = 1,96) para a latência média é:
(a) [141,20; 258,80]  (b) [197,06; 202,94]  (c) [194,12; 205,88]  (d) [190,00; 210,00]  (e) [196,08; 203,92]

**Q16 (soma).** A equipe de engenharia faz um Teste A/B para saber se o novo modelo de previsão de demanda reduz o tempo de processamento. Sobre o teste de hipóteses:
- (01) O erro Tipo I consiste em não rejeitar H₀ quando ela é falsa.
- (02) A hipótese alternativa Hₐ pode conter o sinal de igualdade.
- (04) Se o p-valor é menor que α, rejeita-se H₀.

**Q17.** Por restrição de orçamento de nuvem, uma simulação rodou só n = 25 baterias de avaliação. O desvio-padrão populacional é desconhecido; a amostra teve média de 96 requisições e desvio-padrão amostral s = 15. A meta é μ₀ = 100. Usando a distribuição t, a estatística de teste é:
(a) −4,00  (b) −0,27  (c) −1,33  (d) 1,33  (e) −0,80

**Q18.** Um modelo de risco de queda de servidor usa g(x, y) = ln(4 − x² − y²), em que x é o tráfego normalizado e y é a latência normalizada. As entradas fora do domínio real são rejeitadas. O domínio válido é:
(a) x² + y² ≤ 4  (b) x² + y² < 4  (c) x² + y² > 4  (d) x² + y² ≥ 4  (e) todo o plano ℝ²

**Q19.** Um algoritmo de precificação de nuvem usa C(x, y) = x² + 4y², em que x é processamento e y é armazenamento. A diretoria quer as combinações de custo fixo k = 16. A curva de nível é:
(a) circunferência de raio 4  (b) elipse de semieixos 4 e 2  (c) hipérbole  (d) parábola  (e) par de retas paralelas

**Q20.** No projeto MAHLE, dois indicadores correlacionados (juros e inflação) passam por uma transformação cuja matriz é A = [[4, 1], [2, 3]]. Os autovalores de A são:
(a) 1 e 6  (b) 2 e 5  (c) 3 e 4  (d) −2 e −5  (e) 7 e 10

---

## NÍVEL 3 — Nível de prova

**Q21 (soma).** Um modelo preditivo processa o risco de tráfego numa fronteira triangular D, delimitada pelo eixo y = 0, pela reta x = 3 e pela reta y = 2x, com densidade de carga f(x, y).
- (01) Na ordem dy dx, y varia de 0 a 3 (limites fixos).
- (02) Na ordem dx dy, 0 ≤ y ≤ 6 e y/2 ≤ x ≤ 3.
- (04) A área de D vale 9.

**Q22.** A carga de processamento de um motor de anomalias é f(x, y) = x + y, integrada sobre a região triangular D = {0 ≤ x ≤ 2, 0 ≤ y ≤ x}. O escore agregado ∬ f dA é:
(a) 2  (b) 4  (c) 6  (d) 8  (e) 12

**Q23 (soma).** Uma IA de precificação dinâmica tem custo computacional f(x, y) = x²y³, e os desenvolvedores analisam sua sensibilidade às entradas por derivadas parciais.
- (01) f_x = 2xy³.
- (02) f_y = 2x²y².
- (04) f_xy = f_yx = 6xy².

**Q24.** A produção prevista de um modelo é z = x·y³, em que x é o tráfego e y é a latência, ambos variando no tempo: x(t) = t² e y(t) = t + 1. Pela regra da cadeia, dz/dt em t = 1 vale:
(a) 16  (b) 20  (c) 24  (d) 28  (e) 32

**Q25.** No projeto MAHLE, o lucro previsto (em R$ mil) de uma linha de produção, em função das horas de dois turnos x e y, é L(x, y) = 40x + 24y − 2x² − 3y². O valor máximo de L é:
(a) 152  (b) 200  (c) 248  (d) 296  (e) 344

**Q26.** O volume de transações de alto risco interceptadas por um cluster de IA é determinado integrando a superfície z = 10 − 2x − y sobre a região R = [0, 1] × [0, 2]. O volume total é:
(a) 8  (b) 12  (c) 16  (d) 18  (e) 20

**Q27.** Uma função de densidade modela falhas simultâneas em contêineres: f(x, y) = 2y·e^(2x) sobre R = [0, 1] × [0, 3]. A IA integra essa distribuição sobre R. O resultado é:
(a) 9(e² − 1)/2  (b) 9(e² − 1)  (c) 9e²/2  (d) 3(e² − 1)/2  (e) 9(e − 1)/2

**Q28 (soma).** Uma plataforma de IA analisa o risco de saída de clientes corporativos. A meta de retenção é μ₀ = 50, com desvio populacional σ = 20 (conhecido). O módulo analisou n = 100 clientes e reportou média amostral 55. Teste bilateral com Z crítico = ±1,96.
- (01) O denominador da estatística Z vale 20.
- (02) Z_teste = 2,5.
- (04) H₀ é rejeitada.

**Q29.** Um pipeline propaga dados por 3 etapas de tempo com a matriz A = [[2, 1], [0, 3]]. Usando A = PDP⁻¹, o elemento da linha 1, coluna 2 de A³ é:
(a) 8  (b) 9  (c) 13  (d) 19  (e) 27

**Q30 (soma).** Na fase de pré-processamento, a IA calcula a matriz de covariância de dois atributos correlacionados, custo e receita: A = [[6, 2], [2, 3]].
- (01) Os autovalores de A são 7 e 2.
- (02) v = (1, −2) é autovetor associado a λ = 2.
- (04) Os autovalores de A² são 14 e 4.

---
---

# GABARITO COMENTADO

| Q | R | Q | R | Q | R |
|---|---|---|---|---|---|
| 1 | **2** | 11 | **1** | 21 | **6** |
| 2 | **a** | 12 | **b** | 22 | **b** |
| 3 | **7** | 13 | **d** | 23 | **5** |
| 4 | **a** | 14 | **d** | 24 | **d** |
| 5 | **b** | 15 | **c** | 25 | **c** |
| 6 | **b** | 16 | **4** | 26 | **c** |
| 7 | **b** | 17 | **c** | 27 | **a** |
| 8 | **3** | 18 | **b** | 28 | **6** |
| 9 | **c** | 19 | **b** | 29 | **d** |
| 10 | **c** | 20 | **b** | 30 | **3** |

**Q1 = 2.** (01) F: a densidade pode passar de 1 (ex.: uniforme em [0; 0,5] tem altura 2). (02) V. (04) F: vale 1 para qualquer PDF. → 2.

**Q2.** 3·4 − 2·2·3 + 3 = 12 − 12 + 3 = 3. (27 vem de errar o sinal do termo cruzado.)

**Q3 = 7.** Polinômio → ℝ²; raiz par → radicando ≥ 0; curva de nível é f = k. Tudo V.

**Q4.** Deriva em x, y é constante: 12x²y² − 7. O termo 2y some.

**Q5.** C(4,2)·(1/2)⁴ = 6/16 = 0,375.

**Q6.** Comprimento 10 / total 40 = 25%.

**Q7.** z = (250 − 200)/25 = 2.

**Q8 = 3.** (01) V. (02) V: √4 = 2, o erro cai pela metade. (04) F: o TLC fala da distribuição das **médias**, não da população. → 1 + 2 = 3.

**Q9.** ∫₀³ 4x dx = 18; vezes a largura em y (2) = 36.

**Q10.** A = [[3, −1], [1, 2]]. det = 3·2 − (−1)(1) = 7. (5 vem de errar o sinal.)

**Q11 = 1.** (01) V. (02) F: T(0,0) = (1,0) ≠ (0,0). (04) F: são as **colunas**. → 1.

**Q12.** z = (130 − 100)/20 = 1,5 → tabela 0,9332.

**Q13.** 1 − P(0 falhas) = 1 − 0,8⁵ = 1 − 0,32768 = 0,6723.

**Q14.** n = (1,96·25/5)² = 9,8² = 96,04 → arredonda **para cima** = 97.

**Q15.** EP = 30/√100 = 3; E = 1,96·3 = 5,88 → [194,12; 205,88]. (a) usa σ sem dividir por √n; (e) usa EP = 2.

**Q16 = 4.** (01) F: isso é erro Tipo II. (02) F: Hₐ nunca tem igualdade. (04) V.

**Q17.** EP = 15/√25 = 3; t = (96 − 100)/3 = −1,33.

**Q18.** Log exige argumento **estritamente** positivo: 4 − x² − y² > 0 → disco aberto.

**Q19.** x²/16 + y²/4 = 1 → elipse de semieixos 4 e 2.

**Q20.** tr = 7, det = 12 − 2 = 10 → λ² − 7λ + 10 = 0 → λ = 2 e 5.

**Q21 = 6.** (01) F: y vai de 0 a 2x. (02) V: y = 2x ⇒ x = y/2; y máximo é 6 (em x = 3). (04) V: base 3 × altura 6 / 2 = 9. → 2 + 4 = 6.

**Q22.** Interna: ∫₀ˣ (x + y) dy = x² + x²/2 = 3x²/2. Externa: ∫₀² 3x²/2 dx = x³/2 |₀² = 4.

**Q23 = 5.** (01) V. (02) F: f_y = 3x²y². (04) V: f_xy = ∂/∂y(2xy³) = 6xy² e Clairaut vale para polinômios. → 1 + 4 = 5.

**Q24.** ∂z/∂x = y³, dx/dt = 2t, ∂z/∂y = 3xy², dy/dt = 1. Em t = 1: x = 1, y = 2 → 8·2 + 3·1·4·1 = 16 + 12 = 28. Conferindo: z = t²(t+1)³, z′(1) = 2·8 + 3·4 = 28.

**Q25.** L_x = 40 − 4x = 0 → x = 10. L_y = 24 − 6y = 0 → y = 4. L_xx = −4 < 0 e L_yy = −6 < 0 (máximo). L = 400 + 96 − 200 − 48 = 248.

**Q26.** Verificação: mínimo de z em (1, 2) = 10 − 2 − 2 = 6 > 0. Interna em y: 20 − 4x − 2 = 18 − 4x. Externa: ∫₀¹ (18 − 4x) dx = 18 − 2 = 16.

**Q27.** Separável em retângulo: ∫₀¹ e^(2x) dx = (e² − 1)/2 e ∫₀³ 2y dy = 9 → 9(e² − 1)/2.

**Q28 = 6.** EP = 20/√100 = 2. (01) F (o denominador é 2). (02) V: z = 5/2 = 2,5. (04) V: 2,5 > 1,96. → 2 + 4 = 6.

**Q29.** Triangular: λ = 2 e 3. Autovetores: (1,0) e (1,1). P = [[1,1],[0,1]], P⁻¹ = [[1,−1],[0,1]], D³ = diag(8, 27). PD³ = [[8,27],[0,27]]; vezes P⁻¹ = [[8,19],[0,27]]. Elemento (1,2) = 19. Conferindo: A² = [[4,5],[0,9]], A³ = A²·A = [[8,19],[0,27]].

**Q30 = 3.** tr = 9, det = 14 → λ² − 9λ + 14 = 0 → 7 e 2. (01) V. (02) V: A(1,−2) = (6−4, 2−6) = (2, −4) = 2·(1,−2). (04) F: os autovalores de A² são os **quadrados**: 49 e 4. → 1 + 2 = 3.
