# Controle de temperatura de um CSTR: PID vs. IA

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guilhermemachadogm26-sys/controle-pid/blob/main/cstr_pid_vs_ia.ipynb)

Projeto de estudo que compara estratégias de controle de temperatura em um **reator CSTR não-isotérmico** (reação exotérmica irreversível A → B). É um sistema fortemente não-linear e com múltiplos estados estacionários, o que o torna um bom caso para confrontar o controle clássico com técnicas de IA.

## O que é comparado

1. **PID** discreto com anti-windup e filtro na derivada (referência).
2. **Agente neural (CEM)**: política MLP (4 → 8 → 1) treinada pelo *Cross-Entropy Method*, sem backpropagation.
3. **PID + Supervisor**: um supervisor que atua por meio de "ferramentas" (mudar setpoint, reajustar ganhos, override do resfriamento, sinalizar falha de sensor). Hoje ele é baseado em regras, com a mesma interface pensada para um supervisor com LLM.
4. **PID + Chronos**: o modelo de previsão de séries temporais [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-small) (Amazon) prevê a temperatura e o PID age sobre uma mistura da medida atual com a prevista.
5. **PID sintonizado por PSO** (*Particle Swarm Optimization*): uma sintonia no modelo nominal e uma sintonia **robusta**, feita sobre plantas com erro de modelo e ruído de medição (pasta [`pso/`](pso/)).

## Estrutura do código

Todos os módulos são gerados pelas células do notebook `cstr_pid_vs_ia.ipynb`:

| Arquivo | Função |
|---|---|
| `cstr_model.py` | Modelo do CSTR (balanços de massa e energia, integração RK4) |
| `cstr_env.py` | Ambiente no estilo Gymnasium (observação, ação, recompensa) |
| `pid_controller.py` | PID discreto com anti-windup |
| `policy.py` | Política MLP em NumPy puro |
| `train_cem.py` | Treinamento do agente pelo Cross-Entropy Method |
| `compare.py` | PID vs. agente em diferentes setpoints |
| `plant_with_faults.py` | Injeção de perturbações e falhas na planta |
| `supervisor_tools.py` | Ferramentas (function calling) do supervisor |
| `mock_supervisor.py` | Supervisor baseado em regras |
| `run_experiment.py` | PID sozinho vs. PID + supervisor nos cenários de falha |
| `chronos_control.py` | PID vs. agente vs. PID + Chronos |

O estudo de sintonia por PSO fica em um notebook próprio, [`pso/pso_sintonia_pid.ipynb`](pso/pso_sintonia_pid.ipynb):

| Arquivo | Função |
|---|---|
| `batch_sim.py` | Simulação vetorizada do CSTR + PID (centenas de simulações em paralelo, idêntica à original) |
| `pso_tune_pid.py` | PSO no modelo nominal (Kp, Ki, Kd) |
| `robustez_pid_pso.py` | Sensibilidade a erros de modelo e Monte Carlo com ruído |
| `pso_robusto.py` | PSO robusto (Kp, Ki, Kd e filtro da derivada) e avaliação em 200 plantas novas |

## Como rodar

Abra o notebook no Colab (botão acima) e execute todas as células em ordem. Dependências: `numpy`, `matplotlib`, `gymnasium`, `torch` e `chronos-forecasting`.

O notebook do PSO é independente: [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guilhermemachadogm26-sys/controle-pid/blob/main/pso/pso_sintonia_pid.ipynb). Ele só precisa de `numpy` e `matplotlib`, não precisa de GPU e roda em cerca de 4 minutos.

## Resultados

Os gráficos completos estão no notebook.

### 1. PID vs. agente CEM (rastreamento de setpoint)

| Setpoint (K) | Controlador | ISE | IAE | Overshoot (K) | Acomodação (min) |
|---|---|---|---|---|---|
| 345 | PID | 145,0 | 28,4 | 4,8 | 1,05 |
| 345 | Agente CEM | 143,0 | 45,3 | 3,7 | 0,25 |
| 360 | PID | 3080,2 | 129,2 | 60,0 | 7,50 |
| 360 | Agente CEM | 1141,7 | 61,2 | 68,4 | 2,85 |
| 375 | PID | 332,2 | 39,9 | 9,9 | 3,00 |
| 375 | Agente CEM | 1181,4 | 38,1 | 77,1 | 1,15 |

### 2. PID sozinho vs. PID + supervisor (falha em t = 8 min)

| Cenário | Modo | ISE | IAE | Erro máx. (K) |
|---|---|---|---|---|
| Degrau na temperatura de alimentação | PID sozinho | 110,2 | 25,6 | 11,8 |
| | PID + supervisor | 110,2 | 25,6 | 11,8 |
| Degrau na concentração de alimentação | PID sozinho | 1,06 × 10⁹ | 89 740 | 31 010 |
| | PID + supervisor | 24 718 | 547,1 | 131,2 |
| Setpoint agressivo (saturação) | PID sozinho | 2584,8 | 139,9 | 68,1 |
| | PID + supervisor | 2584,8 | 139,9 | 68,1 |
| Falha de sensor (viés de +10 K) | PID sozinho | 7534,0 | 334,0 | 78,2 |
| | PID + supervisor | 7836,8 | 340,4 | 91,2 |

### 3. PID + Chronos (degrau na temperatura de alimentação)

Peso da previsão escolhido por varredura: α = 0,05 (0 = PID puro).

| Métrica | PID | Agente CEM | PID + Chronos |
|---|---|---|---|
| ISE | 110,2 | 3463,8 | 110,9 |
| IAE | 25,6 | 281,1 | 25,8 |
| IAE pós-falha | 4,38 | 153,1 | 4,41 |
| Erro máx. (K) | 11,8 | 17,4 | 11,8 |
| Esforço Σ\|ΔTc\| (K) | 88,4 | 47,4 | 105,7 |

Qualidade da previsão do Chronos: RMSE de 0,27 K a 0,5 min à frente (contra 0,62 K de uma previsão por persistência) e 98 % dos valores reais dentro do intervalo de 10–90 %.

### 4. Sintonia do PID por PSO: nominal vs. robusta

**Etapa 1, PSO nominal.** O PSO buscou Kp, Ki e Kd minimizando IAE + esforço + overshoot² no modelo nominal, sem ruído. O resultado parecia excelente: em 360 K o IAE caiu de 129 para 18 e o overshoot de 60 para 5 K, sem piorar os outros setpoints.

**Etapa 2, teste de robustez.** Com erros de modelo (UA, k0 ±20 %; ΔHr, q ±10 %), a melhoria **some**: basta um erro de ~5 % para os dois PIDs alternarem entre acomodar bem e oscilar. E um ruído de medição de apenas **0,5 K** faz os dois oscilarem ±40 K, com o atuador batendo de 280 a 350 K. A causa é a derivada, que amplifica o ruído. Ela não pode ser removida (com Kd = 0 o reator fica instável), e filtrá-la mais piora o desempenho sem ruído.

**Etapa 3, PSO robusto.** O filtro da derivada virou a 4ª variável, e o custo passou a ser a média sobre **30 plantas incertas com ruído**. A avaliação foi feita em **200 plantas novas**, que o PSO não viu.

Plantas incertas + ruído de 0,5 K (IAE mediano):

| Caso | PID de referência | PSO nominal | PSO robusto | PSO robusto melhor em |
|---|---|---|---|---|
| Setpoint 345 K | 24,9 | 26,4 | 28,4 | 46 % das plantas |
| Setpoint 360 K | 213,6 | 230,5 | **24,1** | 84 % |
| Setpoint 375 K | 164,2 | 171,7 | **74,6** | 98 % |
| Falha em Tf (t = 8 min) | 393,3 | 407,8 | **36,1** | 98 % |

Ganhos do PSO robusto: Kp = 5,2 × 10⁻⁵, Ki = 0,047, Kd = 0,129 e filtro da derivada = 0,038 (contra 0,1 no PID de referência).

![Respostas com planta incerta e ruído](pso/pso_robusto_respostas.png)

![Monte Carlo em 200 plantas novas](pso/pso_robusto_montecarlo.png)

O preço da robustez: **sem ruído**, o PSO robusto fica um pouco pior (IAE 32 contra 27 na falha em Tf), e na planta nominal o overshoot em 375 K sobe de 10 para 49 K. Com a falha de concentração, nenhuma sintonia de PID evita o disparo térmico, porque o atuador satura.

## Principais conclusões

- **Agente CEM:** acomoda mais rápido que o PID em todos os setpoints, mas com overshoot maior em 360 e 375 K. No cenário com perturbação, que não fez parte do treino, fica bem pior que o PID (IAE 281 contra 26), o que mostra generalização limitada.
- **Supervisor:** no degrau de concentração o PID sozinho perde o controle do reator (disparo térmico), e o supervisor evita isso (IAE de 89 740 para 547). Nos outros cenários ele não ajudou, e na falha de sensor piorou um pouco: as regras ainda são simples.
- **Chronos:** prevê bem a temperatura, mas usar a previsão dentro do PID não trouxe ganho. Com α = 0,05 o resultado é praticamente igual ao do PID, e com α ≥ 0,3 a malha fica instável.
- **PSO:** sintonizar só no modelo nominal gera sobreajuste, com ganhos que parecem ótimos e não sobrevivem a erros de modelo de 5 %. O problema real do PID de referência é o **ruído amplificado pela derivada**. Otimizando o filtro junto com os ganhos, sobre plantas incertas, o PID fica estável na condição realista, com uma pequena perda no caso ideal.
- **Atenção:** os resultados das seções 1 a 3 foram obtidos **sem ruído de medição e sem erro de modelo**. A seção 4 mostra que isso pode mudar bastante a comparação.

## Próximos passos

- Trocar o supervisor por regras por um supervisor com LLM.
- Treinar o agente também com perturbações e falhas.
- Comparar com controle preditivo (MPC).
- Reavaliar o agente CEM e o PID + Chronos com plantas incertas e ruído, contra o PID robusto.
