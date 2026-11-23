# M3 error analysis (measured)

Router: **lightgbm** (and knn for contrast). Operating point: λ chosen so mean cost ≈ 25% of always-best cost. On the ID split that λ was **39.81**.

Oracle here = route with *true* quality and *true* cost at the same λ.

## Quality gap by source (ID, lightgbm)

```csv
source,split,router,n,frac_wrong,mean_quality_gap,mean_router_quality,mean_oracle_quality,mean_extra_cost
winogrande,id,lightgbm,191,1.0,0.204188,0.795812,1.0,0.000509
mmlu,id,lightgbm,2107,0.942098,0.198861,0.756051,0.954912,0.000586
mbpp,id,lightgbm,65,0.984615,0.169231,0.738462,0.907692,0.005949
hellaswag,id,lightgbm,1507,0.732581,0.165893,0.806238,0.97213,0.000399
mtbench,id,lightgbm,12,0.916667,0.158333,0.775,0.933333,0.008677
misc,id,lightgbm,258,0.899225,0.15407,0.625,0.77907,0.00191
gsm8k,id,lightgbm,1118,0.949911,0.1089,0.632603,0.741503,0.000171
arc_challenge,id,lightgbm,221,0.954751,0.067873,0.923077,0.99095,0.000482
```

## Quality gap by domain (ID, lightgbm)

```csv
domain,split,router,n,frac_wrong,mean_quality_gap,mean_router_quality,mean_oracle_quality,mean_extra_cost
code,id,lightgbm,65,0.984615,0.169231,0.738462,0.907692,0.005949
knowledge,id,lightgbm,4026,0.867114,0.179583,0.785892,0.965474,0.000507
math,id,lightgbm,1118,0.949911,0.1089,0.632603,0.741503,0.000171
other,id,lightgbm,270,0.9,0.154259,0.631667,0.785926,0.00221
```

## Quality gap by prompt-length quartile (ID, lightgbm)

```csv
length_bin,split,router,n,frac_wrong,mean_quality_gap,mean_router_quality,mean_oracle_quality,mean_extra_cost
p0-25,id,lightgbm,1371,0.954778,0.164406,0.770678,0.935084,0.001101
p25-50,id,lightgbm,1369,0.953981,0.174069,0.774799,0.948868,0.000861
p50-75,id,lightgbm,1372,0.844752,0.137828,0.739504,0.877332,0.000311
p75-100,id,lightgbm,1367,0.79444,0.178932,0.700732,0.879663,7.3e-05
```

## Costliest wrong routes (ID, lightgbm): spent more, quality ≤ oracle

Top 15 by extra USD cost:

```csv
prompt_id,source,router_model,oracle_model,router_quality,oracle_quality,cost_gap
mtbench.dev.15,mtbench,gpt-4-1106-preview,zero-one-ai/Yi-34B-Chat,1.0,1.0,0.043333
mtbench.dev.36,mtbench,gpt-4-1106-preview,zero-one-ai/Yi-34B-Chat,1.0,1.0,0.036371
chinese-remainder-theorem.dev.11,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.034973
chinese-remainder-theorem.dev.9,misc,gpt-4-1106-preview,mistralai/mistral-7b-chat,0.0,0.0,0.034457
chinese-remainder-theorem.dev.6,misc,gpt-4-1106-preview,mistralai/mistral-7b-chat,0.0,0.0,0.025543
chinese-remainder-theorem.dev.8,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.024866
chinese-remainder-theorem.dev.4,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.024837
chinese-remainder-theorem.dev.0,misc,gpt-4-1106-preview,gpt-3.5-turbo-1106,1.0,1.0,0.0228
chinese-remainder-theorem.dev.12,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.021498
chinese-remainder-theorem.dev.3,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.021147
chinese-remainder-theorem.dev.7,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.021087
mbpp.dev.161,mbpp,gpt-4-1106-preview,gpt-3.5-turbo-1106,1.0,1.0,0.021057
chinese-remainder-theorem.dev.13,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.019419
chinese-remainder-theorem.dev.14,misc,gpt-4-1106-preview,WizardLM/WizardLM-13B-V1.2,0.0,0.0,0.019317
chinese-remainder-theorem.dev.5,misc,gpt-4-1106-preview,mistralai/mistral-7b-chat,0.0,0.0,0.018481
```
