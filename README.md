
## Introdução

Esse repositório contém códigos do desafio Zindi Africa. Aqui, é
descrito todo o processo para chegar no resultado final. Os códigos
completos do estudo estão no arquivo `CaseAnaliseDados.R`. Os dados,
obtidos de
<https://zindi.africa/competitions/financial-inclusion-in-africa/data>
estão na pasta `data`.

O objetivo é fazer um modelo que identifique quais pessoas possuem conta
no banco, correspondente a variável `bank_account`.

O desafio foi feito na linguagem `R` e as etapas do processo são:

  - Análise exploratória dos dados
  - Balanceamento de variáveis
  - Modelagem
  - Resultados

## Importações

Os pacotes `data.table` e `tidyverse` (que inclui `dplyr`, `tidyr` e
outros) serão utilizados para as operações de *data wrangling*. Para as
visualizações, utilizarei o `ggplot2` (incluído no `tidyverse`) e
`magick`; enquanto que os pacotes `caret`, `pROC` e `xgboost` serão
utilizados na parte de *machine learning*.

Os dados originais estão divididos entre treino e teste, sendo que o
teste não contém a variável resposta.

## Análise Exploratória dos Dados

``` r
names(train)
```

    ##  [1] "country"                "year"                   "uniqueid"              
    ##  [4] "bank_account"           "location_type"          "cellphone_access"      
    ##  [7] "household_size"         "age_of_respondent"      "gender_of_respondent"  
    ## [10] "relationship_with_head" "marital_status"         "education_level"       
    ## [13] "job_type"

![](README_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Há um desbalanço na variável resposta. Isso deve ser levado em
consideração na hora de treinar o modelo, pois datasets desbalanceados
podem influenciar no resultado do modelo.

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Os países apresentam taxas diferentes de pessoas com contas bancárias.

## Modelagem

O primeiro passo a ser feito é rodar um modelo de *machine learning* e
vermos os resultados iniciais obtidos, para então tomarmos a decisão de
que caminho seguir. Como não temos a variável resposta no dataset de
teste, iremos criar um dataset de validação a partir do dataset de
treino, que ficará de fora do treinamento do nosso modelo, para que
possamos avaliar como ele está performando.

O pacote `caret` tem uma função que facilita o processo de dividir o
dataset de maneira distribuida como o dataset original. 75% do dataset
original permaneceu no dataset de treino, e 25% ficou separado no de
validação.

``` r
trainIndex <- createDataPartition(train$bank_account, p = .75, 
                                  list = FALSE, times = 1)

validation <- train[-trainIndex]
train <- train[trainIndex]

y_train <- as.factor(train$bank_account)
y_validation <- as.factor(validation$bank_account)

train <- train[, -'year', with=F]
validation <- validation[, -'year', with=F]
```

Para treinar nosso primeiro modelo, utilizaremos a metodologia *Random
Forest*, modelo não-paramétrico que se adapta bem a diversos tipos de
dataset. Ele utiliza apenas um hiperparâmetro, o `mtry`. Uma regra de
bolso é utilizar a raiz quadrada do número de variáveis do dataset para
o `mtry`. Também utilizaremos o método de *Cross Validation*.

``` r
trcontrol = trainControl( method = "cv",
                          number = 5,  
                          allowParallel = TRUE,
                          verboseIter = TRUE )

mtry <- sqrt(ncol(train[, -'bank_account', with=F]))
tunegrid <- expand.grid(.mtry=mtry)

rf_model <- train(x = train[, -'bank_account', with=F], 
                  y = y_train,
                  trControl = trcontrol,
                  tuneGrid = tunegrid,
                  method = "rf")
```

Depois de rodarmos o modelo, iremos avaliar a sua performance. Para
isso, iremos comparar a acurácia das previsões tanto no treino de teste,
quando no de validação, que separamos apenas para isso. Se a performance
no treino de validação não for muito inferior a de teste, quer dizer que
nosso modelo pode performar bem em dados futuros, não apresentando
*overfit*.

### Matriz de Confusão do treino

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    No   Yes
    ##        No  15060  1081
    ##        Yes    99  1403
    ##                                              
    ##                Accuracy : 0.933              
    ##                  95% CI : (0.929, 0.937)     
    ##     No Information Rate : 0.859              
    ##     P-Value [Acc > NIR] : <0.0000000000000002
    ##                                              
    ##                   Kappa : 0.669              
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.993              
    ##             Specificity : 0.565              
    ##          Pos Pred Value : 0.933              
    ##          Neg Pred Value : 0.934              
    ##              Prevalence : 0.859              
    ##          Detection Rate : 0.854              
    ##    Detection Prevalence : 0.915              
    ##       Balanced Accuracy : 0.779              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

### Matriz de Confusão da validação

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  4917  553
    ##        Yes  136  275
    ##                                               
    ##                Accuracy : 0.883               
    ##                  95% CI : (0.874, 0.891)      
    ##     No Information Rate : 0.859               
    ##     P-Value [Acc > NIR] : 0.0000000522        
    ##                                               
    ##                   Kappa : 0.387               
    ##                                               
    ##  Mcnemar's Test P-Value : < 0.0000000000000002
    ##                                               
    ##             Sensitivity : 0.973               
    ##             Specificity : 0.332               
    ##          Pos Pred Value : 0.899               
    ##          Neg Pred Value : 0.669               
    ##              Prevalence : 0.859               
    ##          Detection Rate : 0.836               
    ##    Detection Prevalence : 0.930               
    ##       Balanced Accuracy : 0.653               
    ##                                               
    ##        'Positive' Class : No                  
    ## 

No dataset de treino, obtivemos uma acurácia de 0.933, e no de validação
0.883. Se fosse só por essa informação, poderíamos dizer que nosso
modelo está performando bem. Porém, pelos valores de especificidade,
vemos que o fato de o dataset estar desbalanceado pode estar impactando
o resultado de nossas previsões.

## Downsamplig

Dessa maneira, utilizaremos o método de *Downsampling* para obter um
dataset que tenha 50% de usuários com conta em banco e 50% de usuários
sem conta em banco. Datasets assim tendem a nos dar modelos que capturem
melhor as nuances de ambos os valores. Para isso, mais uma vez
utilizamos o pacote `caret`.

Apenas uma linha de comando e temos nosso novo dataset, agora
balanceado.

``` r
train_ds <- downSample(train, y_train, yname = 'bank_account')
setDT(train_ds)[, .N, bank_account]
```

    ##    bank_account    N
    ## 1:           No 2484
    ## 2:          Yes 2484

Rodamos novamente o modelo de *Random Forest*, dessa vez no dataset
balanceado o analisamos seus resultados.

``` r
y_ds <- train_ds$bank_account
train_ds <- train_ds[, -'bank_account', with=F]

rf_ds_model <- train(x = train_ds, 
                     y = y_ds,
                     trControl = trcontrol,
                     tuneLength = tunegrid,
                     method = "rf")
```

### Matriz de Confusão do treino balanceado

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  2184  375
    ##        Yes  300 2109
    ##                                              
    ##                Accuracy : 0.864              
    ##                  95% CI : (0.854, 0.874)     
    ##     No Information Rate : 0.5                
    ##     P-Value [Acc > NIR] : <0.0000000000000002
    ##                                              
    ##                   Kappa : 0.728              
    ##                                              
    ##  Mcnemar's Test P-Value : 0.0044             
    ##                                              
    ##             Sensitivity : 0.879              
    ##             Specificity : 0.849              
    ##          Pos Pred Value : 0.853              
    ##          Neg Pred Value : 0.875              
    ##              Prevalence : 0.500              
    ##          Detection Rate : 0.440              
    ##    Detection Prevalence : 0.515              
    ##       Balanced Accuracy : 0.864              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

### Matriz de Confusão do modelo com downsampling aplicado na validação

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  3985  207
    ##        Yes 1068  621
    ##                                              
    ##                Accuracy : 0.783              
    ##                  95% CI : (0.772, 0.794)     
    ##     No Information Rate : 0.859              
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.375              
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.789              
    ##             Specificity : 0.750              
    ##          Pos Pred Value : 0.951              
    ##          Neg Pred Value : 0.368              
    ##              Prevalence : 0.859              
    ##          Detection Rate : 0.678              
    ##    Detection Prevalence : 0.713              
    ##       Balanced Accuracy : 0.769              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

Ainda que tenhamos perdido um pouco de acurácia em relação ao modelo
original, o novo modelo, feito a partir do dataset balanceado com a
tecnica de *downsampling*, nos dá resultados mais equilibrados, que
podemos ver pela Sensibilidade e Especificidade.

Nota: É possível utilizar também o *upsampling*, que balanceia o dataset
com mais observações, copiando observações da variável com menos
entradas. Aqui, utilizei apenas o *downsampling*, mas no script de
código completo, faço a avaliação dos dois métodos.

## Modelagem

Agora, iremos testar outras metodologias, como o simples *GLM* e o
*XGBoost*.

### GLM

``` r
glm_model <- train( x = train_ds, 
                    y = y_ds,
                    trControl = trcontrol,
                    tuneLength = 5,
                    method = "glm")
```

### Matriz de Confusão do GLM no treino balanceado

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  1982  640
    ##        Yes  502 1844
    ##                                               
    ##                Accuracy : 0.77                
    ##                  95% CI : (0.758, 0.782)      
    ##     No Information Rate : 0.5                 
    ##     P-Value [Acc > NIR] : < 0.0000000000000002
    ##                                               
    ##                   Kappa : 0.54                
    ##                                               
    ##  Mcnemar's Test P-Value : 0.0000503           
    ##                                               
    ##             Sensitivity : 0.798               
    ##             Specificity : 0.742               
    ##          Pos Pred Value : 0.756               
    ##          Neg Pred Value : 0.786               
    ##              Prevalence : 0.500               
    ##          Detection Rate : 0.399               
    ##    Detection Prevalence : 0.528               
    ##       Balanced Accuracy : 0.770               
    ##                                               
    ##        'Positive' Class : No                  
    ## 

### Matriz de Confusão do GLM na validação

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  4006  208
    ##        Yes 1047  620
    ##                                              
    ##                Accuracy : 0.787              
    ##                  95% CI : (0.776, 0.797)     
    ##     No Information Rate : 0.859              
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.38               
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.793              
    ##             Specificity : 0.749              
    ##          Pos Pred Value : 0.951              
    ##          Neg Pred Value : 0.372              
    ##              Prevalence : 0.859              
    ##          Detection Rate : 0.681              
    ##    Detection Prevalence : 0.717              
    ##       Balanced Accuracy : 0.771              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

### xGBoost

Para rodarmos o modelo de *xGBoost*, precisamos de algumas
transformações antes. O algoritmo do modelo necessita que cada
variável categórica esteja como numérica, como se fossem *dummies*.
Assim, usamos o processo chamado *One Hot Encoding*. O pacote `caret`
nos ajuda mais uma vez.

``` r
dmy_train_ds <- dummyVars('~.', data = train_ds)
train_matrix_ds <- data.table(predict(dmy_train_ds, newdata = train_ds))
train_matrix_ds <- train_matrix_ds %>% as.matrix() %>% xgb.DMatrix()


dmy_validation <- dummyVars('~.', data = validation[, -'bank_account', with = F])
validation_matrix <- data.table(predict(dmy_validation, newdata = validation[, -'bank_account', with = F]))
validation_matrix <- validation_matrix %>% as.matrix() %>% xgb.DMatrix()
```

``` r
xgb_model <- train( x = train_matrix_ds, 
                    y = y_ds,
                    trControl = trcontrol,
                    tuneLength = 4,
                    method = "xgbTree")
```

### Matriz de Confusão do xGBoost no treino balanceado

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  2028  522
    ##        Yes  456 1962
    ##                                              
    ##                Accuracy : 0.803              
    ##                  95% CI : (0.792, 0.814)     
    ##     No Information Rate : 0.5                
    ##     P-Value [Acc > NIR] : <0.0000000000000002
    ##                                              
    ##                   Kappa : 0.606              
    ##                                              
    ##  Mcnemar's Test P-Value : 0.0377             
    ##                                              
    ##             Sensitivity : 0.816              
    ##             Specificity : 0.790              
    ##          Pos Pred Value : 0.795              
    ##          Neg Pred Value : 0.811              
    ##              Prevalence : 0.500              
    ##          Detection Rate : 0.408              
    ##    Detection Prevalence : 0.513              
    ##       Balanced Accuracy : 0.803              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

### Matriz de Confusão do xGBoost no treino balanceado

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   No  Yes
    ##        No  3992  202
    ##        Yes 1061  626
    ##                                              
    ##                Accuracy : 0.785              
    ##                  95% CI : (0.775, 0.796)     
    ##     No Information Rate : 0.859              
    ##     P-Value [Acc > NIR] : 1                  
    ##                                              
    ##                   Kappa : 0.381              
    ##                                              
    ##  Mcnemar's Test P-Value : <0.0000000000000002
    ##                                              
    ##             Sensitivity : 0.790              
    ##             Specificity : 0.756              
    ##          Pos Pred Value : 0.952              
    ##          Neg Pred Value : 0.371              
    ##              Prevalence : 0.859              
    ##          Detection Rate : 0.679              
    ##    Detection Prevalence : 0.713              
    ##       Balanced Accuracy : 0.773              
    ##                                              
    ##        'Positive' Class : No                 
    ## 

## Conclusão

Os resultados dos 3 modelos foram bem parecidos, dessa maneira, entendo
que temos mais vantagens ao escolhermos o mais simples, no caso o
**GLM**. Isso porque ele nos dá parâmetros, dessa maneira podemos
avaliar o impacto de cada variável na previsão. Datasets desbalanceados
são bastante frequentes em casos de detecção de fraude ou detecção de
câncer, por exemplo. O método de *downsampling*, utilizado aqui, pode
ajudar a corrigir algum viés que esse desbalanço pode produzir nos
modelos.

O arquivo `SubmissionFile.csv` está preenchido com as previsões do
teste, no padrão proposto no arquivo original.
