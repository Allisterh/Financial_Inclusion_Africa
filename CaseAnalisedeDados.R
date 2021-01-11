library(data.table)
library(tidyverse)
library(magick)
library(caret)
library(pROC)
library(xgboost)
options(scipen = 999, digits = 3)

######################3 JUST TO SEE A THING

ag_colors <- c('#3e6dbe', '#78ca35', '#253165')


##### IMPORTAÇÃO DE DADOS

vars <- fread('data/VariableDefinitions.csv', header = T, encoding = 'UTF-8')
train <- fread('data/Train_v2.csv')
test <- fread('data/Test_v2.csv')

cols <- train[, lapply(.SD, function(x) {is.character(x)})]
cols <- setDT(as.data.frame(t(cols)), keep.rownames = T)
cols <- cols[V1==T & rn != 'uniqueid']

train[, (cols$rn) := lapply(.SD, as.factor), .SDcols = cols$rn]

##### EDA
######## SUMMARY

summary(train)

######### QUANTIDADE DE VALORES POR VARIÁVEL

train[, lapply(.SD, uniqueN)]

######### BALANÇO DA VARIÁVEL ALVO

train[, .N, bank_account][, perc := prop.table(N)][]

######### uniqueid

top_uniques <- train[, .N, uniqueid][order(-N)][1:10]
train[uniqueid %in% top_uniques$uniqueid][order(uniqueid)]
train[, min(year), country]

######### CONTAGEM POR VARIÁVEL RESPOSTA

logo <- image_read("https://logospng.org/download/agibank/logo-agibank-icon-2048.png")

p1 <- ggplot(train, aes(x = bank_account, fill = bank_account)) +
  geom_bar() +
  scale_fill_manual(values = ag_colors) + 
  labs(title = 'DESBALANÇO ENTRE AS VARIÁVEIS RESPOSTA',
       subtitle = 'CONTAGEM POR CATEGORIA',
       fill = 'CONTA NO BANCO') +
  theme_minimal() +
  theme(axis.title = element_blank())

p1
grid::grid.raster(logo, x = .98, y = 0.015, just = c('right', 'bottom'), width = unit(.3, 'inches'))

######### CONTA NO BANCO POR PAÍS

perc_country <- train[, .(n = .N), by = .(country, bank_account)][, perc := round( n / sum(n), 2), country][]

p2 <- ggplot(perc_country, aes(x = country, 
                               y =  n,
                               fill = bank_account)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(x = country, y = n, label = paste0(perc *100, '%')),
    position = position_dodge(width = 1),
    vjust = -0.5, size = 4
  ) + 
  scale_fill_manual(values = ag_colors) + 
  labs(title = 'Contagem de pessoas que possuem conta no banco por País',
       fill = 'Conta no banco') +
  theme_minimal() +
  theme(axis.title = element_blank(), 
        plot.title = element_text(size = 16))

p2
grid::grid.raster(logo, x = .98, y = 0.015, just = c('right', 'bottom'), width = unit(.3, 'inches'))


######### VALIDATION 

trainIndex <- createDataPartition(train$bank_account, p = .75, 
                                  list = FALSE, times = 1)

validation <- train[-trainIndex]
train <- train[trainIndex]

y_train <- as.factor(train$bank_account)
y_validation <- as.factor(validation$bank_account)

train <- train[, c(1, 5:length(names(train))), with=F]
validation <- validation[, c(1, 5:length(names(validation))), with=F]

cols <- train[, lapply(.SD, function(x) {is.character(x)})]
cols <- setDT(as.data.frame(t(cols)), keep.rownames = T)
cols <- cols[V1==T]

train[, (cols$rn) := lapply(.SD, as.factor), .SDcols = cols$rn]
validation[, (cols$rn) := lapply(.SD, as.factor), .SDcols = cols$rn]

####### TRAIN CONTROL

trcontrol = trainControl( method = "cv",
                          number = 5,  
                          allowParallel = TRUE,
                          verboseIter = TRUE )

####### RANDOM FOREST

mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=mtry)

rf_model <- train(x = train, 
                  y = y_train,
                  trControl = trcontrol,
                  tuneGrid = tunegrid,
                  # tuneLength = 5,
                  method = "rf")

prob_rf_train <- predict(rf_model, train, type='prob')
pred_rf_train <- predict(rf_model, train, type='raw')
pred_rf_train2 <- fifelse(prob_rf_train[,'Yes'] > 0.25, 'Yes', 'No') %>% as.factor()
(cm_rf_train <- confusionMatrix(data = pred_rf_train2, reference = y_train))

prob_rf_validation <- predict(rf_model, validation, type='prob')
pred_rf_validation <- predict(rf_model, validation, type='raw')
pred_rf_validation2 <- fifelse(prob_rf_validation[,'Yes'] > 0.25, 'Yes', 'No') %>% as.factor()
cm_rf_validation <- confusionMatrix(data = pred_rf_validation2, reference = y_validation)
rf_roc <- pROC::roc(y_validation, prob_rf_validation[,'Yes'])

plot(rf_roc)
auc(rf_roc)

train_ds <- downSample(train, y_train, yname = 'bank_account')

setDT(train_ds)[, .N, bank_account]

y_ds <- train_ds$bank_account
train_ds <- train_ds[, -'bank_account', with=F]

rf_ds_model <- train(x = train_ds, 
                     y = y_ds,
                     trControl = trcontrol,
                     tuneLength = tunegrid,
                     method = "rf")

prob_rf_ds_train <- predict(rf_ds_model, train_ds, type='prob')
pred_rf_ds_train <- predict(rf_ds_model, train_ds, type='raw')
pred_rf_ds_train2 <- fifelse(prob_rf_ds_train[,'Yes'] > 0.25, 'Yes', 'No') %>% as.factor()
(cm_rf_ds_train <- confusionMatrix(data = pred_rf_ds_train2, reference = y_ds))

prob_rf_validation_ds <- predict(rf_ds_model, validation, type='prob')
pred_rf_validation_ds <- predict(rf_ds_model, validation, type='raw')
pred_rf_validation_ds2 <- fifelse(prob_rf_validation_ds[,'Yes'] > 0.5, 'Yes', 'No') %>% as.factor()
(cm_rf_ds_validation <- confusionMatrix(data = pred_rf_validation_ds2, reference = y_validation))
rf_ds_roc <- pROC::roc(y_validation, prob_rf_validation[,'Yes'])

plot(rf_ds_roc)
auc(rf_ds_roc)

train_us <- upSample(train, y_train, yname = 'bank_account')

setDT(train_us)[, .N, bank_account]

y_us <- train_us$bank_account
train_us <- train_us[, -'bank_account', with=F]

rf_us_model <- train(x = train_us, 
                     y = y_us,
                     trControl = trcontrol,
                     tuneLength = 5,
                     method = "rf")

prob_rf_ds_train <- predict(rf_us_model, train_us, type='prob')
pred_rf_ds_train <- predict(rf_us_model, train_us, type='raw')
cm_rf_ds_train <- confusionMatrix(data = pred_rf_ds_train, reference = y_us)

prob_rf_validation <- predict(rf_us_model, validation, type='prob')
pred_rf_validation <- predict(rf_us_model, validation, type='raw')
cm_rf_us_validation <- confusionMatrix(data = pred_rf_validation, reference = y_validation)
rf_ds_roc <- pROC::roc(y_validation, prob_rf_validation[,'Yes'])

plot(rf_ds_roc)
auc(rf_ds_roc)











dmy_train <- dummyVars('~.', data = train)
train_matrix <- data.table(predict(dmy_train, newdata = train))

dmy_validation <- dummyVars('~.', data = validation)
validation_matrix <- data.table(predict(dmy_validation, newdata = validation))


################### MODELO

set.seed(100)  # For reproducibility

train_matrix <- train_matrix %>% as.matrix() %>% xgb.DMatrix()
validation_matrix <- validation_matrix %>% as.matrix() %>% xgb.DMatrix()



xgbGrid <- expand.grid(nrounds = c(35,50),  # this is n_estimators in the python code above
                       max_depth = c(10, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 2),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

xgb_model <- train( x = train_matrix, 
                    y = y_train,
                    trControl = xgb_trcontrol,
                    tuneGrid = xgbGrid,
                    # tuneLength = 5,
                    method = "xgbTree")

prob_train <- predict(xgb_model, train_matrix, type="prob")
pred_train <- predict(xgb_model, train_matrix, type="raw")

cm <- confusionMatrix(data = pred_train, reference = y_train)

prob_validation <- predict(xgb_model, validation_matrix, type="prob")
pred_validation <- predict(xgb_model, validation_matrix, type="raw")

cm_validation <- confusionMatrix(data = pred_validation, reference = y_validation)
xgb_roc <- pROC::roc(y_validation, prob_validation[,'Yes'])

plot(xgb_roc)
auc(xgb_roc)



dmy_train_ds <- dummyVars('~.', data = train_downsample)
train_matrix_ds <- data.table(predict(dmy_train_ds, newdata = train_downsample))

train_matrix_ds <- train_matrix_ds %>% as.matrix() %>% xgb.DMatrix()

xgb_model_downsample <- train( x = train_matrix_ds, 
                               y = y_ds,
                               trControl = xgb_trcontrol,
                               tuneGrid = xgbGrid,
                               method = "xgbTree")

prob_train_ds <- predict(xgb_model_downsample, train_matrix_ds, type="prob")
pred_train_ds <- predict(xgb_model_downsample, train_matrix_ds, type="raw")

cm_ds <- confusionMatrix(data = pred_train_ds, reference = y_ds)

prob_validation_ds <- predict(xgb_model_downsample, validation_matrix, type="prob")
pred_validation_ds <- predict(xgb_model_downsample, validation_matrix, type="raw")

cm_validation_ds <- confusionMatrix(data = pred_validation_ds, reference = y_validation)
xgb_roc_ds <- pROC::roc(y_validation, prob_validation_ds[,'Yes'])

plot(xgb_roc_ds)
auc(xgb_roc_ds)







rf_model <- train(x = train, 
                  y = y_train,
                  trControl = xgb_trcontrol,
                  tuneLength = 5,
                  method = "rf")

prob_rf_test <- predict(rf_model, train, type='prob')
pred_rf_test <- predict(rf_model, train, type='raw')
cm_rf_test <- confusionMatrix(data = pred_rf_test, reference = y_train)

prob_rf_validation <- predict(rf_model, validation, type='prob')
pred_rf_validation <- predict(rf_model, validation, type='raw')
cm_rf_validation <- confusionMatrix(data = pred_rf_validation, reference = y_validation)
rf_roc <- pROC::roc(y_validation, prob_rf_validation[,'Yes'])

plot(rf_roc)
auc(rf_roc)


glm_model <- train( x = train, 
                    y = y_train,
                    trControl = xgb_trcontrol,
                    tuneLength = 5,
                    method = "glm")

pred_glm_test <- predict(glm_model, train, type='raw')
cm_glm_test <- confusionMatrix(data = pred_glm_test, reference = y_train)

pred_glm_validation <- predict(glm_model, validation, type='raw')
cm_glm_validation <- confusionMatrix(data = pred_glm_validation, reference = y_validation)


