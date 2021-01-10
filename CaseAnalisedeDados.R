library(data.table)
library(tidyverse)
library(caret)
library(broom)
library(xgboost)
library(pROC)
options(scipen = 999, digits = 3)

######################3 JUST TO SEE A THING



##### IMPORTAÇÃO DE DADOS

vars <- fread('data/VariableDefinitions.csv', header = T, encoding = 'UTF-8')
train <- fread('data/Train_v2.csv')
test <- fread('data/Test_v2.csv')


##### EDA
######## GLIMPSE

summary(train)

######### QUANTIDADE DE VALORES POR VARIÁVEL

train[, lapply(.SD, uniqueN)]

######### BALANÇO DA VARIÁVEL ALVO

train[, .N, bank_account][, perc := prop.table(N)][]

######### uniqueid

top_uniques <- train[, .N, uniqueid][order(-N)][1:10]
train[uniqueid %in% top_uniques$uniqueid][order(uniqueid)]
train[, min(year), country]

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

dmy_train <- dummyVars('~.', data = train)
train_matrix <- data.table(predict(dmy_train, newdata = train))

dmy_validation <- dummyVars('~.', data = validation)
validation_matrix <- data.table(predict(dmy_validation, newdata = validation))


################### MODELO

set.seed(100)  # For reproducibility

train_matrix <- train_matrix %>% as.matrix() %>% xgb.DMatrix()
validation_matrix <- validation_matrix %>% as.matrix() %>% xgb.DMatrix()

xgb_trcontrol = trainControl( method = "cv",
                              number = 5,  
                              allowParallel = TRUE,
                              verboseIter = TRUE )

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


