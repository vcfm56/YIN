heart<- read.csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv", header=TRUE)

#data summary
install.packages("skimr")
library("skimr")
skimr::skim(heart)
install.packages("DataExplorer")
library("DataExplorer")
DataExplorer::plot_bar(heart, ncol = 2)
DataExplorer::plot_histogram(heart, ncol = 2)

#select data
install.packages("tidyverse")
install.packages("ggplot2")
library("tidyverse")
library("ggplot2")

heart <- heart %>%
  select(-time)

heart <- heart %>%
  filter(creatinine_phosphokinase < 3000)

heart <- heart %>%
  filter(serum_creatinine < 5)
ggplot(heart,
       aes(x = age, y =serum_creatinine )) +
  geom_point()
ggplot(heart,
       aes(x = age, y =creatinine_phosphokinase )) +
  geom_point()

DataExplorer::plot_boxplot(heart, by = "fatal_mi", ncol = 3)

# CART model fit
install.packages("data.table")
install.packages("mlr3verse")
library("data.table")
library("mlr3verse")

set.seed(200) # set seed for reproducibility
heart$fatal_mi <- as.factor(heart$fatal_mi)
heart_task <- TaskClassif$new(id = "live",
                              backend = heart,
                              target = "fatal_mi"
)
#nested cross validation
install.packages("rsample")
library("rsample")
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_naive_bayes <- lrn("classif.naive_bayes", predict_type = "prob")

set.seed(150)
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

#choose cost penalty
res_cart_cv <- resample(heart_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.031)

#see our performance
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    #lrn_cart,
                    lrn_naive_bayes,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

#see trees
trees <- res$resample_result(3)
tree1 <- trees$learners[[5]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.04)
text(tree1_rpart, use.n = TRUE, cex = 0.7)

#see roc 
install.packages("precrec")
library('precrec')
lrn_cart_cp$train(heart_task)
pred_cart_cp = lrn_cart_cp$predict(heart_task)
autoplot(pred_cart_cp,type = 'roc')




library("rsample")
set.seed(212) 
#get the training
heart_split <- initial_split(heart)
heart_train <- training(heart_split)
#split the training into validate and test
heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)

install.packages("recipes")
library("recipes")

cake <- recipe(fatal_mi ~ ., data = heart) %>%
  step_meanimpute(all_numeric()) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_unknown(all_nominal(), -all_outcomes()) %>%  
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  prep(training = heart_train) 

heart_train_final <- bake(cake, new_data = heart_train) 
heart_validate_final <- bake(cake, new_data = heart_validate)
heart_test_final <- bake(cake, new_data = heart_test) 

install.packages("tensorflow")
install_keras()# may cause error

library("keras")
heart_train_x <- heart_train_final %>%
  select(-starts_with("fatal_mi_")) %>%
  as.matrix()
heart_train_y <- heart_train_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_validate_x <- heart_validate_final %>%
  select(-starts_with("fatal_mi_")) %>%
  as.matrix()
heart_validate_y <- heart_validate_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

heart_test_x <- heart_test_final %>%
  select(-starts_with("fatal_mi_")) %>%
  as.matrix()
heart_test_y <- heart_test_final %>%
  select(fatal_mi_X0) %>%
  as.matrix()

#start deep learning
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  heart_train_x, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_x, heart_validate_y)
)
